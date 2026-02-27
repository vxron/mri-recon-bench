from __future__ import annotations
import numpy as np
from bench.utils import Configs, MethodConfigs, ReconMethod
import tracemalloc
import time
import pygrappa

def preallocate_buffers(kspace: np.ndarray, methodCfg: MethodConfigs) -> tuple[int,float]:
    # read config
    cfg = methodCfg.grappa
    mask = methodCfg.undersampling_mask
    calib = int(mask.get("acs",32))
    ksp_dtype = cfg.get("ksp_dtype", np.complex64)
    out_dtype = methodCfg.im_bit_depth
    H = kspace.shape[1]
    W = kspace.shape[2]
    R = int(mask.get("R", 2))
    simulate_undersampling = bool(cfg.get("simulate_underampling", True))

    t0 = time.perf_counter()
    tracemalloc.start()

    ksp, _ = preprocess_kspace(kspace, out_dtype=ksp_dtype)
    # store normalization amount (max in kspace) so the solver can normalize consistently
    ksp_scale = float(np.abs(ksp).max()) + 1e-12
    ksp_norm = ksp / ksp_scale

    # extract the ACS we will use from normalized, preprocessed kspace (before the mask is built)
    acs_lo = H // 2 - calib // 2
    acs_hi = H // 2 + calib // 2
    acs_region = ksp_norm[:, acs_lo:acs_hi, :].copy()  # (C, calib, W)

    # simulate undersampling (like with cs)
    if simulate_undersampling:
        mask = methodCfg.undersampling_mask["mask2d"]
        if mask is None:
            raise RuntimeError("No shared mask found - was it built in run_bench.py?")

    # reusable buffers for rss comps
    rss_accum = np.empty((H, W), dtype=np.float64)
    tmp = np.empty((H, W), dtype=np.float64) # for RSS calc

    if out_dtype == "float32":
        out = np.empty((H,W), dtype=np.float32)
    else:
        out = np.empty((H,W), dtype=np.float64)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0

    state = {
        "out_buf": out,
        "rss_accum": rss_accum,
        "tmp": tmp,
        "ksp_scale": ksp_scale,
        "acs": acs_region,
        "mask2d": mask if simulate_undersampling else None
    }
    methodCfg.state = state

    return peak, time_elapsed


def preprocess_kspace(kspace: np.ndarray, *, out_dtype=np.complex64) -> np.ndarray:
    ksp = kspace.astype(out_dtype, copy=False)
    # automatically handle dc placement/shifting:
    # compare energy in center vs corner to make sure it's centered (Sense/Espirit API expects centered DC)
    mag = np.abs(ksp).sum(axis=0)
    H,W = mag.shape
    cy,cx = H//2, W//2
    w = max(4, H//16)
    center = mag[cy-w:cy+w, cx-w:cx+w].mean()
    corner = mag[:2*w, :2*w].mean()
    in_is_centered = center > corner
    return (ksp if in_is_centered else np.fft.fftshift(ksp, axes=(-2,-1))), in_is_centered


def run_grappa(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    # (1) read configs
    cfg = methodCfg.grappa
    mask = methodCfg.undersampling_mask
    kernel_size = cfg.get("kernel_size", (5,5))
    coil_axis = int(cfg.get("coil_axis", 0))
    calib = int(mask.get("acs",32))
    ksp_type = cfg.get("ksp_dtype", np.complex64)
    acs = methodCfg.state["acs"]
    lambda_g = float(cfg.get("lambda", 1e-3))
    gt = methodCfg.ground_truth_im
    outType = methodCfg.im_bit_depth
    debug = bool(cfg.get("debug_verify", False))
    simulate_undersampling = bool(cfg.get("simulate_undersampling", True))
    H = kspace.shape[1]
    W = kspace.shape[2]

    # (2) PREPROCESS
    ksp, _ = preprocess_kspace(kspace, out_dtype=ksp_type)
    ksp_scale = methodCfg.state.get("ksp_scale", 1.0)
    ksp = ksp / ksp_scale

    if "mask2d" not in methodCfg.state and simulate_undersampling:
        # no setup was run -> need to build mask on the fly
        methodCfg.state["mask2d"] = methodCfg.undersampling_mask["mask2d"]

    if simulate_undersampling:
        mask = methodCfg.state["mask2d"]
        w = mask[None, ...]                    # (1,H,W) broadcastable
        # apply sampling for CS
        ksp = ksp * w
    
    t0 = time.perf_counter()
    tracemalloc.start()

    # (3) call to pygrappa -> returns filled k-space of shape (C,H,W)
    # PYGRAPPA EXPECTS (H,W,C) -> MOVE AXES
    ksp_hwc = np.moveaxis(ksp, 0, -1)      # (H, W, C)
    acs_hwc = np.moveaxis(acs, 0, -1)      # (calib, W, C)
    solved_ksp = pygrappa.grappa(ksp_hwc, acs_hwc, kernel_size=kernel_size, coil_axis=-1,lamda=lambda_g)
    solved_ksp = np.moveaxis(solved_ksp, -1, 0)  # (H,W,C) -> (C,H,W) so IFFT takes the right {H,W}

    # (4) IFFT filled k-space to recover im
    img_c = np.fft.ifftshift(solved_ksp, axes=(-2, -1))  # move DC to corner before ifft2   
    img_c = np.fft.ifft2(img_c, axes=(-2, -1))           # ifft expects DC at [0,0]
    img_c = np.fft.fftshift(img_c, axes=(-2, -1))        # shift result so DC is centered for display

    # (5) RSS combine across coils
    if methodCfg.state is None:
        rss = np.sqrt(np.sum(np.abs(img_c) ** 2, axis=0)) # NumPy allocations
    else:
        # Use preallocated buffers from setup
        rss = methodCfg.state["rss_accum"]
        tmp = methodCfg.state["tmp"]
        rss.fill(0)
        for c in range(kspace.shape[0]): # this is C (num coils)
            np.abs(img_c[c], out=tmp)
            np.square(tmp, out=tmp)
            rss += tmp
        np.sqrt(rss, out=rss)

    # (6) fill output buffer
    if methodCfg.state is None: # numpy astype allocations
        if out_dtype == "float32":
            rss = rss.astype(np.float32, copy=False)
        elif out_dtype == "float64":
            rss = rss.astype(np.float64, copy=False)
        else:
            pass
    else:
        out = methodCfg.state["out_buf"]
        if out.dtype == rss.dtype:
            out[:] = rss # put rss directly into out buffer
        else:
            np.copyto(out, rss, casting="unsafe") # avoids allocation

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0

    return out, peak, time_elapsed


def cleanup(methodCfg: MethodConfigs):
    # clear state
    if methodCfg.state is not None:
        methodCfg.state.clear()