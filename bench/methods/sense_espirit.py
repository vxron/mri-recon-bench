from __future__ import annotations
from bench.utils import Configs, MethodConfigs, ReconMethod, RESULTS
import numpy as np
import tracemalloc
import time
import sigpy as sp
from sigpy.mri.app import EspiritCalib, SenseRecon
import matplotlib.pyplot as plt
from pathlib import Path


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


def debug_plot_espirit_maps(methodCfg: MethodConfigs, maps, max_coils=8):
    """
    Plot magnitude of ESPIRiT sensitivity maps for a couple coils only
    maps: complex ndarray (C, H, W)
    """
    C = maps.shape[0]
    n_show = min(C, max_coils)

    fig, axes = plt.subplots(1, n_show, figsize=(3*n_show, 3))
    if n_show == 1:
        axes = [axes]

    for c in range(n_show):
        mag = np.abs(maps[c])                       # 1
        mag /= mag.max() + 1e-12                    # 2: normalize for vis
        axes[c].imshow(mag, cmap="gray")            # 3
        axes[c].set_title(f"Coil {c}")              # 4
        axes[c].axis("off")                         # 5

    out_dir = RESULTS / "images" / "debug" / f"{methodCfg.sense_espirit['calib']}" # sort by calib size in folders
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.suptitle("ESPIRiT Sensitivity Map Magnitudes")
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_map.png", dpi=150)
    plt.close()


def setup_and_espirit(kspace: np.ndarray, methodCfg: MethodConfigs):
    """
    Runs espirit algorithm to estimate coil sensitivity maps
    kspace shape: (C,H,W)
    dtype: complex64 (speed/memory efficient) or complex128 (more accurate/heavier)
    """
    # Data integrity assumptions
    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace shape (C,H,W). Got {kspace.shape}")
    if not np.iscomplexobj(kspace):
        raise TypeError("kspace must be complex. If dataset stores real/imag separately, combine first.")
    
    H = kspace.shape[1]
    W = kspace.shape[2]

    cfg = methodCfg.sense_espirit
    debug = bool(cfg.get("debug_verify", False))
    espirit_grid = int(cfg.get("calib", 24))
    espirit_thresh = float(cfg.get("thresh", 0.02))
    espirit_kernel_width = int(cfg.get("kernel_width", 6))
    device = cfg.get("sigpy_device", sp.Device(-1))
    ksp_type = cfg.get("ksp_dtype", np.complex64)

    # don't include preprocessing in time/memory computations
    ksp, was_centered = preprocess_kspace(kspace, out_dtype=ksp_type)
    print("input centered?", was_centered)

    # =========================== TIME/MEMORY TRACKING STARTS =======================
    t0 = time.perf_counter()
    tracemalloc.start()

    # Run Espirit
    maps = EspiritCalib(
        ksp,
        calib_width=espirit_grid, 
        thresh=espirit_thresh, 
        kernel_width=espirit_kernel_width, 
        device=device).run()

    # Make sure it worked as intended
    if debug:
        if(maps.shape != ksp.shape):
            raise ValueError(f"Expected maps shaped like {ksp.shape}. Got {maps.shape}")
        debug_plot_espirit_maps(methodCfg, maps)

    # reusable resources
    out_buf = np.empty((H,W), dtype=np.float32) # to avoid allocating a new array for mag each run of sense

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0
    # =========================== TIME/MEMORY TRACKING ENDS =======================

    state = {
        "maps": maps, # coil maps
        "device": device,
        "out_buf": out_buf,
    }
    methodCfg.state = state
    
    return peak, time_elapsed


def run_sense_solver(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    """
    Runs SENSE reconstruction to solve for image x
    Reuses sensitivity maps calculated once at setup
    """
    # Data integrity assumptions
    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace shape (C,H,W). Got {kspace.shape}")
    if not np.iscomplexobj(kspace):
        raise TypeError("kspace must be complex. If dataset stores real/imag separately, combine first.")

    # (1) use config defaults
    cfg = methodCfg.sense_espirit
    device = cfg.get("sigpy_device", sp.Device(-1))
    sense_max_iter = int(cfg.get("max_iter", 30))
    lambda_reg = cfg.get("lambda", 0.0)
    ksp_type = cfg.get("ksp_dtype", np.complex64)
    gt = methodCfg.ground_truth_im
    outType = methodCfg.im_bit_depth
    debug = bool(cfg.get("debug_verify", False))

    # (2) preprocess
    ksp, _ = preprocess_kspace(kspace, out_dtype=ksp_type)

    # =========================== TIME/MEMORY TRACKING STARTS ===================================
    tracemalloc.start()
    t0 = time.perf_counter()

    if "maps" not in methodCfg.state:
        # no setup was run -> need to build maps on the fly
        espirit_grid = int(cfg.get("calib", 24,))
        methodCfg.state["maps"] = EspiritCalib(
            ksp,
            calib_width=espirit_grid,
            thresh=float(cfg.get("thresh", 0.02)),
            kernel_width=int(cfg.get("kernel_width", 6)),
            device=device
        ).run()
    
    maps = methodCfg.state["maps"]

    # (3) RUN SENSE
    # Create mask to ensure no missing NaNs, 0s in kspace to SENSE
    kabs = np.abs(ksp).sum(axis=0)
    thr = 1e-12 * np.max(kabs)      # start extremely low
    mask = (kabs > thr).astype(np.float32)
    w = mask[None, ...]
    print("acquired fraction (thr):", mask.mean())

    # use prealloc buffer
    im = SenseRecon(
        ksp,maps,lambda_reg, weights=w,
        device=device,max_iter=sense_max_iter).run()
    
    if methodCfg.state["out_buf"] is None:
        if outType == "float32":
            out = np.abs(im).astype(np.float32, copy=False) #  mag only (no im)
        elif outType == "float64":
            out = np.abs(im).astype(np.float64, copy=False)
    else:
        # take reference to prealloc buffer object
        out_buf = methodCfg.state["out_buf"]
        # compute the magnitude |im| elementwise and store directly into out_buf using Numpy out=
        np.abs(im, out=out_buf)

    time_elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # =========================== TIME/MEMORY TRACKING ENDS =======================================

    # (4) Verification against ground truth image 
    if debug:
        if gt is None:
            print("[sense_espirit] debug_verify=True but cfg doesn't have gt image (skipping).")
        else:
            pred_im = out if methodCfg.state.get("out_buf") is None else methodCfg.state.get("out_buf")
            gt = np.asarray(gt)
            if gt.shape != pred_im.shape:
                print(f"[sense_espirit] GT shape {gt.shape} != pred {pred_im.shape} (skipping metrics).")
            else:
                # High-level similarity checks:
                # - correlation: invariant to global scaling
                # - relative L2 error: sensitive to scaling/shift differences
                pred = out.astype(np.float64, copy=False) if methodCfg.state.get("out_buf") is None else methodCfg.state.get("out_buf").astype(np.float64, copy=False)
                gt64 = gt.astype(np.float64, copy=False)
                eps = 1e-12
                gt_norm = np.linalg.norm(gt64) + eps
                rel_l2 = np.linalg.norm(pred - gt64) / gt_norm
                # Pearson correlation (flattened)
                p = pred.ravel()
                g = gt64.ravel()
                p = p - p.mean()
                g = g - g.mean()
                corr = float((p @ g) / (np.linalg.norm(p) * np.linalg.norm(g) + eps))

                print(f"[sense_espirit] verify: corr={corr:.4f}, rel_l2={rel_l2:.4f}")

    if methodCfg.state["out_buf"] is None:
        return out, peak, time_elapsed
    else:
        return out_buf, peak, time_elapsed


def cleanup(methodCfg: MethodConfigs):
    # clear state if not none
    if methodCfg.state is not None:
        methodCfg.state.clear()
