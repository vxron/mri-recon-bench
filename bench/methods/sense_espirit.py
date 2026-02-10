from __future__ import annotations
from bench.utils import Configs, MethodConfigs, ReconMethod
import numpy as np
import tracemalloc
import time
import sigpy as sp
from sigpy.mri.app import EspiritCalib, SenseRecon

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
    espirit_grid = tuple(cfg.get("calib", [24,24]))
    espirit_thresh = float(cfg.get("thresh", 0.02))
    espirit_kernel_width = int(cfg.get("kernel_width", 6))
    device = cfg.get("sigpy_device", sp.cpu_device)
    debug = bool(cfg.get("debug_verify", False))

    # Convert kspace 
    ksp = kspace.astype(np.complex64, copy=False)
    shift_dc = methodCfg.get("shift_DC", False)
    if shift_dc:
        # implement ifftshift
        pass #TODO

    # Run Espirit
    maps = EspiritCalib(
        ksp, 
        calib=espirit_grid, 
        thresh=espirit_thresh, 
        kernel_width=espirit_kernel_width, 
        device=device).run()

    # Make sure it worked as intended
    if(maps.shape != ksp.shape):
        raise ValueError(f"Expected maps shaped like {ksp.shape}. Got {maps.shape}") 
    if debug:
        pass
        #todo: debug plot (magnitude maps of maps)
    
    # reusable resources
    out_buf = np.empty((H,W), dtype=np.float32) # to avoid allocating a new array for mag each run of sense
    tmp_buf = np.empty((H,W), dtype=np.float64) # in case we need storage for magnitude computations

    state = {
        "maps": maps, # coil maps
        "device": device,
        "out_buf": out_buf,
        "tmp_buf": tmp_buf,
        "ksp": ksp, # cleaned kspace
    }
    methodCfg.state = state

def run_sense_solver(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    """
    Runs SENSE reconstruction to solve for image x
    Reuses sensitivity maps calculated once at setup
    """
    
    # (1) use config defaults
    cfg = methodCfg.sense_espirit
    device = cfg.get("sigpy_device", sp.cpu_device)
    debug = bool(cfg.get("debug_verify", False))
    sense_max_iter = int(cfg.get("max_iter", 30))
    lambda_reg = cfg.get("lambda", 0.0)

    gt = methodCfg.ground_truth_im
    outType = methodCfg.im_bit_depth

    # (2) state elements
    if methodCfg.state["ksp"] is None:
        ksp = kspace.astype(np.complex64, copy=False)
        shift_dc = methodCfg.get("shift_DC", False)
        if shift_dc:
            # implement ifftshift
            pass #TODO
    else:
        kspace = methodCfg.state["ksp"]
    
    maps = methodCfg.state["maps"]

    # (3) RUN SENSE
    # use prealloc buffer
    im = SenseRecon(
        kspace,maps,lambda_reg,
        device=device,max_iter=sense_max_iter).run()
    
    if methodCfg.state["out_buf"] is None:
        if outType == "float32":
            out = abs(im).astype(np.float32, copy=False) #  mag only (no im)
        elif outType == "float64":
            out = abs(im).astype(np.float64, copy=False)
    else:
        methodCfg.state["out_buf"] = abs(im).astype(np.float32, copy=False)

    # (4) Verification against ground truth image 
    if debug:
        if gt is None:
            print("[sense_espirit] debug_verify=True but cfg doesn't have gt image (skipping).")
        else:
            pred_im = out if methodCfg.state["out_buf"] is None else methodCfg.state["out_buf"]
            gt = np.asarray(gt)
            if gt.shape != pred_im.shape:
                print(f"[sense_espirit] GT shape {gt.shape} != pred {pred_im.shape} (skipping metrics).")
            else:
                # High-level similarity checks:
                # - correlation: invariant to global scaling
                # - relative L2 error: sensitive to scaling/shift differences
                pred = out.astype(np.float64, copy=False) if methodCfg.state["out_buf"] else out.astype(np.float64, copy=False)
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

                print(f"[baseline_ifft] verify: corr={corr:.4f}, rel_l2={rel_l2:.4f}")

    if methodCfg.state is None:
        return out
    else:
        return methodCfg.state["out_buf"]

def cleanup(methodCfg: MethodConfigs):
    # clear state
    methodCfg.state.clear()
