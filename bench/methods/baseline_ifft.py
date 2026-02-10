# bench/methods/rss_ifft.py
"""
RSS + IFFT baseline (standard Cartesian multi-coil MRI sanity check)

Input:
  multi-coil kspace: complex ndarray, shape (C, H, W)
Output:
  rss: float32 ndarray, shape (H, W)

Algorithm:
  1) 2D inverse FFT on each coil image: img_c = IFFT2(kspace_c)
  2) coil combine with Root-Sum-of-Squares:
        rss(x,y) = sqrt( sum_c |img_c(x,y)|^2 )

Notes on k-space centering:
  Some datasets store k-space with DC at the center of the array.
  In that case ifftshift must be applied before ifft2.
  This method supports both modes via use_ifftshift.
"""

from __future__ import annotations
import numpy as np
from bench.utils import Configs, MethodConfigs, ReconMethod
import tracemalloc
import time

# NOTE: NumPy's ffts still allocate internally, so baseline won't ever get to "zero allocations per frame" the way a real C++/CUDA pipeline could
# What we CAN reuse: rss_accum buffer for multiple coils avgd tog (H,W) float64, output im buffer (H,W) float32/float64
# What will still be allocated by NumPy FFT: img_c = np.fft.ifft2 (allocates fresh arrays internally), shifted arrays from fftshift/ifftshift

def preallocate_buffers(kspace: np.ndarray, methodCfg: MethodConfigs) -> tuple[int,float]:
    # preallocate arrays we can reuse
    H = kspace.shape[1]
    W = kspace.shape[2]
    out_dtype = methodCfg.im_bit_depth
    
    t0 = time.perf_counter()
    tracemalloc.start()
    
    rss_accum = np.empty((H, W), dtype=np.float64)
    if out_dtype == "float32":
        out = np.empty((H,W), dtype=np.float32)
    else:
        out = np.empty((H,W), dtype=np.float64)
    tmp = np.empty((H, W), dtype=np.float64) # for RSS calculation

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0

    state = {
        "out": out,
        "rss_accum": rss_accum,
        "tmp": tmp,
    }
    # fill methodCfg object state to feed to baseline_ifft (Python defaults pass by ref)
    methodCfg.state = state

    return peak, time_elapsed

def baseline_ifft(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    """
    Args:
      kspace: complex ndarray of shape (C, H, W)
      cfg: Configs object. Expected optional fields:
           - use_ifftshift: bool (default True)
           - norm: str in {"none","ortho"} (default "ortho")
           - im_bit_depth: str in {"float32","float64"} (default "float32")
           - debug_verify: bool (default False)
           - ground_truth_im: optional ground truth image (H,W) to compare against in debug
    Returns:
      rss image (H, W) float
    """
    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace shape (C,H,W). Got {kspace.shape}")
    if not np.iscomplexobj(kspace):
        raise TypeError("kspace must be complex. If dataset stores real/imag separately, combine first.")

    # Config defaults
    cfg = methodCfg.baseline_ifft
    use_ifftshift = bool(cfg.get("use_ifftshift", True))
    norm = cfg.get("norm", "ortho")
    out_dtype = methodCfg.im_bit_depth
    debug_verify = bool(cfg.get("debug_verify", False))
    gt = methodCfg.ground_truth_im

    # (2) IFFT per coil
    k = np.fft.ifftshift(kspace, axes=(-2, -1)) if use_ifftshift else kspace
    # Compute coil images: shape (C,H,W), complex
    img_c = np.fft.ifft2(k, axes=(-2, -1), norm=norm)
    img_c = np.fft.fftshift(img_c, axes=(-2, -1))

    # (3) rss = sqrt(sum_c |img_c|^2)
    state = methodCfg.state
    if state is None:
        rss = np.sqrt(np.sum(np.abs(img_c) ** 2, axis=0)) # NumPy allocations
    else:
        # Use preallocated buffers from setup
        out=state["out"]
        rss=state["rss_accum"]
        tmp=state["tmp"]
        rss.fill(0)
        for c in range(kspace.shape[0]): # this is C (num coils)
            np.abs(img_c[c], out=tmp)
            np.square(tmp, out=tmp)
            rss += tmp
        np.sqrt(rss, out=rss)
        
    # (4) Cast output
    if state is None: # numpy astype allocations
        if out_dtype == "float32":
            rss = rss.astype(np.float32, copy=False)
        elif out_dtype == "float64":
            rss = rss.astype(np.float64, copy=False)
        else:
            pass
    else:
        if out.dtype == rss.dtype:
            out[:] = rss # put rss directly into out buffer
        else:
            np.copyto(out, rss, casting="unsafe") # avoids allocation

    # (5) Verification against ground truth image if available/allowed
    if debug_verify:
        if gt is None:
            print("[baseline_ifft] debug_verify=True but cfg doesn't have gt image (skipping).")
        else:
            pred_im = rss if state is None else out
            gt = np.asarray(gt)
            if gt.shape != pred_im.shape:
                print(f"[baseline_ifft] GT shape {gt.shape} != pred {pred_im.shape} (skipping metrics).")
            else:
                # High-level similarity checks:
                # - correlation: invariant to global scaling
                # - relative L2 error: sensitive to scaling/shift differences
                pred = out.astype(np.float64, copy=False) if state else rss.astype(np.float64, copy=False)
                gt64 = gt.astype(np.float64, copy=False)

                # avoid division by zero
                eps = 1e-12
                pred_norm = np.linalg.norm(pred) + eps
                gt_norm = np.linalg.norm(gt64) + eps

                rel_l2 = np.linalg.norm(pred - gt64) / gt_norm

                # Pearson correlation (flattened)
                p = pred.ravel()
                g = gt64.ravel()
                p = p - p.mean()
                g = g - g.mean()
                corr = float((p @ g) / (np.linalg.norm(p) * np.linalg.norm(g) + eps))

                print(f"[baseline_ifft] verify: corr={corr:.4f}, rel_l2={rel_l2:.4f}, "
                      f"use_ifftshift={use_ifftshift}, norm={norm}")

    if state is None:
        return rss
    else:
        return out

def cleanup(methodCfg: MethodConfigs):
    # clear state
    methodCfg.state.clear()