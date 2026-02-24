from __future__ import annotations
from bench.utils import Configs, MethodConfigs, ReconMethod, RESULTS, make_vds_ky_mask
import numpy as np
import tracemalloc
import time
import sigpy as sp
from sigpy.mri.app import EspiritCalib, SenseRecon
import matplotlib.pyplot as plt
from pathlib import Path
from sigpy.mri.app import L1WaveletRecon

# This file also includes SigPy compressed sensing because they work similarly.

def infer_cartesian_sampling(ksp: np.ndarray, *, eps_rel: float = 1e-8):
    """
    Infer a Cartesian sampling mask from k-space energy.
    Assumes missing samples are ~0.
    Returns:
      mask2d: (H,W) float32
      R_eff:  float  (H / acquired_ky_lines)
      acs_eff: int   (# contiguous acquired ky lines around center)
      meta: dict with thresholds and line energy
    """
    # ksp shape: (C,H,W)
    kabs = np.abs(ksp).sum(axis=0)            # (H,W)
    line_energy = kabs.sum(axis=1)            # (H,) sum over kx per ky

    mx = float(line_energy.max() + 1e-30)
    thr = eps_rel * mx
    ky_acq = line_energy > thr                # (H,) bool

    acquired_ky = int(ky_acq.sum())
    H = ky_acq.shape[0]
    R_eff = float(H / max(acquired_ky, 1))

    # ACS: largest contiguous acquired region around center ky
    cy = H // 2
    lo = cy
    hi = cy
    if ky_acq[cy]:
        while lo - 1 >= 0 and ky_acq[lo - 1]:
            lo -= 1
        while hi + 1 < H and ky_acq[hi + 1]:
            hi += 1
        acs_eff = hi - lo + 1
    else:
        acs_eff = 0

    # Build full (H,W) mask
    mask2d = ky_acq[:, None].astype(np.float32) * np.ones((1, kabs.shape[1]), np.float32)

    meta = {
        "thr": thr,
        "eps_rel": eps_rel,
        "acquired_ky": acquired_ky,
        "ky_acq_frac": float(acquired_ky / H),
    }
    return mask2d, R_eff, acs_eff, meta


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


def setup_and_espirit(kspace: np.ndarray, methodCfg: MethodConfigs, **kwargs):
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

    # passed a kwarg
    curr_method = kwargs.get("curr_method")
    if curr_method == ReconMethod.SENSE:
        cfg = methodCfg.sense_espirit
    else:
        cfg = methodCfg.cs_l1_wavelet
        R = int(cfg.get("R", 4))
        acs = int(cfg.get("acs", 24))
        simulate_undersampling = bool(cfg.get("simulate_undersampling", True))

    debug = bool(cfg.get("debug_verify", False))
    espirit_grid = int(cfg.get("calib", 24))
    espirit_thresh = float(cfg.get("thresh", 0.02))
    espirit_kernel_width = int(cfg.get("kernel_width", 6))
    device = cfg.get("sigpy_device", sp.Device(-1))
    ksp_type = cfg.get("ksp_dtype", np.complex64)

    # don't include preprocessing in time/memory computations
    ksp, was_centered = preprocess_kspace(kspace, out_dtype=ksp_type)
    print("input centered?", was_centered)
    # is our dataset actually undersampled (R_eff >> 1) or fully sampled (R_eff ~1) and whether ACS region looks reasonable
    mask2d_inferred, R_eff, acs_eff, meta = infer_cartesian_sampling(ksp)
    print(f"[setup] inferred sampling: R_eff={R_eff:.2f}x, acs_eff={acs_eff}px, "
          f"acquired_ky={meta['acquired_ky']} ({meta['ky_acq_frac']*100:.1f}%)")
    if meta["ky_acq_frac"] > 0.95:
        print("[setup] WARNING: k-space appears fully sampled — undersampling may not be active")

    # normalize for ESPIRiT the same way we do for SENSE/CS solving so that we don't have a mismatch in coil sensitivty mapping images
    ksp_scale = float(np.abs(ksp).max()) + 1e-12
    ksp_norm = ksp / ksp_scale

    # =========================== TIME/MEMORY TRACKING STARTS =======================
    t0 = time.perf_counter()
    tracemalloc.start()

    # Run Espirit
    maps = EspiritCalib(
        ksp_norm,
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

    # if it's L1 wavelet, we can make sampling mask ahead of time...
    if curr_method == ReconMethod.CS_L1 and simulate_undersampling:
        mask2d = make_vds_ky_mask(H, W, R=R, acs=acs)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0
    # =========================== TIME/MEMORY TRACKING ENDS =======================

    state = {
        "maps": maps, # coil maps
        "ksp_scale": ksp_scale, # normalization amount of kspace for consistency btwn espirit & solver
        "device": device,
        "out_buf": out_buf,
        "mask2d": mask2d if curr_method == ReconMethod.CS_L1 and simulate_undersampling else None
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
    ksp_scale = methodCfg.state.get("ksp_scale", 1.0)
    ksp = ksp / ksp_scale

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
                pred64 = pred.astype(np.float64, copy=False) if methodCfg.state.get("out_buf") is None else methodCfg.state.get("out_buf").astype(np.float64, copy=False)
                gt64 = gt.astype(np.float64, copy=False)
                # normalize by GT max so rel_l2 is scale-invariant
                gt_max = gt64.max() + 1e-12
                pred = pred64 / pred64.max() * gt_max
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
    

def run_l1wavelet_solver(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    # Data integrity assumptions
    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace shape (C,H,W). Got {kspace.shape}")
    if not np.iscomplexobj(kspace):
        raise TypeError("kspace must be complex. If dataset stores real/imag separately, combine first.")

    # (1) use config defaults
    cfg = methodCfg.cs_l1_wavelet
    device = cfg.get("sigpy_device", sp.Device(-1))
    wavelet_max_iter = int(cfg.get("max_iter", 30))
    lambda_reg = cfg.get("lambda", 1e-3)
    ksp_type = cfg.get("ksp_dtype", np.complex64)
    gt = methodCfg.ground_truth_im
    outType = methodCfg.im_bit_depth
    debug = bool(cfg.get("debug_verify", False))
    wave_basis = cfg.get("wavelet_basis", "db4")
    R = int(cfg.get("R", 4))
    acs = int(cfg.get("acs", 24))
    simulate_undersampling = bool(cfg.get("simulate_undersampling", True))

    # (2) PREPROCESS
    ksp, _ = preprocess_kspace(kspace, out_dtype=ksp_type)
    ksp_scale = methodCfg.state.get("ksp_scale", 1.0)
    ksp = ksp / ksp_scale

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

    if "mask2d" not in methodCfg.state and simulate_undersampling:
        # no setup was run -> need to build mask on the fly
        methodCfg.state["mask2d"] = make_vds_ky_mask(kspace.shape[1], kspace.shape[2], R=R, acs=acs)
    
    if simulate_undersampling:
        mask2d = methodCfg.state["mask2d"] 
        w = mask2d[None, ...]                    # (1,H,W) broadcastable
        # apply sampling for CS
        ksp = ksp * w

    if debug and simulate_undersampling:
        # CS+L1 wavelet
        acquired_frac = float(mask2d.mean())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # (a) Sampling mask M
        axes[0].imshow(mask2d, cmap="gray", aspect="auto", interpolation="nearest")
        axes[0].set_title(f"Sampling Mask M\n(acquired={acquired_frac:.2f}, R={R:.1f}x)")
        axes[0].set_xlabel("kx")
        axes[0].set_ylabel("ky")

        # (b) ky line acquisition profile to show sampling density along phase-encode direction
        ky_coverage = mask2d.mean(axis=1)   # fraction of kx acquired per ky line — shape (H,)
        axes[1].plot(ky_coverage, np.arange(len(ky_coverage)), linewidth=0.8)
        axes[1].set_xlim(0, 1.05)
        axes[1].set_title("ky Sampling Profile\n(fraction of kx acquired per ky line)")
        axes[1].set_xlabel("Acquired fraction")
        axes[1].set_ylabel("ky index")
        axes[1].axvline(acquired_frac, color="red", linestyle="--", linewidth=0.8, label=f"mean={acquired_frac:.2f}")
        axes[1].legend(fontsize=8)

        plt.suptitle(f"L1-Wavelet Mask Debug")
        plt.tight_layout()

        out_dir = RESULTS / "images" / "debug" / "l1wavelet"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "mask_debug.png", dpi=150)
        plt.close()
        print(f"[l1wavelet] mask debug plot saved to {out_dir / 'mask_debug.png'}")

    # =========================== TIME/MEMORY TRACKING STARTS ===================================
    tracemalloc.start()
    t0 = time.perf_counter()
       
    # Solve the CS objective with L1-Wavelet regularization 
    # 'weights=w' restricts the data-consistency term to acquired k-space locations (mask M)
    im = L1WaveletRecon(
        ksp,maps,lambda_reg, weights=w, wave_name=wave_basis,
        device=device,max_iter=wavelet_max_iter).run()
    
    if methodCfg.state["out_buf"] is None:
        if outType == "float32":
            out = np.abs(im).astype(np.float32, copy=False)
        elif outType == "float64":
            out = np.abs(im).astype(np.float64, copy=False)
    else:
        out_buf = methodCfg.state["out_buf"]
        np.abs(im, out=out_buf)

    time_elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # =========================== TIME/MEMORY TRACKING ENDS =======================================

    if debug:
        if gt is None:
            print("[l1wavelet] debug_verify=True but no ground truth image provided (skipping).")
        else:
            pred_im = out if methodCfg.state.get("out_buf") is None else methodCfg.state["out_buf"]
            gt_arr  = np.asarray(gt)
            if gt_arr.shape != pred_im.shape:
                print(f"[l1wavelet] GT shape {gt_arr.shape} != pred {pred_im.shape} (skipping metrics).")
            else:
                eps   = 1e-12
                # before computing metrics, normalize both to same scale
                pred64 = pred_im.astype(np.float64, copy=False)
                gt64 = gt_arr.astype(np.float64, copy=False)

                # normalize by GT max so rel_l2 is scale-invariant
                gt_max = gt64.max() + 1e-12
                pred= pred64 / pred64.max() * gt_max  # rescale pred to GT range

                # Relative L2 error: sensitive to magnitude/scaling differences
                rel_l2 = np.linalg.norm(pred - gt64) / (np.linalg.norm(gt64) + eps)

                # Pearson correlation (flattened): invariant to global scaling
                p = pred.ravel();  p = p - p.mean()
                g = gt64.ravel();  g = g - g.mean()
                corr = float((p @ g) / (np.linalg.norm(p) * np.linalg.norm(g) + eps))

                print(f"[l1wavelet] verify: corr={corr:.4f}, rel_l2={rel_l2:.4f}")
    
    out_final = out if methodCfg.state.get("out_buf") is None else methodCfg.state["out_buf"]
    return out_final, peak, time_elapsed


def cleanup(methodCfg: MethodConfigs):
    # clear state if not none
    if methodCfg.state is not None:
        methodCfg.state.clear()
