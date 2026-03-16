from __future__ import annotations
import numpy as np
import sys
from bench.utils import Configs, MethodConfigs, ReconMethod, MODEL_REUSE_PATH, kspace_to_x_image, RESULTS
import tracemalloc
import time
from dl.m4raw_pytorch import train_val_split
from dl.train_unet_varnet_m4raw import train, init_model
import torch
import torch.nn as nn
from pathlib import Path
from fastmri.models.unet import Unet
from dl.viz import plot_diff, plot_triplet, plot_training_curve


def run_training_and_prealloc(kspace: np.ndarray, methodCfg: MethodConfigs, **kwargs) -> tuple[int,float]:
    """
    kspace here is kspace_all (S,C,H,W) if train_new true, otherwise single slice (C,H,W)
    responsibilities:
    1 - train a new model OR load existing checkpoint
    2 - preallocate numpy buffers for inference pipeline
    3 - store model + buffers in methodCfg.state
    """
    # configs
    unet = methodCfg.unet
    out_dtype = methodCfg.im_bit_depth
    use_ifftshift = bool(unet.get("use_ifftshift", True))
    max_iters = int(unet.get("max_iters", 30))
    lr = float(unet.get("lr", 1e-4))
    batch_size = int(unet.get("batch_size", 2))
    n_decode_blks_to_freeze = int(unet.get("n_decode_blks_to_freeze", 3))
    debug = bool(unet.get("debug_verify", False))
    freeze_on = bool(unet.get("freeze_layers", False))
        
    # ========================= START MEMORY/TIME TRACKING ================================
    t0 = time.perf_counter()
    tracemalloc.start()

    # TODO: automate running with train_new true and train_new false on diff occasions ...
    # (1) TRAINING
    if kwargs.get("train_new"):
        slices = [kspace[s] for s in range(kspace.shape[0])]
        H, W = slices[0].shape[1], slices[0].shape[2]
        mask2d = methodCfg.undersampling_mask["mask2d"]
        
        train_ds, val_ds = train_val_split(slices, mask2d, use_ifftshift=use_ifftshift, reconMethod="UNET")

        ckpt_path, train_losses, val_losses = train(
            train_ds, val_ds,
            out_ckpt="results/unet_m4raw.pt", 
            epochs=max_iters, batch_size=batch_size, lr=lr,
            n_decoder_blocks_to_freeze=n_decode_blks_to_freeze,
            freeze_on=freeze_on,
            debug=debug,
            reconMethod="UNET",
        )

        # choose a slice for debug
        if debug:
            debug_slice = kspace[ kspace.shape[0] // 2 ]

    else:
        # reuse existing checkpoint
        ckpt_path = MODEL_REUSE_PATH
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run with new_train=true !!")
        # infer H,W from the single slice kspace passed
        H, W = kspace.shape[1], kspace.shape[2]

    # (2) MODEL LOADING
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model, device = init_model(reconMethod="UNET") # blank slat model w random weights
    # weights are saved in the model.state_dict() created during training
    model.load_state_dict(ckpt["model_state"])
    model.eval() # switch to inference mode (dropout off, batchnorm uses running stats)
    
    # (3) PREALLOC TENSORS FOR REPEATED INFERENCE (per-slice images)
    # NOTE: we cannot preallocate fwd pass layers bcuz it's all internal to PyTorch
    # can preallocate:
    # - x_im (slice transformed to im domain), computed from masked kspace each run
    # - model input tensor x_tensor (1,1,H,W)
    # - output out_buf (H,W)

    x_im = np.empty((H,W), dtype=np.float32)
    x_tensor = torch.zeros((1,1,H,W), dtype=torch.float32, device=device)
    
    if out_dtype == "float32":
        out_buf = np.empty((H,W), dtype=np.float32)
    else:
        out_buf = np.empty((H,W), dtype=np.float64)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0
    # ========================= STTOP MEMORY/TIME TRACKING ================================

    methodCfg.state = {
        "model": model,
        "device": device,
        "input_im": x_im,
        "input_tensor": x_tensor,
        "out": out_buf,
    }

    if debug and kwargs.get("train_new"):
        # visualization debug plots
        zf_mag = kspace_to_x_image(debug_slice, mask2d, use_ifftshift)
        scale = np.percentile(zf_mag, 99) + 1e-12
        with torch.no_grad():
            x = torch.from_numpy((zf_mag / scale)[None, None, ...]).to(device)
            pred = model(x)[0, 0].cpu().numpy() * scale
        gt = methodCfg.ground_truth_im

        out_dir = RESULTS / "images" / "debug" / "unet"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_triplet(zf_mag, pred, gt, out_dir / "triplet.png")
        plot_diff(pred, gt, out_dir / "diff.png")

    return peak, time_elapsed


@torch.no_grad()
def run_inference(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    # read config
    unet = methodCfg.unet
    out_dtype = methodCfg.im_bit_depth
    mask = methodCfg.undersampling_mask
    simulate_undersampling = bool(unet.get("simulate_undersampling", True))
    model =  methodCfg.state["model"]
    input_tensor = methodCfg.state["input_tensor"]
    out = methodCfg.state["out"]
    use_ifft_shift = bool(unet.get("use_ifftshift", True))

    # convert ksp to zero-filled mag image
    x_mag = kspace_to_x_image(kspace, mask["mask2d"] if simulate_undersampling else np.ones_like(mask), use_ifft_shift)
    x_scale = float(np.percentile(x_mag, 99)) + 1e-12

    # ========================= START MEMORY/TIME TRACKING ================================
    t0 = time.perf_counter()
    tracemalloc.start()

    x_mag = x_mag/x_scale # normalize
    input_tensor[0,0].copy_(torch.from_numpy(x_mag))
    # INFERENCE
    pred = model(input_tensor)
    # copy out to prealloc buffer in-place, undo norm
    np.multiply(pred[0,0].cpu().numpy(), x_scale, out=out)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0
    # ========================= STOP MEMORY/TIME TRACKING ================================

    return out, peak, time_elapsed


def cleanup(methodCfg: MethodConfigs):
    # clear state
    if methodCfg.state is not None:
        if "model" in methodCfg.state:
            del methodCfg.state["model"] # explicitly delete bcuz this shit gna be heavyy
        methodCfg.state.clear()