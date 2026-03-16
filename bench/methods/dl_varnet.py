from __future__ import annotations
import numpy as np
import sys
from bench.utils import Configs, MethodConfigs, ReconMethod, MODEL_REUSE_PATH_VARNET, RESULTS
import tracemalloc
import time
from dl.m4raw_pytorch import train_val_split
from dl.train_unet_varnet_m4raw import train, init_model
import torch
import torch.nn as nn
from pathlib import Path
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
    varnet = methodCfg.varnet
    out_dtype = methodCfg.im_bit_depth
    use_ifftshift = bool(varnet.get("use_ifftshift", True))
    max_iters = int(varnet.get("max_iters", 30))
    lr = float(varnet.get("lr", 1e-4))
    batch_size = int(varnet.get("batch_size", 1))
    n_decode_blks_to_freeze = int(varnet.get("n_decode_blks_to_freeze", 3))
    debug = bool(varnet.get("debug_verify", False))
    freeze_on = bool(varnet.get("freeze_layers", False))
    model_arch = varnet["model_architecture"] if varnet["model_architecture"] is not None else None
    seed = int(varnet.get("seed", 42))
        
    # ========================= START MEMORY/TIME TRACKING ================================
    t0 = time.perf_counter()
    tracemalloc.start()

    # TODO: automate running with train_new true and train_new false on diff occasions ...
    # (1) TRAINING
    if kwargs.get("train_new"):
        slices = [kspace[s] for s in range(kspace.shape[0])]
        H, W = slices[0].shape[1], slices[0].shape[2]
        mask2d = methodCfg.undersampling_mask["mask2d"]
        
        train_ds, val_ds = train_val_split(slices, mask2d, use_ifftshift=use_ifftshift, reconMethod="VARNET", seed=seed)

        ckpt_path, train_losses, val_losses = train(
            train_ds, val_ds,
            out_ckpt="results/varnet_m4raw.pt", 
            epochs=max_iters, batch_size=batch_size, lr=lr,
            n_decoder_blocks_to_freeze=n_decode_blks_to_freeze,
            freeze_on=freeze_on,
            debug=debug,
            reconMethod="VARNET",
            model_arch=model_arch,
        )

        # choose a slice for debug
        if debug:
            debug_slice = kspace[ kspace.shape[0] // 2 ]

    else:
        # reuse existing checkpoint
        ckpt_path = MODEL_REUSE_PATH_VARNET
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run with new_train=true !!")
        # infer H,W from the single slice kspace passed
        H, W = kspace.shape[1], kspace.shape[2]

    # (2) MODEL LOADING
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model, device = init_model(n_decoder_blocks_to_freeze=n_decode_blks_to_freeze, reconMethod="VARNET", freeze_on=freeze_on, model_arch=model_arch) # blank slat model w random weights
    # weights are saved in the model.state_dict() created during training
    model.load_state_dict(ckpt["model_state"])
    model.eval() # switch to inference mode (dropout off, batchnorm uses running stats)
    
    # (3) PREALLOC TENSORS FOR REPEATED INFERENCE (per-slice images)
    # NOTE: we cannot preallocate fwd pass layers bcuz it's all internal to PyTorch
    # can preallocate:
    # - masked tensor mask_tensor (1,1,H,W,1) -> undersampling ksp mask never changes btwn runs -> build tensor once for all runs
    # - kspace input tensor x_tensor (1,C=4,H,W,2) -> ksp input real/im stacked tensor we build every inference call, C=num of coils
    # - output out_buf (H,W)

    mask2d = methodCfg.undersampling_mask["mask2d"]
    mask_tensor = torch.from_numpy(
        mask2d[None, None, :, :, None].astype(np.float32) # expected shape [1,1,H,W,1]
    ).to(device).bool() 
    chan = kspace.shape[0] if kspace.ndim == 3 else kspace.shape[1] # (C,H,W) or (S,C,H,W)
    x_tensor = torch.zeros((1,chan,H,W,2), dtype=torch.float32, device=device)
    
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
        "mask_tensor": mask_tensor,
        "input_tensor": x_tensor,
        "out": out_buf,
    }

    if debug and kwargs.get("train_new"):
        # visualization debug plots
        x_ri, scale = preprocess_ksp_for_x_input(debug_slice, methodCfg)
        with torch.no_grad():
            x_ri = torch.from_numpy((x_ri[None, ...])).to(device)
            pred = model(x_ri, mask_tensor)[0].cpu().numpy() * scale
        gt = methodCfg.ground_truth_im

        out_dir = RESULTS / "images" / "debug" / "varnet"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_triplet(debug_slice, pred, gt, out_dir / "triplet.png")
        plot_diff(pred, gt, out_dir / "diff.png")

    return peak, time_elapsed

# MUST MATCH PREPROCESSING OF KSP IN train_unet_varnet_m4raw.py varnet dataclass
def preprocess_ksp_for_x_input(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    ksp = kspace.astype(np.complex64)
    scale = float(np.percentile(np.abs(ksp), 99)) + 1e-12  # from full kspace, before masking
    ksp = ksp * methodCfg.undersampling_mask["mask2d"][None, ...] # apply mask
    x_mag = ksp / scale
    return np.stack([x_mag.real, x_mag.imag], axis=-1).astype(np.float32), scale

@torch.no_grad()
def run_inference(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    # read config
    varnet = methodCfg.varnet
    model =  methodCfg.state["model"]
    input_tensor = methodCfg.state["input_tensor"]
    mask_tensor = methodCfg.state["mask_tensor"] #prebuilt
    out = methodCfg.state["out"]
    use_ifft_shift = bool(varnet.get("use_ifftshift", True))

    # ========================= START MEMORY/TIME TRACKING ================================
    t0 = time.perf_counter()
    tracemalloc.start()

    x_ri, x_scale = preprocess_ksp_for_x_input(kspace, methodCfg)
    input_tensor[0].copy_(torch.from_numpy(x_ri)) # (1,C,H,W,2) -> fill batch dim 1
    # INFERENCE
    pred = model(input_tensor, mask_tensor)
    # copy out to prealloc buffer in-place, undo norm
    np.multiply(pred[0].cpu().numpy(), x_scale, out=out) # return (B,H,W) where B=batch size

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