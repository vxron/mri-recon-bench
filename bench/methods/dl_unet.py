from __future__ import annotations
import numpy as np
import sys
from bench.utils import Configs, MethodConfigs, ReconMethod, MODEL_REUSE_PATH, kspace_to_x_image
import tracemalloc
import time
from dl.m4raw_pytorch import train_val_split, make_training_example_pair
from dl.train_unet_m4raw import train, build_model
import torch
import torch.nn as nn
from pathlib import Path
from fastmri.models.unet import Unet


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
    mask = methodCfg.undersampling_mask
    simulate_undersampling = bool(unet.get("simulate_underampling", True))
    out_dtype = methodCfg.im_bit_depth
    use_ifftshift = bool(unet.get("use_ifftshift", True))
    max_iters = int(unet.get("max_iters", 30))
    lr = float(unet.get("lr", 1e-4))
    batch_size = int(unet.get("batch_size"), 2)
    n_decode_blks_to_freeze = int(unet.get("n_decode_blks_to_freeze"), 3)
        
    # ========================= START MEMORY/TIME TRACKING ================================
    t0 = time.perf_counter()
    tracemalloc.start()

    # TODO: automate running with train_new true and train_new false on diff occasions ...
    # (1) TRAINING
    if kwargs.get("train_new"):
        slices = [kspace[s] for s in range(kspace.shape[0])]
        H, W = slices[0].shape[1], slices[0].shape[2]
        mask2d = methodCfg.undersampling_mask["mask2d"]
        
        train_ds, val_ds = train_val_split(slices, mask2d, use_ifftshift=use_ifftshift)

        ckpt, train_losses, val_losses = train(
            train_ds, val_ds,
            out_ckpt="results/unet_m4raw.pt", 
            epochs=max_iters, batch_size=batch_size, lr=lr
        )
    else:
        # reuse existing checkpoint
        ckpt = MODEL_REUSE_PATH
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run with new_train=true !!")
        # infer H,W from the single slice kspace passed
        H, W = kspace.shape[1], kspace.shape[2]

    # (2) MODEL LOADING

    
    # (3) PREALLOC TENSORS FOR REPEATED INFERENCE (per-slice images)
    # NOTE: we cannot preallocate fwd pass layers bcuz it's all internal to PyTorch
    # can preallocate:
    # - masked k-space (input)
    # - x_im (slice transformed to im domain)
    # - model input tensor x_tensor (1,1,H,W)
    # - output out_buf (H,W)

    

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0
    # ========================= STTOP MEMORY/TIME TRACKING ================================

    methodCfg.state = {
        "masked_ksp":
        "device":
        "input_im"
        "input_tensor"
        "out_buf":
    }

    return out, peak, time_elapsed

def run_inference(kspace: np.ndarray, methodCfg: MethodConfigs) -> np.ndarray:
    # read config
    cfg = methodCfg.unet
    mask = methodCfg.undersampling_mask
    out_dtype = methodCfg.im_bit_depth
    H = kspace.shape[1]
    W = kspace.shape[2]
    simulate_undersampling = bool(cfg.get("simulate_underampling", True))

    # ========================= START MEMORY/TIME TRACKING ================================
    t0 = time.perf_counter()
    tracemalloc.start()



    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_elapsed = time.perf_counter() - t0
    # ========================= START MEMORY/TIME TRACKING ================================


    return out, peak, time_elapsed


def cleanup(methodCfg: MethodConfigs):
    # clear state
    if methodCfg.state is not None:
        methodCfg.state.clear()