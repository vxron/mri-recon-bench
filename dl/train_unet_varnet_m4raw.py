from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from fastmri.models.unet import Unet
from fastmri.models import VarNet
from dl.viz import plot_training_curve
import numpy as np
from typing import Callable

def freeze_encoder_and_decoder_up_to_n(model: Unet, n_decoder_blocks_to_freeze: int = 3):
    """
    Freeze all encoder blocks (coarse MRI details same across field strengths)
    Freeze decoder blocks up to last one ~10% of params and most responsible for adapting low-field characteristics
    """
    # freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze necessary decoder blocks
    total_blocks = len(model.up_conv) # 4
    for i in range(n_decoder_blocks_to_freeze, total_blocks):
        for param in model.up_conv[i].parameters():
            param.requires_grad = True
        for param in model.up_transpose_conv[i].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def init_model(*, n_decoder_blocks_to_freeze: int = 3, reconMethod: str = "UNET", device: torch.device = None, freeze_on: bool = False, model_arch: dict = None) -> tuple[Unet, torch.device]:
    """
    Instantiate the fastMRI U-Net or VARNET and freeze appropriate layers.
    chans=32, num_pool_layers=4 matches Unet architecture.
    Returns model and device so callers don't have to repeat this boilerplate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if reconMethod == "UNET":
        model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0).to(device)
    elif reconMethod == "VARNET":
        num_cascades = model_arch.get("num_cascades", 8)
        pools = model_arch.get("pools", 4)
        chans = model_arch.get("chans", 18)
        sens_pools = model_arch.get("sens_pools", 4)
        sens_chans = model_arch.get("sens_chans", 8)
        model = VarNet(
            num_cascades=num_cascades,
            pools=pools,
            chans=chans,
            sens_pools=sens_pools,
            sens_chans=sens_chans,
        ).to(device)
    else:
        print("issue w varnet/unet config")
        return None

    if freeze_on:
        freeze_encoder_and_decoder_up_to_n(model, n_decoder_blocks_to_freeze)
    
    trainable = sum(p.numel() for p in model.parameters())
    print(f"Training from scratch: {trainable:,} params")
    
    return model, device


def train_one_epoch(model: Unet, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device) -> float:
    """
    One full pass over training data.
    L1 loss (MAE) is standard for MRI recon - more robust to outliers than L2.
    Returns mean loss over all batches.
    """
    model.train()
    loss_fn = nn.L1Loss()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item()
    return total / max(1, len(loader))

def train_one_epoch_varnet(model: VarNet, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.L1Loss()
    total = 0.0

    for i, (ksp, mask, target) in enumerate(loader):
        x, y, mask = ksp.to(device), target.to(device), mask.to(device)

        #if i == 0:  # first batch only debug
            #print("ksp shape:", x.shape)
            #print("mask shape:", mask.shape)
            #print("target shape:", y.shape)
            #print("ksp min/max:", x.min().item(), x.max().item())
            #print("target min/max:", y.min().item(), y.max().item())
        
        pred = model(x, mask.bool()) # varnet's fwd call is model(ksp,mask) *uses mask explicitly to keep dl predictions only for unsampled lines in ksp
        loss = loss_fn(pred, y) 
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_one_epoch(model: Unet, loader: DataLoader, device: torch.device) -> float:
    """
    Validation pass. @torch.no_grad() bcuz we don't need gradients here
    since we're not updating weights.
    Returns mean loss over all batches.
    """
    model.eval()
    loss_fn = nn.L1Loss()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += loss_fn(model(x), y).item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_one_epoch_varnet(model: VarNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.L1Loss()
    total = 0.0
    for ksp, mask, target in loader:
        x, y, mask = ksp.to(device), target.to(device), mask.to(device)
        total += loss_fn(model(x, mask.bool()), y).item()
    return total / max(1, len(loader))

def train(
    train_ds,
    val_ds,
    out_ckpt: str = "results/unet_m4raw.pt",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 1e-4,
    n_decoder_blocks_to_freeze: int = 3,
    freeze_on: bool = False,
    debug: bool = False,
    reconMethod: str = "UNET",
    model_arch: dict = None,
    patience: int = 8, # for early stopping based on val (max consecutive no-improve epochs)
    min_delta: float = 1e-4 # minimum diff btwn 2 subsequent val losses to consider it a meaningful change
) -> str:
    """
    Full training loop. 
    Chooses best models based on lowest val loss (sequentially) for good generalization.
    This is the PUBLIC API USED.
    """
    if reconMethod == "UNET":
        model, device = init_model(n_decoder_blocks_to_freeze=n_decoder_blocks_to_freeze, reconMethod="UNET", freeze_on=freeze_on)
    elif reconMethod == "VARNET":
        model, device = init_model(n_decoder_blocks_to_freeze=n_decoder_blocks_to_freeze, reconMethod="VARNET", model_arch=model_arch, freeze_on=freeze_on)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    # only pass trainable params to optimizer - passing frozen params wastes
    # memory on momentum/variance buffers that never get used
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    # reduceLRonPlateau halves lr if val loss doesn't improve for 3 epochs [important for ensuring stability]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5)

    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    train_losses, val_losses = [], []
    epochs_no_val_improve = 0

    trainer: Callable = None
    validator: Callable = None
    if reconMethod == "UNET":
        trainer = train_one_epoch
        validator = eval_one_epoch
    elif reconMethod == "VARNET":
        trainer = train_one_epoch_varnet
        validator = eval_one_epoch_varnet
    else:
        print("issue w reconmethod setup in train()")
        return

    for ep in range(epochs):
        tr = trainer(model, train_loader, optim, device)
        va = validator(model, val_loader, device)
        scheduler.step(va)

        train_losses.append(tr)
        val_losses.append(va)
        print(f"epoch {ep+1:03d}  train={tr:.5f}  val={va:.5f}  lr={optim.param_groups[0]['lr']:.2e}")

        if va < best_val and (best_val - va) >= min_delta:
            best_val = va
            # reset "no-improve" counter
            epochs_no_val_improve = 0
            # the 'save' step saves the model w the appropriate new changes
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "val_loss": va,
                "chans": model_arch["chans"] if model_arch is not None else 32,
                "num_pool_layers": model_arch["pools"] if model_arch is not None else 4,
                "num_cascades": model_arch["num_cascades"] if model_arch is not None else 0,
                "n_decoder_blocks_to_freeze": n_decoder_blocks_to_freeze if freeze_on else 0,
            }, out_ckpt)
            print(f"  saved -> {out_ckpt}")
        else:
            epochs_no_val_improve += 1
            if epochs_no_val_improve >= patience:
                print(f"  early stop at epoch {ep+1} (no improvement for {patience} epochs)")
                break
    
    if debug: 
        out_dir = Path(out_ckpt).parent / "plots"
        out_dir.mkdir(exist_ok=True)
        plot_training_curve(train_losses, val_losses, out_dir / "training_curve.png")

    return out_ckpt, train_losses, val_losses

