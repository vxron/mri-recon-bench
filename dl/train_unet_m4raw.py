from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from fastmri.models.unet import Unet
import numpy as np
    import sys

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


def init_model(n_decoder_blocks_to_freeze: int = 3, device: torch.device = None) -> tuple[Unet, torch.device]:
    """
    Instantiate the fastMRI U-Net and freeze appropriate layers.
    chans=32, num_pool_layers=4 matches Unet architecture.
    Returns model and device so callers don't have to repeat this boilerplate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0).to(device)
    freeze_encoder_and_decoder_up_to_n(model, n_decoder_blocks_to_freeze)
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


def train(
    train_ds,
    val_ds,
    out_ckpt: str = "results/unet_m4raw.pt",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 1e-4,
    n_decoder_blocks_to_freeze: int = 3,
) -> str:
    """
    Full training loop. 
    Chooses best models based on lowest val loss (sequentially) for good generalization.
    """
    model, device = build_model(n_decoder_blocks_to_freeze)

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

    for ep in range(epochs):
        tr = train_one_epoch(model, train_loader, optim, device)
        va = eval_one_epoch(model, val_loader, device)
        scheduler.step(va)

        train_losses.append(tr)
        val_losses.append(va)
        print(f"epoch {ep+1:03d}  train={tr:.5f}  val={va:.5f}  lr={optim.param_groups[0]['lr']:.2e}")

        if va < best_val:
            best_val = va
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "val_loss": va,
                "chans": 32,
                "num_pool_layers": 4,
                "n_decoder_blocks_to_freeze": n_decoder_blocks_to_freeze,
            }, out_ckpt)
            print(f"  saved -> {out_ckpt}")

    return out_ckpt, train_losses, val_losses


if __