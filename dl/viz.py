from __future__ import annotations
from fastmri.models.unet import Unet
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# see what specific unet struct is in fastmri so we can freeze everything up to last 1-2 up-conv blocks...
model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
for name, param in model.named_parameters():
    print(name, param.shape)

"""
Encoder:
  down_sample_layers.0  (32 ch)
  down_sample_layers.1  (64 ch)
  down_sample_layers.2  (128 ch)
  down_sample_layers.3  (256 ch)
  conv                  (512 ch) 

Decoder:
  up_transpose_conv.0 + up_conv.0   (256 ch)
  up_transpose_conv.1 + up_conv.1   (128 ch)
  up_transpose_conv.2 + up_conv.2   (64 ch)
  up_transpose_conv.3 + up_conv.3   (32 ch -> 1 ch) 
  *up_conv.3 is sequential with up_conv.30.0 and up_conv.3.1 (final 1x1 that maps out_chans=1)
"""

def norm01(im: np.ndarray) -> np.ndarray:
    """Percentile clip and normalize to [0,1] for display."""
    im = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(im, [1, 99])
    im = np.clip(im, lo, hi)
    return (im - lo) / (hi - lo + 1e-12)


def plot_training_curve(
    train_losses: list[float],
    val_losses: list[float],
    out_path: str | Path,
):
    """
    Loss curves over epochs.
    Vertical dotted line marks the best val epoch — useful for spotting
    if you should have stopped training earlier (overfitting).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    best_ep = int(np.argmin(val_losses)) + 1  # 1-indexed

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="train L1")
    ax.plot(epochs, val_losses,   label="val L1")
    ax.axvline(best_ep, linestyle="--", color="gray", linewidth=0.8,
               label=f"best val epoch {best_ep}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("U-Net Training Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] training curve -> {out_path}")


def plot_triplet(
    zf_mag: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: str | Path,
):
    """
    Three panel comparison: zero-filled input | network output | ground truth.
    This is the most important visual — tells you if the network is actually
    removing aliasing artifacts or just passing the input through unchanged.
    MAE shown in titles so you can quantify improvement at a glance.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # compute MAE against gt for both input and pred so you can see improvement
    mae_zf   = float(np.mean(np.abs(norm01(zf_mag) - norm01(gt))))
    mae_pred = float(np.mean(np.abs(norm01(pred)   - norm01(gt))))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(norm01(zf_mag), cmap="gray")
    axes[0].set_title(f"Zero-filled input\nMAE={mae_zf:.4f}")
    axes[0].axis("off")

    axes[1].imshow(norm01(pred), cmap="gray")
    axes[1].set_title(f"U-Net output\nMAE={mae_pred:.4f}")
    axes[1].axis("off")

    axes[2].imshow(norm01(gt), cmap="gray")
    axes[2].set_title("Ground truth")
    axes[2].axis("off")

    fig.suptitle("Reconstruction Comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] triplet -> {out_path}")


def plot_diff(
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: str | Path,
):
    """
    Absolute difference map |pred - gt|.
    Hot colormap makes errors pop — bright regions are where the network
    is still struggling. Structured errors (e.g. always wrong at edges or
    in a specific tissue) suggest the network needs more training or more
    unfrozen layers. Random/diffuse errors are just residual noise.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diff = np.abs(norm01(pred) - norm01(gt))
    mean_err = float(np.mean(diff))
    max_err  = float(np.max(diff))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(diff, cmap="hot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"|pred - gt|\nmean={mean_err:.4f}  max={max_err:.4f}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] diff map -> {out_path}")