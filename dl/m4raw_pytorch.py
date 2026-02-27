# Goal: manufacture for every training sample
# (x) the undersampled, "bad" input image
# (y) the ground truth, "good" output image
# U-Net is setup as a single-slice image to image network meaning we must convert from k-space

# flow:
# x: m4raw kspace (C,H,W) --> mask --> IFFT+RSS --> bad_mag --> x (1,H,W) --> Unet
# y: m4raw kspace (C,H,W) --> IFFT+RSS --> gt_mag --> y (1,H,W)

import numpy as np
import torch
from bench.utils import kspace_to_x_image

def make_training_example_pair(kspace: np.ndarray, mask2d: np.ndarray, use_ifftshift: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns x, y as described above from kspace
    """

    kspace_y = kspace
    # apply mask for x:
    w = mask2d[None, ...]
    kspace_x = kspace * w
    # reconstruct
    imx = kspace_to_x_image(kspace_x)
    imy = kspace_to_x_image(kspace_y)
    # normalize        
    scale = np.percentile(imy, 99) + 1e-12
    imx = (imx / scale).astype(np.float32)
    imy = (imy / scale).astype(np.float32)
    return imx, imy

class M4RawSlices(Dataset):
    def __init__(self, kspace_slices: list, mask2d: np.ndarray, use_fft_shift: bool):
        # store kspace_slices, mask2d
        self.ksp_slices = kspace_slices
        self.mask2d = mask2d
        self.ifft_shift = use_fft_shift
    
    def __len__(self):
        # number of training examples
        return len(self.ksp_slices)

    def __getitem__(self, idx):
        # method called by pytorch's dataloader
        # per slice training example
        x,y = make_training_example_pair(self.ksp_slices[idx], self.mask2d, self.ifft_shift)
        # wrap in torch tensors, add channel dim: [None,...] -> shape (1,H,W)
        return torch.from_numpy(x[None, ...]), torch.from_numpy(y[None, ...])

# build 2 datasets (train/test)
def train_val_split(kspace_slices, mask2d, use_ifftshift, val_frac=0.1, seed=42):
    # shuffle indices with rng
    rng = np.random.default_rng(seed)
    indices = np.arange(len(kspace_slices))
    rng.shuffle(indices)
    # split into train/val lists
    n_val = max(1, int(len(indices)*val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return M4RawSlices(train_slices, mask2d, use_ifftshift), M4RawSlices(val_slices, mask2d, use_ifftshift)
