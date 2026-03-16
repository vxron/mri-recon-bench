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
from torch.utils.data import Dataset

class M4RawSlices(Dataset):
    def __init__(self, kspace_slices: list, mask2d: np.ndarray, use_fft_shift: bool):
        # store kspace_slices, mask2d
        self.ksp_slices = kspace_slices
        self.mask2d = mask2d
        self.ifft_shift = use_fft_shift
    
    def __len__(self):
        # number of training examples
        return len(self.ksp_slices)
    
    def make_training_example_pair(self, kspace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kspace_y = kspace
        # apply mask for x:
        w = self.mask2d[None, ...]
        kspace_x = kspace * w
        ones = np.ones(self.mask2d.shape, dtype=np.float32) # kspace_to_x_image expects mask arg
        # reconstruct
        imx = kspace_to_x_image(kspace_x, ones, use_ifftshift=self.ifft_shift)
        imy = kspace_to_x_image(kspace_y, ones, use_ifftshift=self.ifft_shift)
        # normalize        
        scale = np.percentile(imy, 99) + 1e-12
        imx = (imx / scale).astype(np.float32)
        imy = (imy / scale).astype(np.float32)
        return imx, imy

    def __getitem__(self, idx):
        # method called by pytorch's dataloader
        # per slice training example
        x,y = self.make_training_example_pair(self.ksp_slices[idx])
        # wrap in torch tensors, add channel dim: [None,...] -> shape (1,H,W)
        return torch.from_numpy(x[None, ...]), torch.from_numpy(y[None, ...])

# build 2 datasets (train/test)
def train_val_split(kspace_slices, mask2d, use_ifftshift, *, reconMethod:str = "UNET", val_frac:float=0.1, seed:int=42):
    # reconMethod should be "VARNET" or "UNET"
    # shuffle indices with rng
    rng = np.random.default_rng(seed)
    indices = np.arange(len(kspace_slices))
    rng.shuffle(indices)
    # split into train/val lists
    n_val = max(1, int(len(indices)*val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_slices = [kspace_slices[i] for i in train_idx]
    val_slices = [kspace_slices[i] for i in val_idx]
    
    if reconMethod == "UNET":
        return M4RawSlices(train_slices, mask2d, use_ifftshift), M4RawSlices(val_slices, mask2d, use_ifftshift)
    elif reconMethod == "VARNET":
        return M4RawSlicesKSpace(train_slices, mask2d, use_ifftshift), M4RawSlicesKSpace(val_slices, mask2d, use_ifftshift)
    else:
        return None

# FOR VARNET -> training examples are in kspace instead of image space
class M4RawSlicesKSpace(Dataset):
    def __init__(self, kspace_slices: list, mask2d: np.ndarray, use_fft_shift: bool):
        self.kspace_slices = kspace_slices
        self.mask2d = mask2d                 # varnet needs mask2d explicitly so it knows which ksp lines were actually acquired (since it only updated unacquired locations with DL predictions)
        self.ifft_shift = use_fft_shift

    def __len__(self):
        return len(self.kspace_slices)
    
    def make_training_example_pair(self, kspace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # for varnet the training pair is:
        # input: (undersampled ksp (ksp * mask2d), mask2d)
        # target output: rss gt computed from fully sampled ksp
        ksp = kspace.astype(np.complex64)
        # normalize by kspace scale and apply same scale to target for consistency
        scale = float(np.percentile(np.abs(ksp), 99)) + 1e-12

        # apply mask for x (input):
        w = self.mask2d[None, ...]
        kspace_x = ksp * w
        kspace_x = kspace_x / scale
        kspace_x = (np.stack([kspace_x.real, kspace_x.imag], axis=-1)).astype(np.float32) # organizes into real and imag where [...,0] is mapped to real and [...,1] is mapped to im -> 2 channels

        # reconstruct the gt im
        kspace_y = ksp / scale
        ones = np.ones(self.mask2d.shape, dtype=np.float32) # kspace_to_x_image expects mask arg
        imy = kspace_to_x_image(kspace_y, ones, use_ifftshift=self.ifft_shift) 

        return kspace_x, imy
    
    def __getitem__(self, idx):
        # method called by dataloader per slice tr example
        kspace_x, y = self.make_training_example_pair(self.kspace_slices[idx])
        return (
            torch.from_numpy(kspace_x),                      # (C,H,W,2) after stack re/im
            torch.from_numpy(self.mask2d[None, ..., None]),  # (1, H, W, 1) is what Varnet expects w/ trailing dim for re/im
            torch.from_numpy(y)                   # output im is just (H, W)
        )
