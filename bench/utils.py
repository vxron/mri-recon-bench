from enum import Enum
from dataclasses import dataclass, fields, asdict, field
from typing import List, Any
import numpy as np
import sigpy as sp
from pathlib import Path

# DATAPATHS

ROOT = Path(__file__).resolve().parents[1]   # mri-recon-bench/
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# ENUMS / DICTIONARIES

class ReconMethod(str, Enum):
    CS_L1 = "cs_l1wavelet"
    UNET = "dl_unet"
    SWIN = "dl_swin"
    IFFT_BASE = "baseline_ifft"
    SENSE = "sense_espirit"
    GRAPPA = "grappa"

SETUP_KWARGS = {
    ReconMethod.SENSE: {"curr_method": ReconMethod.SENSE},
    ReconMethod.CS_L1: {"curr_method": ReconMethod.CS_L1},
    ReconMethod.GRAPPA: {}, # no addtn kwargs
    ReconMethod.IFFT_BASE: {}
}

# DATACLASSES

@dataclass
class Configs:
    # default methods list factory :)
    methods: list[ReconMethod] = field(default_factory=lambda: [ReconMethod.IFFT_BASE, ReconMethod.SENSE, ReconMethod.CS_L1, ReconMethod.GRAPPA])
    shape: list[int] = field(default_factory=lambda: [256, 256])
    dataset: str = "m4raw"
    setups: int = 3 # can be 0 if we want to omit setup tests
    runs: int = 5
    bench_mode: str = "slice" # "slice" | "volume"
    save_ims: bool = True
    output_level: str = "detail" # "detail" | "base" where "detail" publishes csv w all runs

@dataclass   
class MethodConfigs:
    im_bit_depth: str = "float32"
    ground_truth_im: np.ndarray | None = None 
    state: dict[str, Any] = None

    undersampling_mask: dict[str, Any] = field(default_factory=lambda:{
        "acs": 32,                     # ACS calibration region size (number of fully sampled ky lines in the center) -> from which weights are learned to interpolate missing k-space points
        "R": 2,                        # acceleration factor (how much of the data was undersampled)
        "seed": 42,                    # for deterministic runs
        "mask2d": None,                # created once at bench startup, shared across all methods
    })

    # method specific configs
    baseline_ifft: dict[str, Any] = field(default_factory=lambda:{
        "use_ifftshift": True,         # for when DC has been centered in kspace 
        "norm": "ortho",
        "debug_verify": False,
        "simulate_undersampling": False 
    })
    
    sense_espirit: dict[str, Any] = field(default_factory=lambda: {
        "debug_verify": True,       
        "sigpy_device": sp.Device(-1), # -1 represents CPU
        "calib": 32,                   # [ky,kx] size of fully-sampled central k-space window used to estimate coil maps by ESPIRiT
        "thresh": 0.02,                # eigenvalue thresh for keeping sensitivity modes in ESPIRiT
        "kernel_width": 6,             # size of the convolution kernel used in ESPIRiT calibration (how many k-space neighbors are used to model coil correlations)
        "max_iter": 50,                # number of iterations for the SENSE solver
        "lambda": 5e-3,                # regularization strength (0.0 is pure SENSE = no regularization)
        "ksp_dtype": np.complex128,    # start with complex64, also complex128 for more accuracy/lower efficiency
        "simulate_undersampling": False 
    })

    cs_l1_wavelet: dict[str, Any] = field(default_factory= lambda: {
        "debug_verify": True,
        "sigpy_device": sp.Device(-1), 
        "ksp_dtype": np.complex128,    
        "calib": 32,                   # [ky,kx] size of fully-sampled central k-space window used to estimate coil maps by ESPIRiT
        "thresh": 0.02,                # eigenvalue thresh for keeping sensitivity modes in ESPIRiT
        "kernel_width": 6,             # size of the convolution kernel used in ESPIRiT calibration (how many k-space neighbors are used to model coil correlations)
        "lambda": 1e-4,                # regularization , defaults 1e-3
        "max_iter": 50,                # max num iters in solving sparse objective
        "wavelet_basis": "db4",        # wavelet transform basis kernel that gets slid over image for decomposing into wavelets... 
                                       # db4 is standard default for mri; '4' refers to how many polynomial orders the wavelet "ignores" for better rep of curves
                                       # alternative: Haar's template [1, 1, -1, -1]: fires at edges (sees boxy transitions)
        "simulate_undersampling": True
    })

    grappa: dict[str, Any] = field(default_factory=lambda: {
        "kernel_size": (6,6),          # GRAPPA kernel size in (ky,kx), i.e. local neighborhood used to interpolate missing samples
        "coil_axis": 0,                # which axis is the coil dimension in kspace (C,H,W), axis C contains coils
        "debug_verify": False,
        "lambda": 1e-3,        
        "ksp_dtype": np.complex128,
        "simulate_undersampling": True # for fully acquired k-space datasets, need to simulate undersampling for cs to be tested appropriately
    })

    u_net_fft: dict[str, Any] = field(default_factory=lambda: {
        "simulate_undersampling": True
    }) 

@dataclass
class Payload_Out:
    method: ReconMethod
    shape: List[int]
    setup_time: List[float]     # for simulating init/prepare phase (plan FFTs, load model, allocate buffers, set up streams, etc)
    setup_memory: List[float]
    setup_power: List[float]
    runs_time: List[float]      # actual steady-state runs
    runs_power: List[float]
    runs_memory: List[float]

# HELPER FUNCTIONS

# Mask building to simulate undersampling (CS, GRAPPA)

def make_uniform_ky_mask(H: int, W: int, *, R: int, acs: int):
    """
    Uniform undersampling in ky with fully sampled ACS band, at desired acceleration factor (R).
    (i.e.: zeroing out certain phase encode lines to simulate what we would use in L1-wavelet recon)
    Returns mask2d (H,W) float32.
    Obsolete for CS methods because pseudo-random sampling is necessary -> NOT UNIFORM
    """
    mask = np.zeros((H, W), np.float32)

    # uniform ky lines
    mask[::R, :] = 1.0

    # ACS band
    cy = H // 2
    half = acs // 2
    # extract ACS region from center of k-space
    mask[max(cy - half, 0): min(cy + half, H), :] = 1.0
    return mask

def make_vds_ky_mask(H: int, W: int, *, R: int, acs: int, seed: int = 42) -> np.ndarray:
    """
    Variable-density random undersampling in ky (pseudo-random sampling REQUIRED for CS/L1-wavelet).
    Fully samples ACS center, randomly samples outer ky with density ~ 1/|ky|
    so lower frequencies are sampled more densely (matches MRI signal energy distribution).
    """
    mask = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(seed)

    cy = H // 2
    half_acs = acs // 2

    # (1) fully sample ACS band
    acs_lo = max(cy - half_acs, 0)
    acs_hi = min(cy + half_acs, H)
    mask[acs_lo:acs_hi, :] = 1.0

    # (2) variable density for outer ky lines
    # target: acquire (H/R - acs) lines from outside the ACS
    target_total = H // R
    target_outer = max(target_total - acs, 0)

    outer_lines = [i for i in range(H) if i < acs_lo or i >= acs_hi]

    # density ~ 1 / (distance from center + 1), normalized to probability
    dist = np.array([abs(i - cy) for i in outer_lines], dtype=np.float32)
    density = 1.0 / (dist + 1.0)
    density /= density.sum()

    n_outer = min(target_outer, len(outer_lines))
    chosen = rng.choice(outer_lines, size=n_outer, replace=False, p=density)
    mask[chosen, :] = 1.0

    return mask

