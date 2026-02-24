from enum import Enum
from dataclasses import dataclass, fields, asdict, field
from typing import List, Any
import numpy as np
import sigpy as sp
from pathlib import Path

# Datapaths
ROOT = Path(__file__).resolve().parents[1]   # mri-recon-bench/
DATA = ROOT / "data"
RESULTS = ROOT / "results"

class ReconMethod(str, Enum):
    CS_L1 = "cs_l1wavelet"
    UNET = "dl_unet"
    SWIN = "dl_swin"
    IFFT_BASE = "baseline_ifft"
    SENSE = "sense_espirit"

@dataclass
class Configs:
    # default methods list factory :)
    methods: list[ReconMethod] = field(default_factory=lambda: [ReconMethod.IFFT_BASE, ReconMethod.SENSE, ReconMethod.CS_L1])
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

    # method specific configs
    baseline_ifft: dict[str, Any] = field(default_factory=lambda:{
        "use_ifftshift": True, # for when DC has been centered in kspace 
        "norm": "ortho",
        "debug_verify": False,
    })
    
    sense_espirit: dict[str, Any] = field(default_factory=lambda: {
        "debug_verify": True,       
        "sigpy_device": sp.Device(-1), # -1 represents CPU
        "calib": 32,                   # [ky,kx] size of fully-sampled central k-space window used to estimate coil maps by ESPIRiT
        "thresh": 0.02,                # eigenvalue thresh for keeping sensitivity modes in ESPIRiT
        "kernel_width": 6,             # size of the convolution kernel used in ESPIRiT calibration (how many k-space neighbors are used to model coil correlations)
        "max_iter": 50,                # number of iterations for the SENSE solver
        "lambda": 0.005,               # regularization strength (0.0 is pure SENSE = no regularization)
        "ksp_dtype": np.complex128     # start with complex64, also complex128 for more accuracy/lower efficiency
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
        "acs": 32,                     # fully sampled center of k-space (number of fully sampled central lines)
        "R": 2,                        # acceleration factor (how much of the data was undersampled)
        "simulate_undersampling": True # for fully acquired k-space datasets, need to simulate undersampling for cs to be tested appropriately
    })

# For cases where setup fxns take additional kwargs
SETUP_KWARGS = {
    ReconMethod.SENSE: {"curr_method": ReconMethod.SENSE},
    ReconMethod.CS_L1: {"curr_method": ReconMethod.CS_L1}
}

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


