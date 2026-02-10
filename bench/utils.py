from enum import Enum
from dataclasses import dataclass, fields, asdict, field
from typing import List, Any
import numpy as np
import sigpy as sp

class ReconMethod(str, Enum):
    CS_L1 = "cs_l1wavelet"
    UNET = "dl_unet"
    SWIN = "dl_swin"
    IFFT_BASE = "baseline_ifft"
    SENSE = "sense_espirit"

@dataclass
class Configs:
    method: ReconMethod = ReconMethod.IFFT_BASE
    shape: list[int] = field(default_factory=lambda: [256, 256])
    dataset: str = "m4raw"
    setups: int = 3 # can be 0 if we want to omit setup tests
    runs: int = 5
    bench_mode: str = "slice" # "slice" | "volume"
    save_ims: bool = True

@dataclass   
class MethodConfigs:
    im_bit_depth: str = "float32"
    ground_truth_im: np.ndarray = None
    state: dict[str, Any] = None

    # method specific configs
    baseline_ifft: dict[str, Any] = field(default_factory=lambda:{
        "use_ifftshift": True, # for when DC has been centered in kspace 
        "norm": "ortho",
        "debug_verify": False,
    })
    
    sense_espirit: dict[str, Any] = field(default_factory=lambda: {
        "debug_verify": False,
        "sigpy_device": sp.cpu_device,
        "calib": [24,24],           # [ky,kx] size of fully-sampled central k-space window used to estimate coil maps by ESPIRiT
        "thresh": 0.02,             # eigenvalue thresh for keeping sensitivity modes in ESPIRiT
        "kernel_width": 6,          # size of the convolution kernel used in ESPIRiT calibration (how many k-space neighbors are used to model coil correlations)
        "max_iter": 30,             # number of iterations for the SENSE solver
        "lambda": 0.0,              # regularization strength (0.0 is pure SENSE = no regularization)
        "shift_DC": False,          # should be true if DC is centered in the kspace arrays
        "ksp_dtype": "complex64"    # start with complex64, also complex128 for more accuracy/lower efficiency
    })


@dataclass
class Payload_Out:
    method: ReconMethod
    shape: List[int]
    setup_time: List[float] # for simulating init/prepare phase (plan FFTs, load model, allocate buffers, set up streams, etc)
    setup_memory: List[float]
    setup_power: List[float]
    runs_time: List[float] # actual steady-state runs
    runs_power: List[float]
    runs_memory: List[float]

