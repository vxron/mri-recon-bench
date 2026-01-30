from enum import Enum
from dataclasses import dataclass, fields, asdict, field
from typing import List, Any

class ReconMethod(str, Enum):
    CS_L1 = "cs_l1wavelet"
    UNET = "dl_unet"
    SWIN = "dl_swin"
    IFFT_BASE = "baseline_ifft"

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
    baseline_ifft: dict[str, Any] = field(default_factory=lambda:{
        "state": {}, # dictionary
        "use_ifftshift": True,
        "norm": "ortho",
        "im_bit_depth": "float32",
        "debug_verify": False,
        "ground_truth_im": None
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

