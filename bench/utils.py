from enum import Enum
from dataclasses import dataclass, fields, asdict, field
from typing import List

class ReconMethod(str, Enum):
    STUB = "stub_fft_loop"
    CS_L1 = "cs_l1wavelet"
    UNET = "dl_unet"
    SWIN = "dl_swin"

@dataclass
class Configs:
    method: ReconMethod = ReconMethod.STUB
    shape: list[int] = field(default_factory=lambda: [256, 256])
    dataset: str = "stub"
    warmup: int = 1
    runs: int = 5

@dataclass
class Payload_Out:
    method: ReconMethod
    shape: List[int]
    runs_time: List[float]
    runs_power: List[float]


