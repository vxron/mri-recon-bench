from bench.utils import ReconMethod
from . import baseline_ifft

'''
METHODS ARHICTECTURE: each ReconMethod exposes:
1) setup(kspace_example, methodCfg) -> state
2) recon (kspace_frame, methodCfg, state) -> out
The state contains any prelloacted buffers, loaded models, or FFT plans for recon.
The state also contains shape/dtype so we can sanity-check streaming inputs.
'''

# map recon method enum to function on methods module init
METHODS = {
  ReconMethod.IFFT_BASE: baseline_ifft.baseline_ifft,
}

# map recon method enum to prepare function
PREPARE_SETUP = {
    ReconMethod.IFFT_BASE: baseline_ifft.preallocate_buffers,
} 

def get_method_fxn(method: ReconMethod):
    try:
        return METHODS[method]
    except KeyError:
        raise ValueError(f"Unknown method: {method}")
    
def get_setup_fxn(method: ReconMethod):
    try:
        return PREPARE_SETUP[method]
    except KeyError:
        raise ValueError(f"Unknown method: {method}")