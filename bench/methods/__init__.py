from bench.utils import ReconMethod
from . import baseline_ifft
from . import sense_espirit
from functools import partial
from . import grappa

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
  ReconMethod.SENSE: sense_espirit.run_sense_solver,
  ReconMethod.CS_L1: sense_espirit.run_l1wavelet_solver,
  ReconMethod.GRAPPA: grappa.run_grappa,
}

# map recon method enum to prepare function
PREPARE_SETUP = {
    ReconMethod.IFFT_BASE: baseline_ifft.preallocate_buffers,
    ReconMethod.SENSE: sense_espirit.setup_and_espirit,
    ReconMethod.CS_L1: sense_espirit.setup_and_espirit,
    ReconMethod.GRAPPA: grappa.preallocate_buffers,
}

# map recon method enum to cleanup function after algo has completed fully
CLEANUP = {
    ReconMethod.IFFT_BASE: baseline_ifft.cleanup,
    ReconMethod.SENSE: sense_espirit.cleanup,
    ReconMethod.CS_L1: sense_espirit.cleanup,
    ReconMethod.GRAPPA: grappa.cleanup,
}

def get_method_fxn(method: ReconMethod):
    try:
        return METHODS[method]
    except KeyError:
        raise ValueError(f"Unknown method: {method}")
    
def get_setup_fxn(method: ReconMethod, **kwargs):
    try:
        fxn = PREPARE_SETUP[method]
        return partial(fxn, **kwargs) if kwargs else fxn
    except KeyError:
        raise ValueError(f"Unknown method: {method}")

def get_cleanup_fxn(method: ReconMethod):
    try:
        return CLEANUP[method]
    except KeyError:
        raise ValueError(f"Unknown method: {method}")