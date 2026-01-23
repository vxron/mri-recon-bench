from . import stub
from bench.utils import ReconMethod

# map enum to function on methods module init
METHODS = {
  ReconMethod.STUB: stub.reconstruct,
}

def get_method_fxn(method: ReconMethod):
    try:
        return METHODS[method]
    except KeyError:
        raise ValueError(f"Unknown method: {method}")