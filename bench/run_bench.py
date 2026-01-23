# bench/run_bench.py
import json, time
from pathlib import Path
import numpy as np
import argparse
from dataclasses import dataclass, fields, asdict
from typing import TypeVar, Type

# Generic type
T = TypeVar("T")

from bench.utils import Configs, Payload_Out, ReconMethod
from bench.methods import get_method_fxn

def parse_args():
    # Parse arguments from JSON config if given
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to JSON config file", default=None, required=False)
    return parser.parse_args()

def construct_dataclass_from_json(
    path: str | Path, datacls: Type[T]
    ) -> T:
    """
    Read JSON from 'path' and pull key-value pairs to construct dataclass of type 'T'.
    """
    data = json.loads(Path(path).read_text()) # builds dict

    if datacls is Configs:
        # make sure 'method' value is within enum
        if "method" in data:
            m = data["method"]
            # allow string and convert to enum name!
            if isinstance(m, str) and m in ReconMethod.__members__: # check members (names), "STUB"
                data["method"] = ReconMethod[m] 
            elif isinstance(m,str) and m in [s.value for s in ReconMethod]: # check values (strings), "stub_fft_loop"
                data["method"] = ReconMethod(m) 
            else: 
                # reset to default
                data["method"] = ReconMethod.STUB

    # Only keep keys that exist in the dataclass
    valid_keys = {f.name for f in fields(datacls)}
    # Get key-value pairs where valid
    filtered = {k: v for k,v in data.items() if k in valid_keys}
    return datacls(**filtered)  # automatically constructs fields from strings while maintaining defaults

def main():
    # init configs
    cfg = Configs()
    args = parse_args()
    if args.config is not None:
        cfg = construct_dataclass_from_json(args.config, Configs) # pass type itself
    print(cfg.method, cfg.shape, cfg.dataset, cfg.warmup, cfg.runs)

    rng = np.random.default_rng(0)
    # placeholder k space input
    x = rng.normal(size=tuple(cfg.shape)) + 1j * rng.normal(size=tuple(cfg.shape))

    # run recon method
    recon = get_method_fxn(cfg.method)

    # warmup
    for _ in range(cfg.warmup):
        _ = recon(x, cfg)

    # run a couple times 
    runs = []
    for i in range(cfg.runs):
        t0 = time.perf_counter()
        y = recon(x, cfg)
        time_elapsed = time.perf_counter() - t0
        runs.append(time_elapsed)

    out = Payload_Out(
        method = cfg.method.value,
        shape = list(cfg.shape),
        runs_time = [float(t) for t in runs],
        runs_power=[]
    )
    
    # out.runs_power
    # pack for json
    out_json = asdict(out)

    # find reults directory
    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"run_{cfg.method.value}.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
