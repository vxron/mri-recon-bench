# bench/run_bench.py
import json, time
from pathlib import Path
import numpy as np
import argparse
from dataclasses import dataclass, fields, asdict
from typing import TypeVar, Type
import tracemalloc # to get pk memory usage throughout diff algos
import gc
import matplotlib.pyplot as plt

# Generic type
T = TypeVar("T")

from bench.utils import Configs, MethodConfigs, Payload_Out, ReconMethod
from bench.methods import get_method_fxn, get_setup_fxn
from bench.data_loaders.m4raw import pick_first_h5, load_m4raw_kspace

# Datapaths
ROOT = Path(__file__).resolve().parents[1]   # mri-recon-bench/
DATA = ROOT / "data"
RESULTS = ROOT / "results"

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
    # (1) INIT CONFIGS
    cfg = Configs()
    methodCfg = MethodConfigs()

    args = parse_args()
    if args.config is not None:
        cfg = construct_dataclass_from_json(args.config, Configs) # pass type itself
    print(cfg.method, cfg.shape, cfg.dataset, cfg.setups, cfg.runs)
    
    # (2) INPUT RAW K-SPACE DATA FROM DATASET
    match cfg.dataset:
        case "m4raw":
            DATA_M4RAW = DATA / "m4raw_zenodo" / "multicoil_test"
            h5_path = pick_first_h5(DATA_M4RAW)
            kspace_in, rss_gt, meta = load_m4raw_kspace(h5_path)
            print("Loaded M4Raw:", h5_path.name)
            print("Keys:", meta["keys"])
            print("Attrs:", meta["attrs"])
            print("Kspace shape:", meta["kspace_shape"])
        case _: # default
            print("No matching dataset found. Check configs.")
            return
    
    # (3) PREPROCESS TO FIT EXPECTED SHAPE FOR RECON METHOD
    if kspace_in.ndim == 4 and cfg.bench_mode == "slice": # often given (S,C,H,W), ex. in m4raw
        # narrow to a single z-slice -> (C,H,W)
        S = kspace_in.shape[0]
        kspace_in = kspace_in[S // 2] # select the middle slice
        rss_gt_slice = rss_gt[S // 2] 
        if rss_gt is not None:
            methodCfg.baseline_ifft["ground_truth_im"] = rss_gt_slice
            methodCfg.baseline_ifft["debug_verify"] = False # IMPORTANT: MUST BE FALSE TO KEEP REPRESENTATIVE TIMING (NOT INCLUDE DEBUG TIMING)
        else:
            methodCfg.baseline_ifft["ground_truth_im"] = None
            methodCfg.baseline_ifft["debug_verify"] = False
        print("Input to recon:", kspace_in.shape, kspace_in.dtype, "ndim", kspace_in.ndim)

    # (4) RUN RECON METHOD (TODO: make loop to iter thru recon of all methods)
    recon = get_method_fxn(cfg.method)
    setup = get_setup_fxn(cfg.method)

    # 1) setup (one-time cost for allocations, plans, buffers, model init, etc)
    setup_times = []
    setup_mem_use_peak = []
    if cfg.setups > 0:
        for _ in range(cfg.setups):
            gc.collect()
            t0 = time.perf_counter()
            setup_mem, setup_time = setup(kspace_in, methodCfg)
            setup_times.append(setup_time)
            setup_mem_use_peak.append(setup_mem)

    # 2) steady-state running 
    runtimes = []
    runs_mem_use_peak = []
    tracemalloc.start() # start for each new recon algo we want to test
    for _ in range(cfg.runs):
        gc.collect() # reduce garbage noise between runs
        
        t0 = time.perf_counter()
        
        y = recon(kspace_in, methodCfg)
        
        _, peak = tracemalloc.get_traced_memory()
        time_elapsed = time.perf_counter() - t0
        runtimes.append(time_elapsed)
        runs_mem_use_peak.append(peak)

        tracemalloc.reset_peak() # reset: start measuring peak from now for next recon round 
    
    tracemalloc.stop()

    # reset state for next runs
    methodCfg.state.clear()

    # (5) GENERATE OUTPUT RESULTS
    out = Payload_Out(
        method = cfg.method.value,
        shape = list(kspace_in.shape),
        runs_time = [float(t) for t in runtimes],
        runs_power=[],
        runs_memory=runs_mem_use_peak,
        setup_memory=setup_mem_use_peak,
        setup_power=[],
        setup_time=[float(t) for t in setup_times] 
    )
    # visualize image
    if cfg.save_ims and rss_gt is not None and cfg.bench_mode == "slice":
        out_dir = RESULTS / "images" / f"{cfg.dataset}" / cfg.method.value
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = y
        gt = methodCfg.baseline_ifft["ground_truth_im"]
        # simple window for display
        def norm01(im):
            im = np.asarray(im)
            lo, hi = np.percentile(im, [1, 99])
            im = np.clip(im, lo, hi)
            return (im - lo) / (hi - lo + 1e-12)
        plt.imsave(out_dir / "pred.png", norm01(pred), cmap="gray")
        plt.imsave(out_dir / "gt.png", norm01(gt), cmap="gray")
        plt.imsave(out_dir / "diff.png", np.abs(norm01(pred) - norm01(gt)), cmap="gray")
        print("Wrote vis to", out_dir)

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
