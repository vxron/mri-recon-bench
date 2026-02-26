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
import csv

# Generic type
T = TypeVar("T")

from bench.utils import Configs, MethodConfigs, Payload_Out, ReconMethod, DATA, ROOT, RESULTS, SETUP_KWARGS, make_vds_ky_mask
from bench.methods import get_method_fxn, get_setup_fxn, get_cleanup_fxn
from bench.data_loaders.m4raw import pick_first_h5, load_m4raw_kspace


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
        if "methods" in data:
            m = data["methods"]
            # allow string and convert to enum name!
            idx = 0
            for method in m:
                if isinstance(method, str) and method in ReconMethod.__members__: # check members (names), "STUB"
                    pass # correct format  
                elif isinstance(method, str) and method in [s.value for s in ReconMethod]: # check values (strings), "stub_fft_loop"
                    m[m.index(method)] = ReconMethod(method)
                else: 
                    # reset to default
                    m.remove(method)
                idx+=1

    # Only keep keys that exist in the dataclass
    valid_keys = {f.name for f in fields(datacls)}
    # Get key-value pairs where valid
    filtered = {k: v for k,v in data.items() if k in valid_keys}
    return datacls(**filtered)  # automatically constructs fields from strings while maintaining defaults


def write_detailed_csv(results: list[Payload_Out], output_path: Path):
    """
    Write detailed CSV with all individual runs.
    Format: Each row is one run, columns include method, run_idx, metric, value
    """
    rows = []
    
    for result in results:
        method = result.method
        
        # Setup runs
        for idx, (time_val, mem_val) in enumerate(zip(result.setup_time, result.setup_memory)):
            rows.append({
                "method": method,
                "phase": "setup",
                "run_idx": idx,
                "time_s": time_val,
                "memory_bytes": mem_val,
            })
        
        # Steady-state runs
        for idx, (time_val, mem_val) in enumerate(zip(result.runs_time, result.runs_memory)):
            rows.append({
                "method": method,
                "phase": "run",
                "run_idx": idx,
                "time_s": time_val,
                "memory_bytes": mem_val,
            })
    
    if not rows:
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["method", "phase", "run_idx", "time_s", "memory_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(results: list[Payload_Out], output_path: Path):
    """
    Write summary CSV with aggregated statistics.
    Format: Each row is one method, columns show mean/std for each metric
    """
    rows = []
    
    for result in results:
        row = {
            "method": result.method,
            "shape": "x".join(map(str, result.shape)),
        }
        
        # Setup statistics
        if result.setup_time:
            row["setup_time_mean_s"] = np.mean(result.setup_time)
            row["setup_time_std_s"] = np.std(result.setup_time)
            row["setup_memory_mean_MB"] = np.mean(result.setup_memory) / 1e6
            row["setup_memory_std_MB"] = np.std(result.setup_memory) / 1e6
        else:
            row["setup_time_mean_s"] = None
            row["setup_time_std_s"] = None
            row["setup_memory_mean_MB"] = None
            row["setup_memory_std_MB"] = None
        
        # Run statistics
        row["run_time_mean_s"] = np.mean(result.runs_time)
        row["run_time_std_s"] = np.std(result.runs_time)
        row["run_memory_mean_MB"] = np.mean(result.runs_memory) / 1e6
        row["run_memory_std_MB"] = np.std(result.runs_memory) / 1e6
        
        rows.append(row)
    
    if not rows:
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # (1) INIT CONFIGS
    cfg = Configs()
    methodCfg = MethodConfigs()
    currMethod_ = ReconMethod.IFFT_BASE

    args = parse_args()
    if args.config is not None:
        cfg = construct_dataclass_from_json(args.config, Configs) # pass type itself
    print(cfg.methods, cfg.shape, cfg.dataset, cfg.setups, cfg.runs)
    
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
    
    # (3) SETUP CONFIGS & PREPROCESS TO FIT EXPECTED SHAPE FOR RECON METHOD 
    if kspace_in.ndim == 4 and cfg.bench_mode == "slice": # often given (S,C,H,W), ex. in m4raw
        # narrow to a single z-slice -> (C,H,W)
        S = kspace_in.shape[0]
        kspace_in = kspace_in[S // 2] # select the middle slice
        rss_gt_slice = rss_gt[S // 2] 
        if rss_gt_slice is not None:
            methodCfg.ground_truth_im = rss_gt_slice
        else:
            methodCfg.ground_truth_im = None
        print("Input to recon:", kspace_in.shape, kspace_in.dtype, "ndim", kspace_in.ndim)

    # (4) BUILD REUSABLE UNDERSAMPLING MASK IN KY
    # for the methods that work on undersampled k-space (simulate_undersampling true)
    H, W = kspace_in.shape[1], kspace_in.shape[2]
    R = methodCfg.undersampling_mask.get("R", 2)
    acs = methodCfg.undersampling_mask.get("acs", 32)
    seed = methodCfg.undersampling_mask.get("seed", 42)
    shared_mask = make_vds_ky_mask(H, W, R=R, acs=acs, seed=seed)
    methodCfg.undersampling_mask["mask2d"] = shared_mask

    # (5) RUN RECON METHOD
    if cfg.output_level == "detail":
        all_results = [] # for csv export
    for meth in cfg.methods:
        currMethod_ = meth
        recon = get_method_fxn(meth)
        setup = get_setup_fxn(meth, **SETUP_KWARGS.get(meth,{}))
        cleanup = get_cleanup_fxn(meth)

        # 1) setup (one-time cost for allocations, plans, buffers, model init, etc)
        setup_times = []
        setup_mem_use_peak = []
        if cfg.setups > 0:
            for _ in range(cfg.setups):
                gc.collect()
                setup_mem, setup_time = setup(kspace_in, methodCfg)
                setup_times.append(setup_time)
                setup_mem_use_peak.append(setup_mem)

        # 2) steady-state running 
        runtimes = []
        runs_mem_use_peak = []
        for _ in range(cfg.runs):
            gc.collect() # reduce garbage noise between runs
            y, peak, time_elapsed = recon(kspace_in, methodCfg)
            runtimes.append(time_elapsed)
            runs_mem_use_peak.append(peak)

        # reset state for next runs
        cleanup(methodCfg)

        # (6) GENERATE OUTPUT RESULTS
        out = Payload_Out(
            method = meth.value,
            shape = list(kspace_in.shape),
            runs_time = [float(t) for t in runtimes],
            runs_power=[],
            runs_memory=runs_mem_use_peak,
            setup_memory=setup_mem_use_peak,
            setup_power=[],
            setup_time=[float(t) for t in setup_times] 
        )
        if cfg.output_level == "detail":
            all_results.append(out)

        # visualize image
        if cfg.save_ims and rss_gt is not None and cfg.bench_mode == "slice":
            out_dir = RESULTS / "images" / f"{cfg.dataset}" / meth.value
            out_dir.mkdir(parents=True, exist_ok=True)
            pred = y
            gt = methodCfg.ground_truth_im
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

        # Save individual JSON
        out_json = asdict(out)
        Path("results").mkdir(exist_ok=True)
        out_path = Path("results") / f"run_{meth.value}.json"
        out_path.write_text(json.dumps(out_json, indent=2))
        print(f"Wrote {out_path}")

        # (7) CLEANUP FOR NEXT METHOD
        cleanup(methodCfg)
    
    # All data collected -> Write CSV logs
    if cfg.output_level == "detail":
        csv_dir = RESULTS / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        detail_path = csv_dir / "benchmark_detailed.csv"
        summary_path = csv_dir / "benchmark_summary.csv"
        write_detailed_csv(all_results, detail_path)
        write_summary_csv(all_results, summary_path)

if __name__ == "__main__":
    main()
