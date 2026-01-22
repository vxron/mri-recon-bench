# bench/run_bench.py
import json, time
from pathlib import Path
import numpy as np

def fake_recon(x: np.ndarray) -> np.ndarray:
    # placeholder compute (acts like a "recon method" stub for now)
    for _ in range(20):
        x = np.fft.ifft2(np.fft.fft2(x))
    return np.abs(x)

def main():
    rng = np.random.default_rng(0)
    # placeholder k space input
    x = rng.normal(size=(256, 256)) + 1j * rng.normal(size=(256, 256))

    # warmup
    _ = fake_recon(x)

    # run a couple times 
    runs = []
    for i in range(5):
        t0 = time.perf_counter()
        y = fake_recon(x)
        time_elapsed = time.perf_counter() - t0
        runs.append(time_elapsed)

    out = {
        "method": "stub_fft_loop",
        "input_shape": [256, 256],
        "runs_s": runs,
        "mean_s": float(np.mean(runs)),
        "std_s": float(np.std(runs)),
    }

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / "run_stub.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
