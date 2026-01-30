from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np

def pick_first_h5(root: str | Path) -> Path:
    root = Path(root)
    files = sorted(root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {root}")
    return files[0]

def load_m4raw_kspace(path: str | Path) -> tuple[np.ndarray, tuple | None, dict]:
    """
    Returns:
      kspace: complex array with shape (S, C, H, W)
      - S: Slice (z-dir, slice-select)
      - H: Height (x-dir, frequency encoding)
      - W: Width (y-dir, phase encoding)
      - C: Coil (the scanner records a separate k-space for each receive coil)
      meta: dict with a few useful attrs
    """
    path = Path(path)
    with h5py.File(path, "r") as hf:
        if "kspace" not in hf:
            raise KeyError(f"'kspace' not found. Keys={list(hf.keys())}")
        rss_gt = hf["reconstruction_rss"][()] if "reconstruction_rss" in hf else None
        kspace = hf["kspace"][()]  # should already be complex
        meta = {
            "keys": list(hf.keys()),
            "attrs": {k: hf.attrs[k] for k in hf.attrs.keys()},
            "rss_shape": (tuple(rss_gt.shape) if rss_gt is not None else None),
            "kspace_shape": tuple(kspace.shape),
        }
    return kspace, rss_gt, meta
