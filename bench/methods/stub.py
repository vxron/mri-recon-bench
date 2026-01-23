import numpy as np

# placeholder compute (acts like a "recon method" stub for now)
def reconstruct(x: np.ndarray, cfg) -> np.ndarray:
    for _ in range(20):
        x = np.fft.ifft2(np.fft.fft2(x))
    return np.abs(x)
