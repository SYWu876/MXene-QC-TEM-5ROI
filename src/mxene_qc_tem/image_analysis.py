
"""TEM ROI/FFT helper functions."""

from __future__ import annotations
import numpy as np
from PIL import Image

def load_grayscale(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))

def crop_roi(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return image[y:y+h, x:x+w]

def fft_log_magnitude(roi: np.ndarray) -> np.ndarray:
    arr = roi.astype(float)
    arr -= arr.mean()
    win = np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(arr * win))
    return np.log1p(np.abs(F))

def one_dimensional_spacing(profile: np.ndarray, nm_per_px: float, low_freq_cut: float = 0.02) -> tuple[float, float]:
    """Return (f_peak cycles/px, d_nm) from a detrended 1D profile."""
    y = np.asarray(profile, dtype=float)
    x = np.arange(len(y))
    coeff = np.polyfit(x, y, 1)
    signal = (y - np.polyval(coeff, x)) * np.hanning(len(y))
    F = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1)
    mag = np.abs(F)
    mask = freqs > low_freq_cut
    f_peak = float(freqs[mask][np.argmax(mag[mask])])
    return f_peak, float(nm_per_px / f_peak)
