"""Extract five ROIs and FFT panels from the raw 200 kV, 500 kX TEM image."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mxene_qc_tem.image_analysis import load_grayscale, crop_roi, fft_log_magnitude

RAW = ROOT / "data" / "raw" / "OneView_200kV_500kX_36135.tif"
OUT = ROOT / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ROIS = {
    "ROI-1": (850, 800, 380, 380),
    "ROI-2": (600, 1150, 380, 380),
    "ROI-3": (1150, 600, 380, 380),
    "ROI-4": (350, 750, 380, 380),
    "ROI-5": (1200, 1050, 380, 380),
}

img = load_grayscale(str(RAW))
for name, (x, y, w, h) in ROIS.items():
    roi = crop_roi(img, x, y, w, h)
    fft = fft_log_magnitude(roi)
    for suffix, arr in [("roi", roi), ("fft", fft)]:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        fig.savefig(OUT / f"{name}_{suffix}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
print(f"Saved ROI and FFT panels to {OUT}")
