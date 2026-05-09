#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MXene five-ROI classical local-ordering-index (LOI) analysis.

Run example:
    python mxene_classical_loi_analysis.py --input-dir ./images --output-dir ./mxene_loi_output

Input files can be either cropped:
    ROI-1_TEM_crop.png ... ROI-5_TEM_crop.png
    ROI-1_FFT_crop.png ... ROI-5_FFT_crop.png

or full exported panels:
    ROI1_TEM500kX.png ... ROI5_TEM500kX.png
    FFT1_TEM500kX.png ... FFT5_TEM500kX.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths

ROI_NAMES = [f"ROI-{i}" for i in range(1, 6)]

QUANTUM_REFERENCE = pd.DataFrame({
    "ROI": ROI_NAMES,
    "d_parallel_nm": [0.7300, 0.7800, 0.3300, 0.9900, 0.7343],
    "t_parallel_Ha": [0.452, 0.423, 1.000, 0.333, 0.449],
    "Delta_Ha": [0.221, 0.236, 0.100, 0.300, 0.222],
    "g_Ha": [0.050, 0.050, 0.050, 0.050, 0.050],
    "E0_exact_Ha": [-0.6790, -0.6650, -1.1120, -0.6370, -0.6770],
})

def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0

def conservative_crop_from_exported_panel(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    if h >= 1500 and w >= 1450:
        return img[180:min(1560, h), 60:min(1446, w)]
    return img

def find_image_pair(input_dir: Path, roi_index: int) -> Tuple[Path, Path]:
    roi_label = f"ROI-{roi_index}"
    tem_candidates = [
        input_dir / f"{roi_label}_TEM_crop.png",
        input_dir / f"ROI{roi_index}_TEM500kX.png",
        input_dir / f"ROI_{roi_index}_TEM500kX.png",
        input_dir / f"ROI{roi_index}.png",
    ]
    fft_candidates = [
        input_dir / f"{roi_label}_FFT_crop.png",
        input_dir / f"FFT{roi_index}_TEM500kX.png",
        input_dir / f"FFT_{roi_index}_TEM500kX.png",
        input_dir / f"FFT{roi_index}.png",
    ]
    tem_path = next((p for p in tem_candidates if p.exists()), None)
    fft_path = next((p for p in fft_candidates if p.exists()), None)
    if tem_path is None:
        raise FileNotFoundError(f"Cannot find TEM image for ROI-{roi_index}.")
    if fft_path is None:
        raise FileNotFoundError(f"Cannot find FFT image for ROI-{roi_index}.")
    return tem_path, fft_path

def fft_peak_descriptors_from_fft_image(fft_img: np.ndarray) -> Tuple[float, float, int]:
    bg = gaussian_filter(fft_img, sigma=20)
    proc = np.clip(fft_img - bg, 0, None)
    y, x = np.indices(proc.shape)
    cy, cx = (np.array(proc.shape) - 1) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = np.floor(r).astype(int)
    radial = np.bincount(r_int.ravel(), weights=proc.ravel()) / np.maximum(np.bincount(r_int.ravel()), 1)
    radial_s = uniform_filter1d(radial, size=5)
    start, stop = 20, min(len(radial_s) // 2, 250)
    prof = radial_s[start:stop]
    peaks, props = find_peaks(prof, prominence=np.std(prof) * 0.2)
    if len(peaks) == 0:
        peak_index = int(np.argmax(prof) + start)
    else:
        peak_index = int(peaks[np.argmax(props["prominences"] * prof[peaks])] + start)
    width_result = peak_widths(radial_s, [peak_index], rel_height=0.5)
    delta_q_eff = float(width_result[0][0])
    idx = np.arange(start, stop)
    bg_mask = np.abs(idx - peak_index) > 2 * max(delta_q_eff, 1.0)
    local_bg = np.median(radial_s[start:stop][bg_mask]) if np.any(bg_mask) else np.median(radial_s[start:stop])
    p_fft = float(radial_s[peak_index] / (local_bg + 1e-12))
    return p_fft, delta_q_eff, peak_index

def roi_real_space_descriptors(tem_img: np.ndarray) -> Tuple[float, float]:
    hp = tem_img - gaussian_filter(tem_img, sigma=6)
    hp = (hp - hp.mean()) / (hp.std() + 1e-12)
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(hp)) ** 2).real)
    cy, cx = (np.array(ac.shape) - 1) / 2.0
    y, x = np.indices(ac.shape)
    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ac = ac / (ac.max() + 1e-12)
    a_off = float(ac[(rr > 5) & (rr < min(tem_img.shape) / 4)].max())
    smoothed = gaussian_filter(tem_img, sigma=1)
    gy, gx = np.gradient(smoothed)
    jxx = gaussian_filter(gx * gx, sigma=4)
    jyy = gaussian_filter(gy * gy, sigma=4)
    jxy = gaussian_filter(gx * gy, sigma=4)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-12)
    c_loc = float(np.quantile(coherence, 0.95))
    return a_off, c_loc

def calculate_loi(descriptor_df: pd.DataFrame, eps: float = 0.05) -> pd.DataFrame:
    df = descriptor_df.copy()
    norm_cols = []
    for col in ["P_FFT", "A_off", "C_loc", "Delta_q_eff"]:
        values = df[col].to_numpy(float)
        norm = (values - values.min()) / (values.max() - values.min() + 1e-12)
        if col == "Delta_q_eff":
            norm = 1.0 - norm
        norm_col = col + "_norm"
        df[norm_col] = norm
        norm_cols.append(norm_col)
    df["LOI"] = 100.0 * df[norm_cols].mean(axis=1)
    norm_values = df[norm_cols].to_numpy(float)
    geom_score = np.prod(norm_values + eps, axis=1) ** (1.0 / norm_values.shape[1])
    df["w_LOI"] = geom_score / geom_score.sum()
    return df

def boxed_axes(ax) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(labelsize=14, width=1.2, length=5)
    ax.set_facecolor("white")

def save_image_grid(images: List[np.ndarray], labels: List[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), constrained_layout=True)
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img, cmap="gray")
        ax.set_title(label, fontsize=17, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        boxed_axes(ax)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def ensemble_summary(df: pd.DataFrame, quantum_df: pd.DataFrame) -> pd.DataFrame:
    merged = quantum_df.merge(df[["ROI", "LOI", "w_LOI"]], on="ROI")
    def row(name: str, weights: np.ndarray) -> dict:
        return {
            "ensemble": name,
            "d_parallel_nm": float(np.sum(weights * merged["d_parallel_nm"])),
            "t_parallel_Ha": float(np.sum(weights * merged["t_parallel_Ha"])),
            "Delta_Ha": float(np.sum(weights * merged["Delta_Ha"])),
            "E0_exact_Ha": float(np.sum(weights * merged["E0_exact_Ha"])),
        }
    equal_weights = np.ones(len(merged)) / len(merged)
    return pd.DataFrame([
        row("Equal-area geometric", equal_weights),
        row("LOI-weighted classical ensemble", merged["w_LOI"].to_numpy(float)),
    ])

def make_plots(df: pd.DataFrame, quantum_df: pd.DataFrame, tem_images: List[np.ndarray], fft_images: List[np.ndarray], output_dir: Path) -> None:
    save_image_grid(tem_images, ROI_NAMES, output_dir / "panel_a_ROI_TEM_images.png")
    save_image_grid(fft_images, [f"FFT-{i}" for i in range(1, 6)], output_dir / "panel_b_ROI_FFT_maps.png")

    plot_df = df[["ROI", "P_FFT", "A_off", "C_loc", "Delta_q_eff", "LOI"]].copy().set_index("ROI")
    heat_vals = plot_df.to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    im = ax.imshow(heat_vals, aspect="auto")
    ax.set_xticks(range(plot_df.shape[1]))
    ax.set_xticklabels(["P_FFT", "A_off", "C_loc", "Δq_eff", "LOI"], rotation=30, ha="right", fontsize=14)
    ax.set_yticks(range(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index, fontsize=14)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            txt = f"{heat_vals[i, j]:.2f}" if j < 4 else f"{heat_vals[i, j]:.1f}"
            t = ax.text(j, i, txt, ha="center", va="center", fontsize=12, color="white")
            t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="black")])
    boxed_axes(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=14)
    ax.set_title("Classical descriptor heatmap", fontsize=20, pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / "panel_c_descriptor_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.bar(df["ROI"], df["LOI"])
    ax.set_ylabel("MXene LOI (0–100)", fontsize=17)
    ax.set_xlabel("ROI", fontsize=17)
    ax.set_title("Local ordering index across MXene ROIs", fontsize=20, pad=12)
    ax.set_ylim(0, df["LOI"].max() * 1.18)
    boxed_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "panel_d_MXene_LOI_barplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.bar(df["ROI"], df["w_LOI"])
    ax.set_ylabel("LOI-derived patch weight", fontsize=17)
    ax.set_xlabel("ROI", fontsize=17)
    ax.set_title("Classical LOI-derived patch weights", fontsize=20, pad=12)
    ax.set_ylim(0, df["w_LOI"].max() * 1.20)
    boxed_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "panel_e_LOI_patch_weights.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    merged = quantum_df.merge(df[["ROI", "LOI", "w_LOI"]], on="ROI")
    m = merged.sort_values("LOI")
    coef = np.polyfit(m["LOI"], m["E0_exact_Ha"], 1)
    xs = np.linspace(m["LOI"].min(), m["LOI"].max(), 200)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(m["LOI"], m["E0_exact_Ha"], s=110)
    ax.plot(xs, np.polyval(coef, xs), "--", linewidth=1.8)
    ax.set_xlabel("MXene LOI (0–100)", fontsize=17)
    ax.set_ylabel(r"Exact ground-state energy, $E_0$ (Ha)", fontsize=17)
    ax.set_title("Classical LOI vs motif-resolved quantum energy", fontsize=20, pad=12)
    boxed_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "panel_f_LOI_vs_E0exact.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    x_centroid = float(np.sum(df["LOI"] * df["w_LOI"]))
    y_centroid = float(np.sum(df["Delta_q_eff"] * df["w_LOI"]))
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(df["LOI"], df["Delta_q_eff"], s=110)
    ax.scatter([x_centroid], [y_centroid], marker="*", s=280)
    ax.set_xlabel("MXene LOI (0–100)", fontsize=17)
    ax.set_ylabel("Effective FFT peak width, Δq_eff", fontsize=17)
    ax.set_title("LOI–Δq_eff order–disorder map", fontsize=20, pad=12)
    boxed_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "panel_g_LOI_Deltaq_order_disorder_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    tem_images, fft_images = [], []

    for i, roi in enumerate(ROI_NAMES, start=1):
        tem_path, fft_path = find_image_pair(input_dir, i)
        tem_img = conservative_crop_from_exported_panel(read_gray(tem_path))
        fft_img = conservative_crop_from_exported_panel(read_gray(fft_path))

        Image.fromarray((np.clip(tem_img, 0, 1) * 255).astype(np.uint8)).save(output_dir / f"{roi}_TEM_crop.png")
        Image.fromarray((np.clip(fft_img, 0, 1) * 255).astype(np.uint8)).save(output_dir / f"{roi}_FFT_crop.png")

        p_fft, delta_q_eff, q_peak_px = fft_peak_descriptors_from_fft_image(fft_img)
        a_off, c_loc = roi_real_space_descriptors(tem_img)

        records.append({
            "ROI": roi,
            "P_FFT": p_fft,
            "A_off": a_off,
            "C_loc": c_loc,
            "Delta_q_eff": delta_q_eff,
            "q_peak_px": q_peak_px,
        })
        tem_images.append(tem_img)
        fft_images.append(fft_img)

    descriptors = calculate_loi(pd.DataFrame(records))
    descriptors.to_csv(output_dir / "mxene_classical_descriptors.csv", index=False)
    QUANTUM_REFERENCE.to_csv(output_dir / "mxene_quantum_reference.csv", index=False)
    ensemble_summary(descriptors, QUANTUM_REFERENCE).to_csv(output_dir / "mxene_ensemble_summary.csv", index=False)
    make_plots(descriptors, QUANTUM_REFERENCE, tem_images, fft_images, output_dir)

    print("Classical LOI analysis finished.")
    print(f"Output directory: {output_dir}")
    print(descriptors[["ROI", "P_FFT", "A_off", "C_loc", "Delta_q_eff", "LOI", "w_LOI"]].round(4).to_string(index=False))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical LOI analysis for five MXene TEM/FFT ROIs.")
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory containing ROI/FFT images.")
    parser.add_argument("--output-dir", type=Path, default=Path("mxene_loi_output"), help="Output directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.input_dir, args.output_dir)
