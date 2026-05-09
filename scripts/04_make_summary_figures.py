"""Regenerate summary figures from processed data."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mxene_qc_tem.ensemble import beta_curves

OUT = ROOT / "data" / "figures"
OUT.mkdir(exist_ok=True, parents=True)
params = pd.read_csv(ROOT / "data" / "processed" / "roi_parameters.csv")
labels = params["roi"].tolist()
x = np.arange(len(labels))

def style(ax):
    for s in ax.spines.values():
        s.set_visible(True)
    ax.tick_params(direction="in", top=True, right=True)

fig, ax = plt.subplots(figsize=(3.2, 3.0))
ax.bar(x, params["d_parallel_nm"])
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel(r"$d_{\parallel}$ (nm)")
style(ax); fig.tight_layout()
fig.savefig(OUT / "panel_d_parallel.png", dpi=600, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.2, 3.0))
w = 0.35
ax.bar(x-w/2, params["t_parallel_Ha"], width=w, label=r"$t_{\parallel}$")
ax.bar(x+w/2, params["Delta_Ha"], width=w, label=r"$\Delta$")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Model parameters (Ha)")
ax.legend(frameon=True, fontsize=8)
style(ax); fig.tight_layout()
fig.savefig(OUT / "panel_tparallel_Delta.png", dpi=600, bbox_inches="tight")
plt.close(fig)

beta = np.linspace(0, 4, 201)
curves = beta_curves(params["E0_exact_Ha"].to_numpy(), params["d_parallel_nm"].to_numpy(), beta)

fig, ax = plt.subplots(figsize=(3.2, 3.0))
ax.plot(beta, curves["E_avg"], color="blue")
ax.set_xlabel(r"$\beta$ (Ha$^{-1}$)")
ax.set_ylabel(r"$\langle E_0\rangle_\beta$ (Ha)")
style(ax); fig.tight_layout()
fig.savefig(OUT / "panel_beta_E0.png", dpi=600, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.2, 3.0))
ax.plot(beta, curves["d_avg"], color="green", linestyle="--")
ax.set_xlabel(r"$\beta$ (Ha$^{-1}$)")
ax.set_ylabel(r"$\langle d_{\parallel}\rangle_\beta$ (nm)")
style(ax); fig.tight_layout()
fig.savefig(OUT / "panel_beta_dparallel.png", dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved figures to {OUT}")
