
"""Patch-ensemble weighting utilities."""

from __future__ import annotations
import numpy as np

def boltzmann_probabilities(E0: np.ndarray, beta: float, weights: np.ndarray | None = None) -> np.ndarray:
    E0 = np.asarray(E0, dtype=float)
    if weights is None:
        weights = np.ones_like(E0) / len(E0)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    raw = weights * np.exp(-beta * E0)
    return raw / raw.sum()

def ensemble_average(values: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=float) * np.asarray(probabilities, dtype=float)))

def beta_curves(E0: np.ndarray, d_parallel: np.ndarray, beta_grid: np.ndarray) -> dict:
    P = np.vstack([boltzmann_probabilities(E0, b) for b in beta_grid])
    E_avg = P @ np.asarray(E0, dtype=float)
    d_avg = P @ np.asarray(d_parallel, dtype=float)
    return {"beta": beta_grid, "P": P, "E_avg": E_avg, "d_avg": d_avg}
