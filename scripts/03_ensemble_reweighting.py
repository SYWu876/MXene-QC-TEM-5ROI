"""Compute beta-dependent motif probabilities and ensemble averages."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mxene_qc_tem.ensemble import boltzmann_probabilities, ensemble_average

params = pd.read_csv(ROOT / "data" / "processed" / "roi_parameters.csv")
E0 = params["E0_exact_Ha"].to_numpy()
d = params["d_parallel_nm"].to_numpy()
tp = params["t_parallel_Ha"].to_numpy()
Delta = params["Delta_Ha"].to_numpy()

rows = []
for beta in [0, 1, 3]:
    P = boltzmann_probabilities(E0, beta)
    rows.append({
        "beta_Ha_inv": beta,
        "avg_d_parallel_nm": ensemble_average(d, P),
        "avg_t_parallel_Ha": ensemble_average(tp, P),
        "avg_Delta_Ha": ensemble_average(Delta, P),
        "avg_E0_Ha": ensemble_average(E0, P),
        **{f"P_{params.loc[i,'roi']}": P[i] for i in range(len(P))}
    })
out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "processed" / "ensemble_probabilities_recomputed.csv", index=False)
print(out.to_string(index=False))
