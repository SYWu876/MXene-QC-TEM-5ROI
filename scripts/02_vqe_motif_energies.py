"""Run exact diagonalization and simple/enhanced VQE for all five motifs."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mxene_qc_tem.quantum_model import build_hamiltonian, exact_ground_energy, run_vqe_simple, run_vqe_enhanced

params = pd.read_csv(ROOT / "data" / "processed" / "roi_parameters.csv")
rows = []
for i, row in params.iterrows():
    H = build_hamiltonian(row.t_parallel_Ha, row.t_perp_Ha, row.Delta_Ha)
    E_exact = exact_ground_energy(H)
    vqe_simple = run_vqe_simple(H)
    vqe_enh = run_vqe_enhanced(H, seed=i)
    rows.append({
        "roi": row.roi,
        "E_exact_Ha": E_exact,
        "E_simple_VQE_Ha": vqe_simple["E_vqe"],
        "DeltaE_simple_Ha": vqe_simple["E_vqe"] - E_exact,
        "E_enhanced_VQE_Ha": vqe_enh["E_vqe"],
        "DeltaE_enhanced_Ha": vqe_enh["E_vqe"] - E_exact,
    })
out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "processed" / "vqe_results_recomputed.csv", index=False)
print(out.to_string(index=False))
