# Documentation folder

This folder is reserved for documentation after publication.

The submitted manuscript, figures, and Supporting Information files are not included in this repository during peer review. Reproducibility is provided through the raw TEM image, processed CSV tables, Python source code, and executable scripts.


## Added classical LOI analysis

The updated package includes a classical TEM/FFT local-ordering-index analysis as an independent microscopy-only control for the five MXene ROIs. This addition is intended to support the comparison between:

1. image-derived structural ordering, represented by LOI and LOI-derived patch weights; and
2. motif-resolved quantum energetics, represented by the two-qubit Hamiltonian and exact/VQE ground-state energies.

The relevant files are:

- `scripts/05_classical_loi_analysis.py`
- `data/processed/classical_loi/mxene_classical_descriptors.csv`
- `data/processed/classical_loi/mxene_quantum_reference.csv`
- `data/processed/classical_loi/mxene_ensemble_summary.csv`
- `data/figures/classical_loi/`
