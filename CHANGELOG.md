# Changelog

## v1.1-classical-LOI

Added a classical TEM/FFT local-ordering-index (LOI) control analysis for the five MXene ROIs.

### Added
- `scripts/05_classical_loi_analysis.py`
- `data/processed/classical_loi/mxene_classical_descriptors.csv`
- `data/processed/classical_loi/mxene_quantum_reference.csv`
- `data/processed/classical_loi/mxene_ensemble_summary.csv`
- `data/figures/classical_loi/` with the additional LOI figure panels

### Purpose
The added classical workflow provides an image-derived control for the microscopy-conditioned quantum hierarchy. It quantifies local order using FFT peak contrast, autocorrelation, local coherence, and FFT peak width, then compares the resulting LOI values and LOI-derived patch weights with the motif-resolved quantum energy landscape.
