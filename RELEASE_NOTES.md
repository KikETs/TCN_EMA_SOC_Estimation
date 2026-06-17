# Release Notes

This folder is a GitHub-ready reproducibility package for the frozen G4 EMA SOC-estimation candidate.

Copied source code:

- `soc_decomp/__init__.py`
- G4 training/evaluation modules and their local dependencies
- source files are copied unchanged from the working repository unless listed in `source_copy_manifest.csv`

Local-only small source metrics:

- seed reproduction summaries
- feature ablation summaries
- EMA group and perturbation diagnostics
- profile rotation summaries
- epoch sweep diagnostics
- model-class baselines
- forbidden-reference diagnostics

Excluded by design from the GitHub upload:

- raw data
- processed trajectory caches
- checkpoints
- large logs
- prediction dumps
- `paper_artifacts/`

Manual steps before publishing:

- choose a license
- verify CALCE/NMC data usage terms
- update citation metadata
- run `python scripts/release_check.py`
