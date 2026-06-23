# Minimal Manuscript Data Package

This repository intentionally excludes raw CALCE data archives and model checkpoints. The files in `data/`, `results/`, `audits/`, and `configs/` provide the minimum manuscript/SI source metrics needed to recompute the reported tables and figures.

Raw data provenance is recorded in `data/source_data_manifest.csv` with CALCE archive names and SHA256 checksums. Rebuild scripts use `configs/paths.example.yaml` to point to local raw files.

Large prediction-row files and leave-one-profile-out scratch outputs are not part of this minimal package unless they are explicitly needed for a manuscript table or figure.
