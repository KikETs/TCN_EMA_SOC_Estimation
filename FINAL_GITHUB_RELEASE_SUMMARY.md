# Final GitHub Release Summary

## What New Folder Was Created?

`g4_ema_soc_release/` was created as a standalone release package.

## What Code Was Copied?

Twenty-four Python source files were copied into `soc_decomp/`, covering the frozen G4 training path and local dependency closure. See `source_copy_manifest.csv` for SHA-256 hashes and unchanged-copy status.

## What Data Is Excluded?

Excluded:

- raw CALCE/NMC data
- processed trajectory data
- feature caches
- checkpoints
- full prediction dumps
- archive files

## How Does a User Place Data?

Place raw CSV files under:

```text
data/raw/NMC_SAMSUNG_INR_18650_2Ah/
```

## Which Script Processes Raw Data?

`scripts/prepare_calce_nmc_data.py` records the local raw-file manifest after data placement.

## Which Script Builds G4 Features?

`scripts/build_g4_features.py` builds the local G4 feature cache and writes the NoCC audit/schema files.

## Which Script Trains/Evaluates G4?

`scripts/train_g4.py` runs the frozen G4 protocol through the copied model code.

`scripts/evaluate_g4.py` and `scripts/recompute_metrics.py` recompute metrics from local prediction CSV files.

## Which Script Builds Paper Artifacts?

`scripts/build_paper_artifacts.py` regenerates tables and figures from `paper_artifacts/source_metrics/`.

## Which Files Are Safe To Commit?

Safe to commit:

- `README.md`
- `docs/`
- `configs/`
- `scripts/`
- `soc_decomp/`
- `tests/`
- `source_copy_manifest.csv`
- `SOURCE_OF_TRUTH_MANIFEST.csv`
- `RELEASE_CHECK_REPORT.md`
- `requirements.txt`
- `pyproject.toml`
- `CITATION.cff`
- `.gitignore`

For the current upload, `paper_artifacts/` is intentionally excluded. It can be uploaded later by removing or overriding the ignore rule.

## Which Files Are Intentionally Ignored?

Ignored:

- `data/raw/`
- `data/processed/`
- `data/cache/`
- `results/checkpoints/`
- `results/predictions/`
- large ML artifact extensions such as `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, `.zip`, and archive files
- `paper_artifacts/`
- Python cache folders

## Which Tests/Checks Pass?

Passed:

- `pytest -p no:cacheprovider tests`: 5 tests passed
- `python scripts/release_check.py --base-dir .`: all release checks passed
- `python scripts/build_paper_artifacts.py --base-dir .`: generated 7 table bundles and 8 figure pairs locally before `paper_artifacts/` exclusion

## Manual Steps Before Git Push

- choose and add a real license
- confirm CALCE/NMC data redistribution constraints
- update `CITATION.cff` with final author/manuscript metadata
- inspect generated paper tables/figures
- initialize Git and commit only after reviewing `git status`
