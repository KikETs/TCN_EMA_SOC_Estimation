# Causal EMA Memory Features for Strict NoCC SOC Estimation

## Project Overview

This release folder packages the frozen G4 EMA SOC-estimation candidate and the small summary metrics needed to regenerate paper tables and figures.

The model follows a strict NoCC protocol: SOC, SOC_CC, cumulative Ah, trajectory progress, and explicit current-integration SOC state updates are not model inputs. Current is still used through instantaneous excitation and causal finite-memory EMA history.

This is not a global SOTA claim and not a temperature-extrapolation claim. It is a reproducibility package for a frozen candidate selected after exploratory ablation.

## What This Repository Contains

- copied model/source code under `soc_decomp/`
- frozen G4 configs under `configs/`
- helper scripts under `scripts/`
- documentation under `docs/`
- scripts that can rebuild paper artifacts when local source metrics are placed under `paper_artifacts/source_metrics/`
- tests under `tests/`

## What This Repository Does Not Contain

- no raw CALCE/NMC data
- no large processed data
- no checkpoints
- no full prediction dumps
- no committed `paper_artifacts/` folder

Raw and processed data directories are ignored by Git. `paper_artifacts/` is also intentionally ignored in this GitHub upload; keep paper source metrics and generated figures local unless you explicitly decide to publish them.

## Main Result

Frozen G4, trained on `DST + US06 + BJDST` and evaluated on `FUDS` at `0C/25C/45C`, has a three-seed unweighted temperature-mean MAE of `0.4189 +/- 0.0157%`.

Mean MAE by temperature:

- `0C`: `0.4528%`
- `25C`: `0.4564%`
- `45C`: `0.3474%`

The corresponding source metric files are kept in the local artifact package and are not committed in this GitHub upload.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Place raw CALCE/NMC CSV files under:

```text
data/raw/NMC_SAMSUNG_INR_18650_2Ah/
```

Then run:

```bash
python scripts/check_data_presence.py
python scripts/prepare_calce_nmc_data.py
python scripts/build_g4_features.py
```

## Run Main G4 Experiment

```bash
python scripts/train_g4.py --seeds 0,1,2
```

For a quick single-seed reproduction:

```bash
python scripts/train_g4.py --seeds 0
```

## Recompute Metrics

If prediction CSV files are available locally:

```bash
python scripts/recompute_metrics.py --predictions results/predictions/*.csv
```

## Build Paper Artifacts

```bash
python scripts/build_paper_artifacts.py
```

This regenerates tables and figures from `paper_artifacts/source_metrics/` if those local source metric files are present.

## Build Section 2 Measurement-Structure Analysis

The manuscript Section 2 package analyzes terminal voltage/current/temperature/SOC records only. It does not train models and does not use G4 predictions or ablation outputs.

```bash
python analysis/section2_measurement_structure.py \
  --data-root data \
  --out-dir paper_ema_analysis_package/section2_measurement_structure \
  --profiles DST US06 FUDS BJDST \
  --temperatures 0 25 45 \
  --main-test-profile FUDS
```

Generated tables, figures, metadata, and the markdown report are written under `paper_ema_analysis_package/section2_measurement_structure/`.

## NoCC Protocol

Forbidden as model inputs:

- SOC or SOC_CC
- cumulative Ah or charge-throughput features
- trajectory progress or absolute time
- test-label-based gates or thresholds
- explicit `SOC_next = SOC - I * dt / Q` state updates

Allowed:

- `V_corr_raw`
- instantaneous `I_raw`
- ambient temperature `T`
- causal finite-memory EMA and deviation channels derived from voltage/current streams

## Important Limitations

G4 was selected after exploratory seed-0 ablation, so it should be described as a frozen candidate requiring confirmation rather than a pre-declared blind model. EMA current-history features may correlate with cumulative charge, so the package includes forbidden-reference diagnostics and explicit caveats.

## Citation Placeholder

Update `CITATION.cff` after the manuscript title, authors, venue, and DOI are finalized.

## License Note

No license has been selected yet. See `LICENSE_OR_TODO.md` before publishing.
