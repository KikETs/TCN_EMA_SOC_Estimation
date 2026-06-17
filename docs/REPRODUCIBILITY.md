# Reproducibility

## Frozen Protocol

- model: `anchor_residual_tcn`
- feature set: `paper_g4_all_ema`
- fixed epoch: `160`
- seeds: `0,1,2`
- train profiles: `DST`, `US06`, `BJDST`
- test profile: `FUDS`
- temperatures: `0C`, `25C`, `45C`
- window length: `50`
- train stride: `3`
- test stride: `1`

## Run Order

```bash
python scripts/check_data_presence.py
python scripts/prepare_calce_nmc_data.py
python scripts/build_g4_features.py
python scripts/train_g4.py --seeds 0,1,2
python scripts/build_paper_artifacts.py
python scripts/release_check.py
```

## Metric Source

The paper tables and figures can be regenerated without raw data from the small summary files in `paper_artifacts/source_metrics/`.

## Frozen Candidate Disclosure

G4 was selected after exploratory seed-0 feature ablation. The package freezes the model and reports seed reproduction, feature ablation, epoch sweep, and profile rotation diagnostics for transparency.

