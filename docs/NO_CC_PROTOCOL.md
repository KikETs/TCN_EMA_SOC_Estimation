# Strict NoCC Protocol

## Forbidden Inputs

- SOC input
- SOC_CC input
- window-start SOC or initial SOC label input
- cumulative Ah or charge-throughput features
- trajectory progress or absolute time
- target or label columns
- explicit current-integration SOC state update

## Allowed Inputs

- corrected voltage proxy `V_corr_raw`
- instantaneous current `I_raw`
- ambient temperature `T`
- causal finite-memory EMA features derived from voltage/current streams
- local derivative/excitation features for ablation variants

## Important Wording

Use:

> Current is used only as instantaneous and finite-memory causal excitation, not integrated into a SOC state.

Do not use:

> Current is unused.

Do not claim that NoCC proves current integration unnecessary. The package only studies a strict NoCC ablation/candidate.

## Audit Scripts

- `scripts/check_no_cc_audit.py`
- `scripts/check_ema_causality.py`
- `tests/test_no_cc_feature_schema.py`
- `tests/test_ema_causality.py`

