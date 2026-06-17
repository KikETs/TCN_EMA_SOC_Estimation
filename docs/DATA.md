# Data

Raw CALCE/NMC data are not included.

Place local raw CSV files under:

```text
data/raw/NMC_SAMSUNG_INR_18650_2Ah/
```

The expected frozen G4 protocol uses:

- train profiles: `DST`, `US06`, `BJDST`
- test profile: `FUDS`
- temperatures: `0C`, `25C`, `45C`

Run:

```bash
python scripts/check_data_presence.py
python scripts/prepare_calce_nmc_data.py
python scripts/build_g4_features.py
```

The generated data cache is local only and ignored by Git.

## Label Policy

SOC labels are supervised targets and evaluation targets only. They must not be used as model inputs, gate inputs, threshold selectors, or correction inputs.

## R0 and Vcorr Provenance

`V_corr_raw` is derived from voltage/current streams using train-profile R0 estimates by temperature. The release documents this dependency because `V_corr_raw` is not a raw sensor column.

## Current EMA Caveat

Current-derived EMA channels are allowed only as finite-memory causal excitation history. They are not cumulative Ah features and are not used in an explicit charge-conservation SOC state update.

