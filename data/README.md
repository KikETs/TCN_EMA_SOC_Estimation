# Data Placement

Raw CALCE/NMC files are not included in this release.

Place the raw files under:

```text
data/raw/NMC_SAMSUNG_INR_18650_2Ah/
```

The repository ignores `data/raw/`, `data/processed/`, and `data/cache/` so local data are not committed accidentally.

Expected drive profiles for the frozen G4 protocol:

- train: `DST`, `US06`, `BJDST`
- test: `FUDS`
- temperatures: `0C`, `25C`, `45C`

