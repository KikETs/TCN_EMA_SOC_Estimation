# Data Preparation

Place dynamic CALCE/NMC profile Excel files in `Data/raw_dynamic/`.

Raw CALCE files are not redistributed in this repository. The source archive URLs and local reference-file checksums are listed in:

```text
Data/source_data_manifest.csv
```

Expected filename metadata:
- profile token: `BJDST`, `DST`, `US06`, or `FUDS`
- temperature token: `0C`, `25C`, or `45C`

Run:

```bash
python Data/prepare_calce_nmc.py
```

The script writes profile-temperature CSV files to `Data/processed/`.

SOC label rule used by default:
- initial SOC = 80 %
- capacity is inferred from `Data/raw_reference/*Initial*capacity*.xls*` when possible
- otherwise capacity defaults to 2.0 Ah
- SOC decreases with discharged Ah and is clipped to `[0, 100]`

Feature construction:
- `V_corr_raw = causal_time_ema(V_raw - I_raw * R0(T), tau=120 s)`
- voltage EMA spans: 50, 200, 800 samples
- current and absolute-current EMA spans: 50, 200 samples
- no SOC, cumulative Ah, or trajectory progress is included as model input

## Manuscript artifact sources

The manuscript/SI figure-table mapping is recorded in:

```text
Data/manuscript_source_map.csv
```

Committed source groups:

- `processed/`: generated locally from raw CALCE files and intentionally excluded from Git.
- `section2_tables/`: Section 2 measurement-structure tables.
- `source_tables/`: compact measurement statistics and support/ambiguity tables.
- `frequency_tables/`: frequency-domain source summaries.
- `model_tables/`: model comparison, ablation, spectral, and regional-error source tables.
- `figures_source/`: compact representative traces.
- `predictions/main_fuds/`: full prediction dumps are intentionally excluded from Git.
- `manuscript_tables/table_6_main_fuds_performance_locked.csv`: Table 6 values visible in the provided DOCX. The exact upstream RMSE/MaxAE CSV matching those values was not found in the local source tree, so this locked table is kept as the manuscript source for Table 6.
