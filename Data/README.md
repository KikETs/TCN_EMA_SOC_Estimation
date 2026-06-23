# Data Preparation

Place dynamic CALCE/NMC profile Excel files in `Data/raw_dynamic/`.

Do not place only low-current OCV files in `Data/raw_dynamic/`. The model
training dataset requires dynamic driving-profile records.

Raw CALCE files are not redistributed in this repository. The source archive URLs and local reference-file checksums are listed in:

```text
Data/source_data_manifest.csv
```

Expected filename metadata:
- profile token: `BJDST`, `DST`, `US06`, or `FUDS`
- temperature token: `0C`, `25C`, or `45C`
- required start-SOC token after the profile, e.g. `0C_DST_80.xls`
  or `0C_DST_80SOC.xls`

Required dynamic files for the manuscript dataset:

```text
Data/raw_dynamic/
  *0C_BJDST_80*.xls*
  *0C_DST_80*.xls*
  *0C_US06_80*.xls*
  *0C_FUDS_80*.xls*
  *25C_BJDST_80*.xls*
  *25C_DST_80*.xls*
  *25C_US06_80*.xls*
  *25C_FUDS_80*.xls*
  *45C_BJDST_80*.xls*
  *45C_DST_80*.xls*
  *45C_US06_80*.xls*
  *45C_FUDS_80*.xls*
```

The leading date/cell prefix may differ, for example:

```text
02_24_2016_SP20-2_0C_DST_80SOC.xls
```

The important part is that each dynamic file name contains temperature,
profile, and an `80` or `80SOC` start-SOC token. Other start-SOC tokens are
rejected by the preprocessing script.

Reference files belong in `Data/raw_reference/`:

```text
Data/raw_reference/
  *Initial*capacity*.xls*
  *lowcurrentOCV*.xls*
  *low current OCV*.xlsx
```

The initial-capacity file is used to infer capacity when `--capacity-ah` is not
provided. Low-current OCV files are reference/provenance files and are not a
substitute for the dynamic profile files above.

Run:

```bash
python Data/prepare_calce_nmc.py
```

By default, the script validates that all 12 manuscript dynamic-profile files
are present. For a single-file or partial conversion check, pass:

```bash
python Data/prepare_calce_nmc.py --allow-incomplete
```

The script writes profile-temperature CSV files to `Data/processed/`.

SOC label rule used by default:
- initial SOC is inferred from the `80` or `80SOC` filename token when available
- otherwise initial SOC defaults to 80 % only for files without any SOC token
- capacity is inferred from `Data/raw_reference/*Initial*capacity*.xls*` when possible
- otherwise capacity defaults to 2.0 Ah
- SOC uses net removed capacity, `Discharge_Capacity(Ah) - Charge_Capacity(Ah)`,
  when both capacity columns are available
- otherwise SOC is estimated from net current integration
- SOC is clipped to `[0, 100]`

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
