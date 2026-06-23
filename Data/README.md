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
profile, and an `80` or `80SOC` manuscript-dataset token. This token is used
only to validate that the correct dynamic-profile files are being converted; it
is not used as the SOC label.

Reference files belong in `Data/raw_reference/`:

```text
Data/raw_reference/
  *Initial*capacity*.xls*
  *lowcurrentOCV*.xls*
  *low current OCV*.xlsx
```

Low-current OCV files are required for the manuscript SOC-label calculation
unless the compact SOC0/Qref provenance tables are present in
`Data/source_tables/`. They are not a substitute for the dynamic profile files
above.

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

SOC label rule:
- the dynamic driving step is extracted from the raw profile file
- `SOC0_Vinit(V)` is taken from the final voltage of the preceding rest step
- `SOC0_OCV_inferred` is obtained from the same-temperature low-current OCV curve
- `Q_ref_lc_ocv_Ah` is the same-temperature low-current OCV discharge capacity
- `Qnet_removed(Ah)` is computed within the driving step as
  `Qdis_seg(Ah) - Qchg_seg(Ah)`
- `SOC_CC_unclipped = SOC0_OCV_inferred - Qnet_removed(Ah) / Q_ref_lc_ocv_Ah`
- `SOC_CC = clip(SOC_CC_unclipped, 0, 1)`

The compact source tables below are included to reproduce the manuscript SOC
anchors exactly:

```text
Data/source_tables/ocv_inferred_start_soc_by_file.csv
Data/source_tables/lc_ocv_capacity_reference.csv
```

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
