# CEMA-TCN SOC Estimation Package

This folder contains the files needed to reproduce the CEMA-TCN manuscript figures/tables and run the model family used in the paper.

## Data Availability

The code, configuration files, audit files, feature schemas, and processed summary results are available at [https://github.com/KikETs/TCN_EMA_SOC_Estimation](https://github.com/KikETs/TCN_EMA_SOC_Estimation). Raw CALCE battery records are not redistributed and should be obtained from the original CALCE data source. Large intermediate files, checkpoints, and full prediction dumps are not included.

## Folder layout

```text
CEMA-TCN/
  Data/
    source_data_manifest.csv
    raw_dynamic/        # put CALCE dynamic profile Excel files here
    raw_reference/      # OCV / initial-capacity reference files
    processed/          # generated model-ready CSV files
    section2_tables/    # source tables for Section 2 figures/tables
    source_tables/      # compact measurement-structure source tables
    source_metrics/     # compact result CSVs for tables/figures
    model_tables/       # model-performance and ablation source tables
    frequency_tables/   # source tables for frequency-domain figures
    figures_source/     # compact trace source data
    manuscript_source_map.csv
  Scripts/
    figure_1.py ... figure_10.py
    figure_S1.py ... figure_S3.py
    table_1.py ... table_10.py
    table_S1.py ... table_S11.py
  Figures/
    make_figures.ipynb  # notebook that calls figure scripts
  Tables/
    make_tables.ipynb   # notebook that calls table scripts
  Models/
    model_zoo.py        # TCN/LSTM/GRU/Transformer/MLP definitions
    feature_sets.py     # G0/G1/G4/G6/G7/G8 input feature definitions
    train.py            # minimal training entrypoint
```

## Data conversion

Place dynamic profile Excel files in `Data/raw_dynamic/`, then run:

```bash
python Data/prepare_calce_nmc.py
```

The output CSV files are written to `Data/processed/`.

Raw CALCE files and generated driving-profile CSV files are intentionally excluded from Git. The expected CALCE source archives and reference-file checksums are listed in:

```text
Data/source_data_manifest.csv
```

## Figure/table generation

Run the notebooks:

```bash
jupyter notebook Figures/make_figures.ipynb
jupyter notebook Tables/make_tables.ipynb
```

Or run scripts directly, for example:

```bash
python Scripts/figure_6.py
python Scripts/table_6.py
```

The main-text/SI mapping is recorded in:

```text
Data/manuscript_source_map.csv
```

## Model training

Example:

```bash
python Models/train.py --model cema_tcn --feature-set G4 --test-profile FUDS
```

The proposed TCN uses one causal-convolution sublayer per residual block. The older two-sublayer variant is intentionally excluded.
