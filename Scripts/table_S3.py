from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "source_tables" / "ambiguity_stratification_summary.csv"))
    labels = {
        "A_raw_VI_bins": "Raw",
        "B_raw_plus_absI_mean_past200_tertile": "+ |I| history",
        "C_raw_plus_V_dev_from_past200_tertile": "+ V dev.",
        "D_raw_plus_both_history_tertiles": "+ both",
    }
    src = src.copy()
    src["condition"] = src["condition"].map(labels).fillna(src["condition"])
    src["median_SOC_IQR_fraction"] *= 100.0
    src["p90_SOC_IQR_fraction"] *= 100.0
    src["fraction_samples_in_ambiguous_bins_IQR_ge_0p10"] *= 100.0
    df = src.rename(
        columns={
            "temperature_C": "Temp. (C)",
            "condition": "Condition",
            "median_SOC_IQR_fraction": "Median SOC IQR (%SOC)",
            "p90_SOC_IQR_fraction": "P90 SOC IQR (%SOC)",
            "fraction_samples_in_ambiguous_bins_IQR_ge_0p10": "Ambiguous samples (%)",
            "n_valid_bins": "Valid bins",
        }
    )
    save_table(df, "table_S3_history_conditioned_ambiguity", digits=4)


if __name__ == "__main__":
    main()
