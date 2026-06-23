from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "source_tables" / "vi_support_coverage.csv"))
    src = src[src["comparison_kind"].eq("main_split")].copy()
    src["test_samples_outside_train_occupied_bins_fraction"] *= 100.0
    df = src[
        [
            "temperature_C",
            "train_profiles",
            "test_profile",
            "overlap_coefficient",
            "jensen_shannon_divergence_bits",
            "occupied_bin_overlap_fraction_intersection_over_union",
            "test_samples_outside_train_occupied_bins_fraction",
        ]
    ].rename(
        columns={
            "temperature_C": "Temperature (℃)",
            "train_profiles": "Train profiles",
            "test_profile": "Test profile",
            "overlap_coefficient": "Overlap coeff.",
            "jensen_shannon_divergence_bits": "JSD (bits)",
            "occupied_bin_overlap_fraction_intersection_over_union": "Occupied-bin IoU",
            "test_samples_outside_train_occupied_bins_fraction": "Test outside train bins (%)",
        }
    )
    df["Train profiles"] = df["Train profiles"].astype(str).str.replace("+", " ", regex=False)
    save_table(df, "table_S2_vi_support_coverage", digits=4)


if __name__ == "__main__":
    main()
