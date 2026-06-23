from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "table_9_spectral_energy_distribution.csv"))
    df = src.rename(
        columns={
            "feature_group": "Feature group",
            "representative_channel": "Representative channel",
            "low_frequency_energy_percent": "Low-frequency energy (%)",
            "mid_frequency_energy_percent": "Mid-frequency energy (%)",
            "high_frequency_energy_percent": "High-frequency energy (%)",
            "median_frequency": "Median frequency (cycles/sample)",
            "high_frequency_reduction_vs_raw_reference_percent": "High-frequency reduction vs raw reference (%)",
        }
    )
    save_table(df, "table_9_spectral_energy_distribution", digits=4)


if __name__ == "__main__":
    main()
