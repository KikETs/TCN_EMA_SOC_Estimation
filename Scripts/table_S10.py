from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "table_9_spectral_energy_distribution.csv"))
    df = src[
        [
            "feature_group",
            "representative_channel",
            "low_frequency_energy_percent",
            "mid_frequency_energy_percent",
            "high_frequency_energy_percent",
            "median_frequency",
        ]
    ].rename(
        columns={
            "feature_group": "Feature group",
            "representative_channel": "Representative channel",
            "low_frequency_energy_percent": "Low-frequency energy (%)",
            "mid_frequency_energy_percent": "Mid-frequency energy (%)",
            "high_frequency_energy_percent": "High-frequency energy (%)",
            "median_frequency": "Median frequency",
        }
    )
    save_table(df, "table_S10_spectral_energy_by_record", digits=4)


if __name__ == "__main__":
    main()
