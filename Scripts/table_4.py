from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "section2_tables" / "table_2_main_raw_vi_soc_spread_compact.csv"))
    df = src.rename(
        columns={
            "temperature_C": "Temperature (℃)",
            "n_valid_bins": "Valid bins",
            "median_SOC_IQR_percent": "Median SOC IQR (%)",
            "p90_SOC_IQR_percent": "P90 SOC IQR (%)",
            "max_SOC_IQR_percent": "Max SOC IQR (%)",
            "ambiguous_sample_fraction_percent": "Ambiguous sample fraction (%)",
        }
    )
    save_table(df, "table_4_vi_bin_soc_iqr", digits=2)


if __name__ == "__main__":
    main()
