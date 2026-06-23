from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "manuscript_tables" / "table_6_main_fuds_performance_locked.csv"))
    df = src.rename(
        columns={
            "test_profile": "Test profile",
            "temperature": "Temperature",
            "MAE_percent": "MAE (%SOC)",
            "MAE_std": "MAE std.",
            "RMSE_percent": "RMSE (%SOC)",
            "MaxAE_percent": "MaxAE (%SOC)",
        }
    )
    df["Temperature"] = df["Temperature"].astype(str).str.replace(" C", " ℃", regex=False)
    save_table(df, "table_6_main_fuds_performance", digits=4)


if __name__ == "__main__":
    main()
