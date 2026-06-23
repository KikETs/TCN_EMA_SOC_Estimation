from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "feature_ablation_summary.csv"))
    df = src.rename(
        columns={
            "feature_set": "Feature set",
            "input_role": "Input role",
            "input_dim": "Input dim.",
            "mae_0C": "0 ℃ MAE",
            "mae_25C": "25 ℃ MAE",
            "mae_45C": "45 ℃ MAE",
            "temp_mean_mae": "Mean MAE",
            "worst_temp_mae": "Worst MAE",
        }
    )
    df.loc[df["Feature set"].eq("G4"), "Feature set"] = "G4 (proposed)"
    save_table(df, "table_8_feature_set_ablation", digits=4)


if __name__ == "__main__":
    main()
