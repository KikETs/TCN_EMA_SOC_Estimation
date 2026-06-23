from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "feature_ablation_by_seed.csv"))
    src = src[src["metric_name"].eq("feature_ablation_mae_pct")].copy()
    src["temperature"] = pd.to_numeric(src["temperature"], errors="coerce")
    pivot = src.pivot_table(
        index=["ablation_group", "input_dim", "seed"],
        columns="temperature",
        values="metric_value",
        aggfunc="mean",
    ).reset_index()
    order = {name: i for i, name in enumerate(["G0", "G1", "G4", "G6", "G7", "G8"])}
    pivot["_order"] = pivot["ablation_group"].map(order)
    pivot = pivot.sort_values(["_order", "seed"])
    df = pd.DataFrame(
        {
            "Feature set": pivot["ablation_group"],
            "Dim.": pivot["input_dim"],
            "Seed": pivot["seed"],
            "0 ℃ MAE": pivot[0.0],
            "25 ℃ MAE": pivot[25.0],
            "45 ℃ MAE": pivot[45.0],
        }
    )
    save_table(df, "table_S8_feature_ablation_by_seed", digits=4)


if __name__ == "__main__":
    main()
