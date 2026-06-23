from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


KEEP = [
    "CEMA-TCN (proposed G4)",
    "LSTM, h128 l1",
    "Transformer, d128 h8 l2 GELU",
    "GRU, h128 l1",
    "Endpoint/window MLP",
]

DISPLAY_MODEL = {
    "CEMA-TCN (proposed G4)": "CEMA-TCN (proposed)",
    "LSTM, h128 l1": "LSTM",
    "Transformer, d128 h8 l2 GELU": "Transformer",
    "GRU, h128 l1": "GRU",
    "Endpoint/window MLP": "MLP",
}


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "model_ablation_g4_input_transformer2.csv"))
    df = src[src["model"].isin(KEEP)].copy()
    order = {name: i for i, name in enumerate(KEEP)}
    df["_order"] = df["model"].map(order)
    df = df.sort_values("_order")[["model", "0C_MAE", "25C_MAE", "45C_MAE", "Mean_MAE", "Max_MAE"]].rename(
        columns={
            "model": "Model",
            "0C_MAE": "0 ℃ MAE",
            "25C_MAE": "25 ℃ MAE",
            "45C_MAE": "45 ℃ MAE",
            "Mean_MAE": "Mean MAE",
            "Max_MAE": "Worst MAE",
        }
    )
    df["Model"] = df["Model"].map(DISPLAY_MODEL).fillna(df["Model"])
    save_table(df, "table_7_model_comparison", digits=4)


if __name__ == "__main__":
    main()
