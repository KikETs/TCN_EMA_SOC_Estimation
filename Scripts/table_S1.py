from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "section2_tables" / "table_s2_dataset_terminal_summary.csv"))
    df = src[["temperature_C", "profile", "n_samples", "duration_s"]].copy()
    df["Role"] = df["profile"].eq("FUDS").map({True: "Test", False: "Train"})
    stride = df["Role"].map({"Train": 3, "Test": 1})
    df["Windows"] = ((df["n_samples"] - 50) // stride + 1).clip(lower=0)
    df = df.rename(
        columns={
            "temperature_C": "Temperature (℃)",
            "profile": "Profile",
            "n_samples": "Samples",
            "duration_s": "Duration (s)",
        }
    )[["Temperature (℃)", "Profile", "Role", "Samples", "Duration (s)", "Windows"]]
    df["Temperature (℃)"] = df["Temperature (℃)"].astype(int)
    df["Duration (s)"] = df["Duration (s)"].round(0).astype(int)
    save_table(df, "table_S1_record_window_counts", digits=2)


if __name__ == "__main__":
    main()
