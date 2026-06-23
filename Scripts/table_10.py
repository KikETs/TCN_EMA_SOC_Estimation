from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "table_10_region_error_reduction.csv"))
    df = src[
        ["region_definition", "group", "G0_MAE", "G4_MAE", "delta_MAE_G4_minus_G0", "relative_change", "n_windows"]
    ].rename(
        columns={
            "region_definition": "Region definition",
            "group": "Group",
            "G0_MAE": "G0 MAE",
            "G4_MAE": "G4 MAE",
            "delta_MAE_G4_minus_G0": "ΔMAE",
            "relative_change": "Relative change (%)",
            "n_windows": "Windows",
        }
    )
    save_table(df, "table_10_region_error_reduction", digits=4)


if __name__ == "__main__":
    main()
