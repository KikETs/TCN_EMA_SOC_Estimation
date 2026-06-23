from __future__ import annotations

import pandas as pd

from common import DATA, require, save_table


def main() -> None:
    src = pd.read_csv(require(DATA / "model_tables" / "regional_thresholds.csv"))
    df = src[["region_definition", "threshold_rule"]].rename(
        columns={
            "region_definition": "Region definition",
            "threshold_rule": "Grouping rule",
        }
    )
    save_table(df, "table_S9_regional_grouping_criteria", digits=4)


if __name__ == "__main__":
    main()
