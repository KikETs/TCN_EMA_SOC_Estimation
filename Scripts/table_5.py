from __future__ import annotations

import pandas as pd

from common import save_table


def main() -> None:
    rows = [
        {
            "Group": "Base measurement-derived channels",
            "Channels": "V_corr_raw, I_raw, T",
            "Memory scales": "-",
        },
        {
            "Group": "Voltage EMA memory",
            "Channels": "V_corr_raw_ema50, V_corr_raw_dev_ema50; V_corr_raw_ema200, V_corr_raw_dev_ema200; V_corr_raw_ema800, V_corr_raw_dev_ema800",
            "Memory scales": "50, 200, 800",
        },
        {
            "Group": "Current EMA memory",
            "Channels": "I_raw_ema50, I_raw_dev_ema50; I_raw_ema200, I_raw_dev_ema200",
            "Memory scales": "50, 200",
        },
        {
            "Group": "Absolute-current EMA memory",
            "Channels": "absI_ema50, absI_dev_ema50; absI_ema200, absI_dev_ema200",
            "Memory scales": "50, 200",
        },
    ]
    save_table(pd.DataFrame(rows), "table_5_causal_ema_measurement_representation", digits=4)


if __name__ == "__main__":
    main()
