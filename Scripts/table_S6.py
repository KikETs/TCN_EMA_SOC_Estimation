from __future__ import annotations

import pandas as pd

from common import save_table


def main() -> None:
    rows = [
        ("Learning/test protocol", "Training records: DST, US06, and BJDST; test record: FUDS at 0, 25, and 45 °C."),
        (
            "Input representation",
            "G4 CEMA input with 17 channels: corrected voltage, current, temperature, and causal EMA/deviation channels from corrected voltage, current, and absolute current.",
        ),
        ("Windowing", "Fixed 50-sample windows with endpoint SOC targets; training stride 3 and test stride 1."),
        ("Estimator", "Causal TCN encoder with 128 hidden channels, six residual blocks, kernel size 5."),
        ("Block operations", "Left causal padding, causal convolution, channel-wise normalization, SiLU activation, and dropout."),
        ("Output head", "Temperature-conditioned anchor-residual head with a linear SOC prediction output."),
        ("Training rule", "AdamW optimization with Huber loss; all reported results use the fixed epoch-160 checkpoint."),
        ("Reproduction seeds", "Three independent runs with seeds 0, 1, and 2."),
    ]
    save_table(pd.DataFrame(rows, columns=["Component", "Final setting"]), "table_S6_training_config", digits=4)


if __name__ == "__main__":
    main()
