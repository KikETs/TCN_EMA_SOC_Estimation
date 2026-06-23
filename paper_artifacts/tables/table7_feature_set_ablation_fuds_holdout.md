| Feature set | Input role | Input dim. | 0 °C MAE | 25 °C MAE | 45 °C MAE | Temp-mean MAE | Worst-temp MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | Corrected voltage + current + temperature | 3 | 1.0592 | 1.8423 | 0.2766 | 1.0594 | 1.8423 |
| G1 | G0 + local derivatives/excitation | 8 | 0.7927 | 1.7248 | 0.2192 | 0.9122 | 1.7248 |
| G4 | G0 + voltage/current/absolute-current EMA memory | 17 | 0.4190 | 0.4648 | 0.3603 | 0.4147 | 0.4648 |
| G6 | G4 + derivative/excitation terms | 23 | 0.4542 | 0.6989 | 0.3710 | 0.5081 | 0.6989 |
| G7 | G6 without current/absolute-current EMA | 15 | 0.4632 | 2.0438 | 0.3359 | 0.9476 | 2.0438 |
| G8 | G6 without voltage EMA | 17 | 0.6104 | 0.6100 | 0.1756 | 0.4653 | 0.6104 |
