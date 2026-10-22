| Condition | Controller | Mean wait (min) ↓ | P95 wait (min) ↓ | Completed (%) ↑ | Abandoned (%) ↓ | Unfinished (%) ↓ |
|---|---|---|---|---|---|---|
| Normal Demand | Fixed | 7.42 | 14.88 | 100.00 | 0.00 | 0.00 |
| Normal Demand | RL (PPO) | 5.63 | 11.69 | 100.00 | 0.00 | 0.00 |
| Normal Demand | Proportional | 7.07 | 14.73 | 99.64 | 0.00 | 0.36 |
| Normal Demand | Threshold | 7.27 | 15.46 | 99.58 | 0.00 | 0.42 |
| Demand Surge | Fixed | 10.49 | 39.13 | 95.35 | 4.65 | 0.00 |
| Demand Surge | RL (PPO) | 8.12 | 30.77 | 98.21 | 1.79 | 0.00 |
| Demand Surge | Proportional | 9.08 | 28.51 | 97.56 | 1.82 | 0.62 |
| Demand Surge | Threshold | 9.11 | 26.64 | 98.59 | 0.90 | 0.52 |
| Traffic Disruption | Fixed | 7.57 | 15.20 | 100.00 | 0.00 | 0.00 |
| Traffic Disruption | RL (PPO) | 5.75 | 12.11 | 100.00 | 0.00 | 0.00 |
| Traffic Disruption | Proportional | 7.18 | 15.04 | 99.55 | 0.00 | 0.45 |
| Traffic Disruption | Threshold | 7.34 | 15.82 | 99.63 | 0.00 | 0.37 |
