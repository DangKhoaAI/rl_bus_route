# Forecast diagnostic (Task 10)

Historical forecaster: time-bin mean of next-15-minute arrivals on **train-day
arrival logs**, times a recent-10-minute correction
`clip((recent+1)/(historical+1), 0.5, 2.0)`. Fit once on the 500 train days,
then frozen. Units below are observation-scaled counts (passengers / 40) per
(route, direction, stop) cell.

| Split | MAE | Bias | Bins |
|---|---:|---:|---:|
| validation (100 days) | 0.01906 | +0.000027 | 1,600 |
| test_id (200 days) | 0.01913 | +0.000173 | 3,200 |

Raw-passenger MAE is about **0.76 passengers / 15-minute cell**. Causal tests
(`tests/test_forecast.py`) check that two tapes identical up to *t* and
different after *t* produce the same forecast at *t*, and that `predict`
rejects future tapes and scenario seeds.

Policy comparison (forecast vs no-forecast core, 3 seeds, same budget) is
trained with `configs/experiments/forecast.toml`. Checkpoints and paired
control results are filled after those runs finish.
