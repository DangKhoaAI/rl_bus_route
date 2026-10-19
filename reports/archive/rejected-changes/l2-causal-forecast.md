# L2.1 — Causal arrival forecast (rejected)

**Status:** `REJECTED` after three-seed confirmation. Control retained.
**Final report:** [`../../README.md`](../../README.md)
**Ledger row:** `L2-FORECAST` in [`../../provenance/experiments.csv`](../../provenance/experiments.csv)

## Question

Does a causal 15-minute-ahead arrival forecast let the policy anticipate demand
and lower `total_cost_core` without changing physics, masks or reward?

## Setup

- Single-group diff vs L0 core: `[forecast] enabled = false -> true`.
- Forecaster fit on the 500 train arrival tapes only, frozen before evaluation
  (same `forecaster.npz` SHA-256 for all seeds).
- Budget 245,760 transitions; seeds 11/22/33; same 100 validation days as core.
- Evidence: [`forecast-summary.csv`](forecast-summary.csv),
  [`forecast-contrast.json`](forecast-contrast.json),
  [`forecast-verification.json`](forecast-verification.json),
  [`forecast-error.json`](forecast-error.json).


## Forecast-only baseline (historical)

Before the policy comparison, the forecaster was evaluated on its own against
train-day arrival logs:

| Split | MAE | Bias | Bins |
|---|---:|---:|---:|
| validation (100 days) | 0.01906 | +0.000027 | 1,600 |
| test_id (200 days) | 0.01913 | +0.000173 | 3,200 |

Raw-passenger MAE is about **0.76 passengers / 15-minute cell**. Units are
observation-scaled counts (passengers / 40) per (route, direction, stop) cell.
The forecaster is a time-bin mean of the next 15 minutes times a recent-10-minute
correction `clip((recent+1)/(historical+1), 0.5, 2.0)`, fit once on the 500 train
days and frozen.

## Result

Forecast error was low (validation MAE 0.01906, bias 2.7e-05), but control got
worse:

| Contrast (candidate − core) | Value |
|---|---:|
| Mean cost delta | +268.05 (+2.034%) |
| Paired 100-day bootstrap 95% CI | [240.38, 295.72] |
| Seeds with lower cost | 1/3 |
| Seed-11 P95 wait delta | +1.81 min |
| Seed-11 worst-route wait delta | +1.56 min |

## Decision

The CI is entirely above zero, only one seed improved, and the P95/worst-route
increases exceed the pre-registered +1.0 min limits. Rejected by every registered
quality/service rule. The forecast observation was confirmed non-zero (~0.042), so
this is a policy-quality result, not a wiring failure.

## Limits

The forecast is a simple historical baseline, not a deep model. A good MAE does
not imply better control, and this experiment does not test other forecasters.
