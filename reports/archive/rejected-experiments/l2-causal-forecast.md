# L2.1 — Causal arrival forecast (rejected)

**Status:** `REJECTED` after three-seed confirmation. Control retained.
**Full narrative:** [`../l0-l3-full-report.md`](../l0-l3-full-report.md)
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
