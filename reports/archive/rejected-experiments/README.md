# Rejected experiments

Each card records one pre-registered change, the exact config diff, the result,
and why it was not adopted. None of these changed the final controller.

| Experiment | Card | Result | Status |
|---|---|---|---|
| L1: GAE `lambda` 0.98 / 1.0 | [`l1-gae-lambda.md`](l1-gae-lambda.md) | +0.54% / +1.78% validation cost at screening | `REJECTED` |
| L2.1: causal 15-min arrival forecast | [`l2-causal-forecast.md`](l2-causal-forecast.md) | +2.03% cost, CI above 0, 1/3 seeds lower | `REJECTED` |
| L2.3: potential-based reward shaping | [`l2-reward-shaping.md`](l2-reward-shaping.md) | +3.59% cost, CI above 0, 0/3 seeds lower | `REJECTED` |

## Tables

| File | Experiment |
|---|---|
| [`l1-screening.csv`](l1-screening.csv) | L1 seed-11 screening |
| [`forecast-summary.csv`](forecast-summary.csv) | L2.1 per-seed validation |
| [`forecast-contrast.json`](forecast-contrast.json) | L2.1 paired contrast and decision |
| [`forecast-error.json`](forecast-error.json) | L2.1 forecast MAE/bias |
| [`forecast-verification.json`](forecast-verification.json) | L2.1 structural audit |
| [`reward-shaping-summary.csv`](reward-shaping-summary.csv) | L2.3 per-seed validation |
| [`reward-shaping-contrast.json`](reward-shaping-contrast.json) | L2.3 paired contrast and decision |
| [`reward-shaping-scale-probe.json`](reward-shaping-scale-probe.json) | L2.3 train-only scale probe |
| [`reward-shaping-verification.json`](reward-shaping-verification.json) | L2.3 structural audit |

## Reading rule

`REJECTED` means “tested under the registered rule and not adopted”, not “the run
failed”. The cards include the registered threshold and the failed rule.

The full change ledger is
[`../../provenance/experiments-ledger.md`](../../provenance/experiments-ledger.md).
The final controller is described in
[`../../FINAL_REPORT.md`](../../FINAL_REPORT.md).
