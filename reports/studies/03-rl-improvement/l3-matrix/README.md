# L3 — Frozen T9 held-out matrix

**Status:** `ACCEPTED` (core retained as incumbent). All ablations are worse.
**Full narrative:** [`../README.md`](../README.md)
**Final summary:** [`../../../FINAL.md`](../../../FINAL.md)
**Frozen protocol:** [`../l3_protocol.json`](../l3_protocol.json)

## Question

On held-out days, does the core controller beat its own ablations and the three
programmed baselines, and what does each component contribute?

## Setup

- 4 arms: `core`, `no_reassign`, `no_short`, `fairness_zero`.
- 3 seeds (11/22/33), 245,760 transitions each, identical runtime and checkpoint
  rule.
- Held-out: 200 `test_id` + 200 `test_ood_burst` + 200 `test_ood_traffic` days.
- Paired day bootstrap: 2,000 resamples, seed 6001, 95% CI; `delta = arm − core`.
- Baselines: `fixed`, `threshold`, `proportional` on the same days.
- Evidence: [`../tables/l3_held_out_summary.csv`](../tables/l3_held_out_summary.csv),
  [`../tables/l3_contrasts.csv`](../tables/l3_contrasts.csv),
  [`../tables/l3_paired_days.csv`](../tables/l3_paired_days.csv),
  [`../tables/l3_baselines_summary.csv`](../tables/l3_baselines_summary.csv),
  [`../tables/l3_baseline_contrasts.csv`](../tables/l3_baseline_contrasts.csv),
  [`../tables/l3_verification.json`](../tables/l3_verification.json),
  [`../plots/`](../plots/), [`../evidence/l3_failure_traces.json`](../evidence/l3_failure_traces.json).

## Result — ablations vs core

| Arm | Split | Mean Δ | 95% CI | % | Seeds lower |
|---|---|---:|---|---:|---:|
| no_reassign | test_id | +60.78 | [42.61, 78.30] | +0.47% | 1/3 |
| no_short | test_id | +215.01 | [197.75, 231.18] | +1.65% | 2/3 |
| fairness_zero | test_id | +277.51 | [250.12, 305.42] | +2.12% | 0/3 |
| no_reassign | test_ood_burst | +231.19 | [18.22, 442.52] | +0.92% | 0/3 |
| no_short | test_ood_burst | +194.72 | [14.49, 385.74] | +0.77% | 0/3 |
| fairness_zero | test_ood_burst | +235.56 | [−111.11, 606.99] | +0.93% | 0/3 |
| no_reassign | test_ood_traffic | +44.22 | [25.99, 63.10] | +0.33% | 1/3 |
| no_short | test_ood_traffic | +202.94 | [185.85, 220.66] | +1.51% | 1/3 |
| fairness_zero | test_ood_traffic | +252.55 | [222.62, 283.24] | +1.88% | 0/3 |

Core has the lowest mean held-out cost on every split. No ablation has a CI below
zero. The `fairness_zero` burst CI crosses zero, so that single contrast is
reported as insufficient evidence rather than a positive difference. Every
ablation also violates the registered P95/worst-route service limits on at least
one split; only core satisfies all of them.

## Result — baselines vs core

`delta = baseline − core`; positive means the baseline is more expensive. All CIs
exclude zero.

| Split | Core mean | Fixed | Proportional | Threshold |
|---|---:|---:|---:|---:|
| test_id | 13,065.7 | +2,795.6 | +2,991.0 | +3,586.6 |
| test_ood_burst | 25,197.2 | +8,550.2 | +2,783.6 | +1,334.7 |
| test_ood_traffic | 13,412.4 | +2,895.4 | +3,137.3 | +3,512.2 |

## Decision

Core is the retained incumbent. Ablating `reassign`, `short-turn` or fairness each
raises cost, confirming their contribution. No L1/L2 algorithm change improved on
core, so there is no algorithm-improvement claim. This is a complete research
result, not a failed run.

## Limits

- Synthetic scenarios only; fixed 3-route topology; no real-city deployment claim.
- Perfect queue/load sensing, no transfers, no endogenous congestion.
- Training-seed variability is material (`seed_std` 147–322 on `test_id`), which
  is why day-bootstrap CIs are reported alongside per-seed spread.
- The `fairness_zero` burst contrast is inconclusive, not positive.
