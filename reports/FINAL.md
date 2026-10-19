# Final result

**Question:** can MaskablePPO learn operational bus-fleet control — reserve
dispatch, headway target, reassign, recall, short-turn — better than fixed and
heuristic controllers on a fixed 3-route network under time-varying demand?

**Answer (synthetic held-out data):** yes for this simulator. The PPO core has
lower mean cost than `fixed`, `proportional` and `threshold` on all three
held-out splits, and every component ablation is worse. No L1/L2 algorithm change
was adopted; the original core remains the incumbent.

Status: final for the study frozen on 2026-09-13. Authoritative numbers are in
[`studies/03-rl-improvement/tables/`](studies/03-rl-improvement/tables/) and
[`studies/03-rl-improvement/l3_protocol.json`](studies/03-rl-improvement/l3_protocol.json).

---

## Setup in one paragraph

Three fixed bidirectional routes (6 stops each), 12 buses (9 assigned + 3 reserve),
capacity 40. Every 2 minutes the controller chooses one masked action from a
221-slot table: `NOOP`, `DISPATCH`, `REASSIGN`, `RECALL`, `SHORT_TURN` or
`SET_HEADWAY`. The simulator runs on 30-second ticks over a 240-minute horizon.
Demand and traffic are synthetic; the controller sees queues, ages, loads,
positions, headways and past arrivals, but not future demand or passenger
destinations. Training uses MaskablePPO on CPU, 245,760 transitions per seed,
seeds 11/22/33. Checkpoints are selected on 100 validation days. The final matrix
is evaluated once on 200 `test_id`, 200 `test_ood_burst` and 200
`test_ood_traffic` days.

---

## Main result: PPO core vs baselines

`delta = baseline − core`, so positive means the baseline is more expensive.
Paired 200-day bootstrap, 95% CI. Every CI excludes zero.

| Split | PPO core mean cost | Fixed | Proportional | Threshold |
|---|---:|---:|---:|---:|
| `test_id` | 13,065.7 | +2,795.6 | +2,991.0 | +3,586.6 |
| `test_ood_burst` | 25,197.2 | +8,550.2 | +2,783.6 | +1,334.7 |
| `test_ood_traffic` | 13,412.4 | +2,895.4 | +3,137.3 | +3,512.2 |

Source: [`l3_baseline_contrasts.csv`](studies/03-rl-improvement/tables/l3_baseline_contrasts.csv),
[`l3_baselines_summary.csv`](studies/03-rl-improvement/tables/l3_baselines_summary.csv),
[`l3_held_out_summary.csv`](studies/03-rl-improvement/tables/l3_held_out_summary.csv).

---

## Ablations: each component contributes

`delta = ablation − core`; positive means removing the component is worse.

| Ablation | `test_id` | `test_ood_burst` | `test_ood_traffic` |
|---|---:|---:|---:|
| no `REASSIGN` | +0.47% | +0.92% | +0.33% |
| no `SHORT_TURN` | +1.65% | +0.77% | +1.51% |
| no fairness weight | +2.12% | +0.93% (CI crosses 0) | +1.88% |

Core is lowest on every split. No ablation has a CI below zero. The
`fairness_zero` burst contrast is inconclusive, not positive. Every ablation also
violates the registered P95/worst-route service limits on at least one split; only
core satisfies all of them.

Source: [`l3_contrasts.csv`](studies/03-rl-improvement/tables/l3_contrasts.csv),
[`l3_paired_days.csv`](studies/03-rl-improvement/tables/l3_paired_days.csv),
[`l3_verification.json`](studies/03-rl-improvement/tables/l3_verification.json).

---

## What was tried and rejected

| Change | Decision | Result |
|---|---|---|
| GAE `lambda` 0.98 / 1.0 | rejected at screening | +0.54% / +1.78% validation cost |
| Causal 15-min arrival forecast | rejected after 3 seeds | +2.03%, CI [240.38, 295.72] above 0 |
| Potential-based reward shaping | rejected after 3 seeds | +3.59%, CI [447.66, 498.57] above 0 |

See [`LEDGER.md`](LEDGER.md) and the cards under
[`studies/03-rl-improvement/`](studies/03-rl-improvement/).

---

## What this shows

- Within the defined simulator, the learned controller beats the three programmed
  baselines on ID, burst and traffic splits.
- `REASSIGN`, `SHORT_TURN` and the fairness term each improve the measured cost;
  removing them is worse.
- The benefit is not explained by a single hyperparameter trick or by forecasting:
  those alternatives were tested and rejected under pre-registered rules.

## What this does not show

- It is **not** a real-city deployment result. Demand and traffic are synthetic.
- The topology is fixed (3 routes); this does not claim generalization to new
  networks.
- Queue/load sensing is perfect, there are no passenger transfers, no endogenous
  congestion, no driver rostering or breakdowns.
- Training-seed variability is material (`seed_std` 147–322 on `test_id`), so
  per-seed spread is reported alongside day-bootstrap CIs.
- The controller is not proven optimal, and no L1/L2 change is claimed as an
  improvement.

---

## Where to read more

| Need | File |
|---|---|
| One-page result | this file |
| All experiments and decisions | [`LEDGER.md`](LEDGER.md) |
| Full study narrative | [`studies/03-rl-improvement/README.md`](studies/03-rl-improvement/README.md) |
| L3 details | [`studies/03-rl-improvement/l3-matrix/README.md`](studies/03-rl-improvement/l3-matrix/README.md) |
| Problem and RL formulation | [`../docs/problem_solution_brief.md`](../docs/problem_solution_brief.md) |
| Full specification | [`../docs/spec/spec_v1.0.md`](../docs/spec/spec_v1.0.md) |
| Infrastructure evidence | [`infra/README.md`](infra/README.md) |
