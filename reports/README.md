# Bus fleet-control study — final report

**Question:** can MaskablePPO learn operational bus-fleet control — reserve
dispatch, headway target, reassign, recall, short-turn — better than fixed and
heuristic controllers on a fixed 3-route network under time-varying demand?

**Answer (synthetic held-out data):** yes for this simulator. The PPO core has
lower mean cost than `fixed`, `proportional` and `threshold` on all three
held-out splits, and every component ablation is worse. No L1/L2 algorithm change
was adopted; the original core remains the incumbent.

Status: final for the study frozen on 2026-09-13. The authoritative numbers are in
[`tables/`](tables/) and the protocols are in [`provenance/`](provenance/).

---

## Config

| Item | Value |
|---|---|
| Network | 3 fixed bidirectional routes, 6 stops each |
| Fleet | 12 buses: 9 assigned + 3 reserve; capacity 40 |
| Control interval | 120 s; horizon 240 min (120 decisions) |
| Simulator tick | 30 s |
| Actions | 221 masked slots: `NOOP`, `DISPATCH`, `REASSIGN`, `RECALL`, `SHORT_TURN`, `SET_HEADWAY` |
| Observation | queues, ages, loads, positions, headways, past arrivals; no future demand or hidden destinations |
| Algorithm | MaskablePPO, MLP [256, 128], actor-critic, GAE, gamma = 1.0 |
| Training | 245,760 transitions per seed; seeds 11/22/33; CPU |
| Checkpoint rule | best validation cost on 100 days; tie → earliest |
| Evaluation | 200 days each on `test_id`, `test_ood_burst`, `test_ood_traffic` |
| Metric | `total_cost_core`, lower is better, normalized by `N_ref = 3000` |
| Uncertainty | paired day bootstrap, 95% CI, 2,000 resamples |

The cost is a weighted sum of passenger waiting, onboard time, crowding,
operating and deadhead bus-minutes, first-denied boarding, abandonment, mission
changes, and unfinished passengers at the end of the shift. It is a synthetic
operational proxy, not a financial cost.

---

## Main result: PPO core vs baselines

Cells are `Δ vs core [95% CI]`; positive means the baseline is more expensive.
All CIs exclude zero. Core is the first column.

| Split | PPO core (mean ± seed std) | Fixed | Proportional | Threshold |
|---|---:|---:|---:|---:|
| `test_id` | 13,065.7 ± 52.4 | +2,795.6 [2,759.6, 2,831.6] | +2,991.0 [2,871.1, 3,114.8] | +3,586.6 [3,450.7, 3,727.8] |
| `test_ood_burst` | 25,197.2 ± 67.2 | +8,550.2 [8,276.7, 8,819.4] | +2,783.6 [2,184.8, 3,403.5] | +1,334.7 [760.8, 1,927.0] |
| `test_ood_traffic` | 13,412.4 ± 65.3 | +2,895.4 [2,850.0, 2,939.4] | +3,137.3 [3,014.4, 3,270.9] | +3,512.2 [3,385.1, 3,638.2] |

Full table with wait, P95 wait, unfinished and completed shares:
[`tables/main_benchmark.csv`](tables/main_benchmark.csv).

---

## Ablations: each component contributes

Cells are `Δ cost % vs core [95% CI]`; positive means removing the component is
worse.

| Split | No `REASSIGN` | No `SHORT_TURN` | No fairness weight |
|---|---:|---:|---:|
| `test_id` | +0.47% [0.33, 0.60] | +1.65% [1.51, 1.77] | +2.12% [1.91, 2.34] |
| `test_ood_burst` | +0.92% [0.07, 1.76] | +0.77% [0.06, 1.53] | +0.93% [−0.44, 2.41] |
| `test_ood_traffic` | +0.33% [0.20, 0.48] | +1.51% [1.38, 1.64] | +1.88% [1.66, 2.11] |

Core is lowest on every split. No ablation has a CI below zero. The
`fairness_zero` burst CI crosses zero: that single contrast is inconclusive, not
a positive difference. Every ablation violates at least one registered
P95/worst-route service limit on at least one split; only core satisfies all of
them. Details: [`tables/ablations.csv`](tables/ablations.csv),
[`tables/l3-details.md`](tables/l3-details.md).

Figures:

- [`figures/held_out_costs.png`](figures/held_out_costs.png) — absolute cost by arm and split.
- [`figures/service_tradeoffs.png`](figures/service_tradeoffs.png) — cost delta vs P95-wait delta.
- [`figures/validation_curves.png`](figures/validation_curves.png) — validation cost during training.

---

## What was tried and rejected

| Change | Decision | Result |
|---|---|---|
| GAE `lambda` 0.98 / 1.0 | rejected at screening | +0.54% / +1.78% validation cost |
| Causal 15-min arrival forecast | rejected after 3 seeds | +2.03%, CI [240.38, 295.72] above 0 |
| Potential-based reward shaping | rejected after 3 seeds | +3.59%, CI [447.66, 498.57] above 0 |

The change ledger is [`provenance/experiments-ledger.md`](provenance/experiments-ledger.md);
the rejected experiment cards are under
[`archive/rejected-experiments/`](archive/rejected-experiments/).

---

## What this shows

- Within the defined simulator, the learned controller beats the three programmed
  baselines on ID, burst and traffic splits.
- `REASSIGN`, `SHORT_TURN` and the fairness term each improve the measured cost;
  removing them is worse.
- The result is not explained by a single hyperparameter trick or by forecasting:
  those alternatives were tested and rejected under pre-registered rules.

## What this does not show

- It is **not** a real-city deployment result. Demand and traffic are synthetic.
- The topology is fixed (3 routes); this does not claim generalization to new
  networks.
- Queue/load sensing is perfect, there are no passenger transfers, no endogenous
  congestion, no driver rostering or breakdowns.
- Training-seed variability is material (`seed_std` 52–67 on these splits), so
  seed spread is reported alongside day-bootstrap CIs.
- The controller is not proven optimal, and no L1/L2 change is claimed as an
  improvement.

---

## Where things are

| Path | Contents |
|---|---|
| [`tables/`](tables/) | Final benchmark and ablation tables; see [`tables/README.md`](tables/README.md) |
| [`figures/`](figures/) | Final figures only |
| [`provenance/`](provenance/) | Protocols, verification, change ledger, L0 detail |
| [`archive/`](archive/) | Historical/rejected work, including pilot, forecast baseline, rejected experiments, infrastructure |
| [`../docs/problem_solution_brief.md`](../docs/problem_solution_brief.md) | Problem and RL formulation |
| [`../docs/spec/spec_v1.0.md`](../docs/spec/spec_v1.0.md) | Full specification |

The root has one README (this file) and four folders. The frozen Rust migration
evidence is under `archive/infrastructure/rust-migration/`.
