# Tables

Human-facing tables for the final claim. All numbers are copied from the raw
curated tables in this directory; no hand-editing.

## Read these first

| File | What it is |
|---|---|
| [`main_benchmark.csv`](main_benchmark.csv) | PPO core vs `fixed`, `proportional`, `threshold` on the three held-out splits |
| [`ablations.csv`](ablations.csv) | `no_reassign`, `no_short`, `fairness_zero` vs core, with CIs and service-limit flags |
| [`l3-details.md`](l3-details.md) | Longer L3 write-up: setup, full tables, decision, limits |

## Raw curated inputs

| File | Role |
|---|---|
| [`baselines_summary.csv`](baselines_summary.csv) | Baseline means per split (200 days) |
| [`baseline_contrasts.csv`](baseline_contrasts.csv) | Paired baseline − core deltas and 95% CIs |
| [`held_out_per_seed.csv`](held_out_per_seed.csv) | Per-seed held-out metrics for all four arms |
| [`ablation_contrasts.csv`](ablation_contrasts.csv) | Paired ablation − core deltas, CIs and service deltas |

## Column guide

### `main_benchmark.csv`

| Column | Meaning |
|---|---|
| `split` | `test_id`, `test_ood_burst`, `test_ood_traffic` |
| `method` | `core`, `fixed`, `proportional`, `threshold` |
| `eval_days` | 200 days per split |
| `seeds` | 3 for PPO core; 1 for deterministic baselines |
| `mean_cost_per_episode` | Mean `total_cost_core` per day; lower is better |
| `seed_std_cost` | Std of per-seed mean cost (0 for baselines) |
| `delta_vs_core` | `method − core` in cost units; positive = worse |
| `delta_ci95_low/high` | Paired 200-day bootstrap 95% CI |
| `delta_pct_of_core` | `delta_vs_core / core` in percent |
| `seeds_lower_than_core` | Number of PPO seeds with lower cost (blank for baselines) |
| `mean_wait_min` | Mean observed wait, minutes |
| `p95_wait_min` | P95 observed wait, minutes |
| `unfinished_share` | Share of generated passengers still waiting/onboard at the end |
| `completed_share` | Share of generated passengers completed |

### `ablations.csv`

| Column | Meaning |
|---|---|
| `ablation` | `no_reassign`, `no_short`, `fairness_zero` |
| `delta_cost` | `ablation − core` in cost units; positive = worse |
| `delta_ci95_low/high` | Paired 200-day bootstrap 95% CI |
| `delta_pct_of_core` | Relative cost difference, percent |
| `seed_std` | Std of per-seed mean cost for the ablation |
| `delta_p95_wait_min` | P95 wait difference, minutes |
| `delta_worst_route_wait_min` | Worst-route mean wait difference, minutes |
| `delta_unfinished`, `delta_abandoned` | Share differences |
| `service_limit_pass` | Whether all registered service limits hold on this split |
| `service_limit_violations` | Which limits fail (`;`-separated) |

## Benchmark protocol

```text
Metric: total_cost_core (lower is better), normalized by N_ref = 3000
Seeds: 3 (11/22/33)
Budget: 245,760 transitions per seed
Eval: 200 days per split: test_id / test_ood_burst / test_ood_traffic
Checkpoint: best validation cost on 100 days; tie -> earliest
CI: paired day bootstrap, 95%, 2,000 resamples, seed 6001
Baselines: fixed, proportional, threshold on the same days
```

The cost is a synthetic operational proxy: weighted passenger waiting, onboard
time, crowding, bus-minutes, first-denied boarding, abandonment, mission changes,
and unfinished passengers at the end of the shift. It is not a financial cost.

## Regenerate

```bash
.venv/bin/python scripts/l3_analysis.py          # writes the raw L3 tables/figures
.venv/bin/python scripts/build_benchmark_tables.py  # writes main_benchmark.csv, ablations.csv
```

`l3_analysis.py` reads the raw git-ignored `runs/rl-improvement/` tree.
`build_benchmark_tables.py` only reads the curated CSVs in this directory.
