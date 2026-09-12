# Pilot report (Task 8)

Plumbing gate for reproducible training, not a claim that PPO beats heuristics.

## Setup

- Config: `configs/pilot.toml` (M3 flags, 12,288 transitions, MaskablePPO CPU, γ=1).
- Data: `data/generated/base` — 500/100/200/200/200 days, schema v2, physical hash locked in `manifest.json`.
- Run: `runs/pilot-11` seed 11. Best checkpoint by mean validation cost on 10 val days (tie: earliest).
- Device: CPU (`torch` 2.14.0+cpu). Git SHA recorded in `runs/pilot-11/metadata.json` (dirty while this report was written).

## Measured runtime (actual, not an estimate)

| Job | Actual wall time | Notes |
|---|---:|---|
| Generate 1,200 days | 22.6 s | full 500/100/200/200/200 split |
| Profile 120 decisions × 3 demands | ~2.2 s | see `runs/profile/profile.json` |
| Pilot train 12,288 transitions | 76.6 s | 100 completed episodes; eval_limit=10 |
| Val eval 100 days PPO | part of 250.6 s below | with event trace |
| Val eval 100 days × 3 baselines | remainder of 250.6 s | fixed, threshold, proportional |

Peak RSS during profiling: **378 MB**. Peak demand throughput: **78 decisions/s**, **311 ticks/s**.

cProfile tottime on a peak day (120 decisions, 1.54 s wall):

| Bucket | Seconds |
|---|---:|
| engine | 0.048 |
| passenger | 0.239 |
| sensor | 0.095 |
| PPO infer | 0.046 |
| other | 1.112 |

Passenger cohort work dominates peak days. Cache remains limited to immutable travel keys; dynamic queue state is not cached.

## Full-study estimate from this measurement

Extrapolation from 12,288 transitions in 76.6 s (including a 10-day validation pass):

- One 245,760-transition seed ≈ **20 × 76.6 s ≈ 26 min** (plus extra validation callbacks).
- 4 configs × 3 seeds ≈ **5.2 h** training if sequential on this CPU.
- Held-out eval 600 days × 12 models ≈ **1–2 h** at ~0.6 s/episode.

These are **estimates** derived from the table above, not completed T9 wall clocks.

## Validation (100 days, core-weighted cost)

| Method | Mean core cost | Mean wait (min) | Completed | Unfinished | Unique denied | Dispatch | Reassign | Short-turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 16017 | 7.41 | 1.000 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 |
| proportional | 16196 | 7.05 | 0.996 | 0.004 | 0.000 | 2.32 | 0.00 | 0.00 |
| threshold | 16641 | 7.20 | 0.997 | 0.003 | 0.000 | 3.57 | 0.10 | 0.04 |
| PPO pilot | 18207 | 6.77 | 0.983 | 0.017 | 0.0002 | 3.02 | 0.17 | 0.00 |

PPO has **not** beaten the heuristics after 12,288 transitions. Waiting is slightly lower; unfinished share and total cost are worse. That is enough to show the pipeline, masks, and metrics work. It is not a learning-success claim.

Event trace of validation day 0 includes a `DISPATCH` (reserve). Threshold used reassign and short-turn on the 100-day average; the pilot policy used reassign rarely and no short-turn.

Plots: `reports/pilot/plots/`.

## Lineage

Each run writes `metadata.json` with action/obs schema hashes, physical/control/reward hashes, git SHA/dirty, `uv.lock` hash, device, seeds, actual transitions, completed episodes, and wall time. Reusing an existing output directory exits non-zero. Loading a checkpoint with a different physical config or obs/action schema is rejected.
