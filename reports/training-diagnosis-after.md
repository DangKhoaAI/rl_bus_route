# Training diagnosis after hot-path optimizations

Same command and seed as the baseline trial (2048 transitions, 10-day eval,
`n_envs=4`, `n_steps=256`). Eval mean cost is unchanged: **15471.25**.

## Log locations (do not mix)

| Role | Path |
|---|---|
| **Old (before)** | `reports/training-diagnose/` and `reports/training-diagnosis.md` |
| **New (after)** | `reports/training-diagnose-after/` and this file |
| Raw rerun dirs | `runs/diagnose/` (old), `runs/diagnose-after/` (new, gitignored) |

## Wall clock

| Stage | Before | After | Change |
|---|---:|---:|---|
| learn() | 21.71 s | **7.69 s** | **2.82×** |
| Rollout collect | 21.42 s | 7.35 s | 2.91× |
| PPO update | 0.28 s | 0.34 s | ~same |
| Eval 10 days | 8.63 s | 3.46 s | 2.49× |
| Decisions/s | 94 | **266** | 2.82× |
| Projected 245,760-step seed | ~43 min | **~15 min** | |

## Env timers (sum of two rollouts)

| Function | Before s | After s |
|---|---:|---:|
| env.step | 20.32 | 6.04 |
| engine.ticks | 18.02 | 2.48 |
| engine.complete_phase | 5.38 | 0.49 |
| engine.board | 4.63 | 0.32 |
| engine.tick_costs | 3.17 | 0.26 |
| engine.abandon | 2.26 | 0.14 |
| engine.conservation | 1.72 | ~0 (skipped in train) |
| env.observe | 1.45 | 2.57 |
| env.action_masks | 0.81 | 0.89 |

`observe()` is now the largest remaining Python cost because it still walks
waiting + onboard + finished via `iter_cohorts()`. Mask computation is next.

## What changed in code

1. `assert_conservation` only when `CONSERVATION_CHECKS` (pytest autouse; off in train).
2. Incremental `waiting/onboard/completed/abandoned` counters.
3. Cached `Vehicle.load`.
4. Hot list is waiting-only; completed/abandoned go to `state.finished`.
5. `IntEnum` instead of `StrEnum` for phase/pattern/status.
