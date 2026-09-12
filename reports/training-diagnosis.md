# Training time diagnosis

Trial: `uv run bus-rl diagnose --config configs/experiments/core.toml --manifest data/generated/base/manifest.json --timesteps 2048 --eval-limit 10 --output runs/diagnose`

Same PPO hyperparams as a full seed (`n_envs=4`, `n_steps=256`, `n_epochs=4`, `batch_size=256`, CPU). 2 rollouts + 10-day validation.

Eval data (committed):

- `reports/training-diagnosis.md` — summary
- `reports/training-diagnose/diagnosis.json` — timers, eval cost, projection
- `reports/training-diagnose/train_events.jsonl` — per-rollout collect vs PPO update
- `reports/training-diagnose/cprofile.txt` — top functions
- `reports/training-diagnose/metadata.json` — run hashes / wall time

Raw rerun artifacts (gitignored, includes checkpoint): `runs/diagnose/`.

## Wall clock

| Stage | Seconds | Share of learn() |
|---|---:|---:|
| Load 500 train days | 0.66 | (setup) |
| Env + model construct | 0.53 | (setup) |
| Rollout collect (env) | 21.42 | **98.7%** |
| PPO `train()` update | 0.28 | **1.3%** |
| learn() total | 21.71 | 100% |
| Validation 10 days | 8.63 | 0.86 s/day |

Throughput while learning: **94 decisions/s**. Projected 245,760-step seed: **~43 min train + ~3 min of 10-day evals** ≈ **46 min/seed** if nothing is optimized. DummyVecEnv is sequential; `n_envs=4` does **not** parallelize the simulator.

PPO/torch is not the bottleneck (`torch.nn.linear` 0.18 s tottime, backward 0.08 s). GPU will not save this run.

## Per control step (~10 ms)

Nested timers (children sit inside parents; do not add every row):

```
env.step                         10.0 ms
  env.advance_interval            9.0 ms
    engine.ticks                  8.9 ms   (4 ticks × 30s)
      engine.complete_phase       2.7 ms
      engine.board                2.3 ms
      engine.tick_costs           1.6 ms
      engine.abandon              1.1 ms
      engine.conservation         0.8 ms
      engine.arrivals             0.15 ms
    engine.apply_action           0.15 ms
  env.observe                     0.72 ms
  env.action_masks                0.40 ms   (called twice: PPO + step guard)
  env.interval_cost               ~0
```

Eval is the same shape (`eval.summarize` is 1.2 ms/episode, negligible).

## Hottest functions (cProfile tottime, 21.7 s learn)

| tottime | Function | Why it is hot |
|---:|---|---|
| **7.54 s** | `WorldState._count` genexpr | Every `waiting_count` / `onboard_count` / … rescans **all** cohorts. 45M iterations. |
| 2.45 s | `sum()` | Driven by `_count`, `Vehicle.load`, cost integrals. |
| 1.32 s | `board_visit` candidate genexpr | Full cohort scan per terminal-idle bus per tick. |
| 1.23 s | `observe()` | Rebuilds stop/vehicle tensors from scratch. |
| 1.10 s | `enum.Enum.__get__` | `c.status is PassengerStatus.WAITING` / `.value` on every scan. |
| 0.98 s | `integrate_tick_costs` genexpr | Full cohort scan for excessive-wait + `bus.load` per vehicle. |
| 0.77 s | `board_visit` listcomp | Compact zero-count cohorts. |
| 0.65 s | `_eligible` | 6.7M calls from the board scan. |
| 0.58 s | `valid_action_mask` | 221-slot mask, twice per step. |
| 0.52 s | `abandon_expired` | Full cohort scan every tick. |
| 0.50 s | `Vehicle.load` genexpr | Recomputed constantly (costs, board, conservation). |
| 0.19 s | `advance_tick` itself | After children. |

`assert_conservation` is called **every tick and every `board_visit`**. Each call walks all four passenger statuses plus every bus load.

## What is already fast

- PPO update, GAE, backward
- `_add_arrivals`, `dispatch_ready`, `apply_action`
- `eval.summarize`
- Manifest load (0.66 s for 500 days)

## Optimize in this order

1. **Incremental passenger counters** on `WorldState` (`waiting/onboard/completed/abandoned`) instead of `_count` scans. Largest single win (~35% of learn()).
2. **Cache `Vehicle.load`** (invalidate on board/alight/split).
3. **Stop calling `assert_conservation` in the hot path**; keep it behind a debug flag or every N ticks. It currently stacks on top of (1) and (2).
4. **Index waiting cohorts by `(route, origin, direction)`** so `board_visit` / `_eligible` do not scan completed/onboard/abandoned people.
5. **`integrate_tick_costs`**: use cached waiting/onboard and a running excessive-wait mass; drop `status.value == "WAITING"` (string) in favor of enum identity.
6. **Call `action_masks()` once** per step (MaskablePPO already has it; `BusDispatchEnv.step` computes it again).
7. Only then consider `SubprocVecEnv` (`n_envs=4` is currently 4 serial Python envs).

Logs during a normal `bus-rl train` now print one line per PPO rollout:

```text
[train] timesteps=12288  +1024  elapsed=12.3s  fps=94.1
```

Re-run a diagnosis after an optimization with the same command as the top of this file.
