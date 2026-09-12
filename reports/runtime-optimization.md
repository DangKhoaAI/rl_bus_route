# Runtime optimization report (O0–O1)

Date: **2026-09-12**. Spec: [docs/spec/runtime_optimize.md](../docs/spec/runtime_optimize.md).

Status: **O0 achieved. O1 achieved (opt-in). O2–O5 not implemented. Defaults unchanged.**

This report records measured evidence only. The comparison baseline is the
current Rust runtime after R4 and memory M1/M3, at 2 Torch threads
(`configs/experiments/core-threads2.toml`). Historical full-workflow walls
597.99 / 156.37 s from [rust-migration.md](rust-migration.md) are **not** the
denominator for these candidates.

## 1. O0 — locked 2-thread Rust baseline

### 1.1 Provenance

| Item | Value |
|---|---|
| Git revision | `5ee8cbcf24d925b85c43452de58b2b6b235d18b5` (dirty: O1 sources + this report) |
| Native library sha256 | `aedc4faa4c434fd1f9565f08586332da0560e09d6349eef7dfebf961eb75bab2` (same frozen R4 kernel) |
| rustc / cargo | 1.97.1 / 1.97.1 |
| Lock (`uv.lock`) | sha256 `5ed56349910166b5fe6232c3d79c8421c6714e8cc8504cc5ac785d68f4ddb530` |
| Config | `configs/experiments/core-threads2.toml` + `runtime.backend=rust` |
| Checkpoint | `tests/backend_parity/reference/diagnose-after/last.zip` sha256 `98f99ebf151cdf6dde914df95ea780c25f4856e6a36f90e30ccd7a9f5906501e` |
| Seeds | algorithm 11; validation split seed 2001 |
| Machine | Linux x86_64, 24 CPUs, Python 3.11.14, torch 2.14.0+cpu, numpy 2.4.6 |
| Torch threads | requested 2, actual 2; interop 16; OMP/MKL/OPENBLAS unset |
| Flags (speed runs) | TIMERS off, cProfile off, conservation off, `validate_observation=true`, trace off |

Raw: `reports/runtime-optimization/o0-unprofiled-run1.json` (and run2).

### 1.2 Unprofiled component timings (TIMERS off, profiler off)

Two process launches. Nested child spans are **not** summed into a published
total. Isolated learn is 12,288 transitions, 4 envs, n_steps=256, batch=256,
n_epochs=4, no periodic eval. Validation is the 100-day fixed checkpoint.

| Segment | Run 1 median | Run 2 median |
|---|---:|---:|
| load/pack (100 val days + store) | 1.755 s | 1.635 s |
| env reset (16 scenarios) | 0.0006 s | 0.0004 s |
| isolated learn wall | **3.077 s** | **3.029 s** |
| ↳ rollout collect (not added to total) | 2.006 s | 1.974 s |
| ↳ PPO update (not added to total) | 1.070 s | 1.054 s |
| validation 100-day wall | **3.936 s** | **3.819 s** |
| ↳ inference | 3.149 s | 3.056 s |
| ↳ native step + wrapper | 0.563 s | 0.544 s |
| ↳ mask | 0.024 s | 0.025 s |
| ↳ summary | 0.185 s | 0.180 s |
| ↳ reset | 0.009 s | 0.009 s |
| checkpoint save / load | 0.005 / 0.026 s | (same order) |
| DummyVecEnv 64 steps (4 envs) | 0.011 s | 0.011 s |
| peak RSS | 448 MB | 449 MB |

Primary numbers are consistent across the two launches (learn ~3.03–3.08 s,
100-day eval ~3.82–3.94 s). Actual transitions were 12,288 in every learn
repetition.

### 1.3 Profile ranking (not a speed number)

TIMERS on, 4096-transition learn + 10-day eval. Nested `env.observe` lives
under `env.step` and is not added to it.

| Rank | Phase | Span | Seconds | Calls |
|---|---|---|---:|---:|
| 1 | eval | `eval.infer` | 0.316 | 1200 |
| 2 | learn | `env.step` | 0.151 | 4096 |
| 3 | learn | `env.observe` (nested) | 0.061 | 4136 |
| 4 | eval | `env.step` | 0.052 | 1200 |
| 5 | eval | `eval.summarize` | 0.026 | 10 |

The 2-thread profile **agrees** with the eval-inference hypothesis: validation
is predict-bound, so O1 batched evaluation is the right first stage. Unprofiled
100-day eval spends ~80% of wall in `model.predict` (3.15 / 3.94 s).

O0 is **achieved**.

## 2. O1 — batched evaluation and pool reuse (opt-in)

Shipped defaults remain the scalar evaluator:

- `runtime.eval_batch_size = 1`
- `runtime.reuse_eval_pool = false`
- training `n_envs=4`, `n_steps=256`, `batch_size=256`, `n_epochs=4` unchanged
- no `BatchKernel`, no native worker pool, no extra cache/prefetch, no default flip

Unsupported combinations fail loudly (no silent fallback):

- `eval_batch_size>1` with `random` / other heuristics
- `eval_batch_size>1` with `forecast.enabled` but no fitted forecaster
- stateful shared forecaster without per-slot state

`HistoricalForecaster` is stateless: each env slot predicts from its own
observation and time and is reset with the episode, so batch+forecast **matches
scalar** rather than erroring. The scalar forecast path is unchanged.

### 2.1 Parity

In-repo tests `tests/backend_parity/test_runtime_o1.py` drive
`evaluate_scenarios` / `mean_cost` / `BestValidationCallback` (16 passed):

| Check | Result |
|---|---|
| `batch=1` via pool vs scalar | same PPO actions and per-scenario metrics except `wall_s` |
| batch 4/8/16/32 vs scalar on the committed checkpoint, 10 days, partial last batch | same hashes, metrics, action indices |
| done slots / mixed horizons | short episode 4 steps, long 120; done slot not stepped |
| 20 consecutive evals, batch=4, 3 days | metrics match scalar; pool size capped; kernels do not grow |
| pool invalidation | order, flags, `validate_observation`, forecast, physical contract; stale pool refused |
| logits/metrics not cached across weight changes | perturbed checkpoint matches scalar of the new weights |
| batch+forecast | matches scalar; missing forecaster errors |
| random + batch>1 | `ValueError` (scalar path still runs at batch=1) |
| callback tie | strict less-than; mode/RNG restored; pool released on `finalize` |

No action divergence. First-divergence logging is in the test helper and was
not triggered.

`python -m pytest tests/backend_parity -q`: **118 passed, 1 skipped**.

### 2.2 Ablation (100 validation days, TIMERS off)

Protocol: warm-up 1; **5 interleaved repetitions**; same checkpoint/scenarios;
Torch threads=2. Two process launches. Every variant evaluated **100 days**
(not a shrunk eval-limit). Mean `total_cost` was **16100.7925** on every
repetition of every variant (parity at the ablation workload).

Median wall, run 1 / run 2:

| Variant | Run 1 median | Run 2 median | vs scalar (run 2) | RSS median |
|---|---:|---:|---:|---:|
| scalar (default path) | 4.501 s | 4.190 s | 1.00× | 439 MB |
| batch-1 (batched path, new pool) | 4.979 s | 4.645 s | 0.90× | 439 MB |
| pool-1 (reuse) | 5.002 s | 4.583 s | 0.91× | 439 MB |
| batch-4 | 2.110 s | 1.948 s | **2.15×** | 439 MB |
| pool-4 | 2.010 s | 1.961 s | 2.14× | 439 MB |
| batch-8 | 1.475 s | 1.431 s | **2.93×** | 439 MB |
| pool-8 | 1.523 s | 1.437 s | 2.92× | 439 MB |
| batch-16 | 1.180 s | 1.153 s | **3.63×** | 439 MB |
| pool-16 | 1.202 s | 1.196 s | 3.50× | 439 MB |
| batch-32 | 1.117 s | 1.016 s | **4.12×** | 439 MB |
| pool-32 | 1.066 s | 1.007 s | 4.16× | 439 MB |

Peak RSS of the ablation process: 438 MB (run 1) / 439 MB (run 2), vs O0
baseline ~448 MB. No size was skipped for RAM.

Batching is the speed win. Pool reuse does not add a clear extra benefit on a
**single** 100-day evaluation (env construction is cheap vs inference; the
store cache already hits). Reuse is still the right lifetime for the
validation callback (20 evals/seed) and is correctness-tested; it is not
folded into the batching speedup.

`batch=1` on the batched path is slightly slower than the scalar path
(stacking overhead) and is kept for compatibility, not as a speed setting.
Smallest batch in the group that is clearly faster than scalar is **4**.

O1 is **achieved** as opt-in. Defaults stay scalar.

## 3. What was not done

- O2 native `BatchKernel` / sequential batch step
- O3 native workers / `SubprocVecEnv`
- O4 extra caches, prefetch, multi-seed launcher
- O5 full 245,760-transition workflow gates or flipping defaults
- R0–R4 / M1 / M3 rewrites; deferred M2/M4/M5
- Training hyperparameter changes

## 4. How to use O1

In a run TOML (does not change `configs/experiments/core-threads2.toml`):

```toml
[runtime]
backend = "rust"
eval_batch_size = 8
reuse_eval_pool = true
```

`BestValidationCallback` owns the pool when `reuse_eval_pool` is true and
releases it in `finalize()`. Invalid combinations raise `ValueError`.

Rollback: omit the fields or set `eval_batch_size=1` and `reuse_eval_pool=false`.

## 5. Commands and artifacts

| Command | Exit | Artifact |
|---|---:|---|
| `python -m pytest tests/backend_parity/test_runtime_o1.py -n 0 --tb=short` | 0 | 16 passed |
| `python -m pytest tests/backend_parity -q` | 0 | 118 passed, 1 skipped |
| `python scripts/benchmark_runtime.py --mode o0-unprofiled ...` (×2) | 0 | `o0-unprofiled-run{1,2}.json` |
| `python scripts/benchmark_runtime.py --mode o0-profile ...` | 0 | `o0-profile.json` |
| `python scripts/benchmark_runtime.py --mode o1-ablation --repetitions 5 --eval-days 100` (×2) | 0 | `o1-ablation-run{1,2}.json` |

Exact argv: `reports/runtime-optimization/commands.json`.

Code: `src/bus_rl/evaluation/runner.py`, `evaluation/pool.py`,
`training/callbacks.py`, `config.py` (`RuntimeConfig.eval_batch_size` /
`reuse_eval_pool`).
