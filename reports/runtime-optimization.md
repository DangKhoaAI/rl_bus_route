# Runtime optimization report (O0–O2)

Date: **2026-09-12**. Spec: [docs/spec/runtime_optimize.md](../docs/spec/runtime_optimize.md).

Status: **O0 achieved. O1 achieved (opt-in). O2 achieved (opt-in). O3 deferred.
PPO update deferred. O4/O5 not implemented. Defaults unchanged.**

This report records measured evidence only. The comparison baseline is the Rust
runtime after R4 and memory M1/M3, at 2 Torch threads
(`configs/experiments/core-threads2.toml`). Historical full-workflow walls
597.99 / 156.37 s from [rust-migration.md](rust-migration.md) are **not** the
denominator for these candidates.

Machine for every number below: Linux x86_64, 24 CPUs, Python 3.11.14,
torch 2.14.0+cpu, numpy 2.4.6, rustc/cargo 1.97.1, Torch threads=2, TIMERS and
profiler off unless a row says otherwise.

## 1. O0 — locked 2-thread Rust baseline

### 1.1 Provenance

| Item | Value |
|---|---|
| Git revision | `11a20f84bd291f1e30315dd6bada951f94f829fd` + working tree |
| Native library sha256 (frozen R4/O0/O1) | `aedc4faa4c434fd1f9565f08586332da0560e09d6349eef7dfebf961eb75bab2` |
| Native library sha256 (O2) | `74b3a8fabbbabb5d4b167b92e56ac30af6f0957c180db78b6c501b21c50622ae` |
| Lock (`uv.lock`) | sha256 `5ed56349910166b5fe6232c3d79c8421c6714e8cc8504cc5ac785d68f4ddb530` |
| Config | `configs/experiments/core-threads2.toml` + `runtime.backend=rust` |
| Checkpoint | `tests/backend_parity/reference/diagnose-after/last.zip` sha256 `98f99ebf151cdf6dde914df95ea780c25f4856e6a36f90e30ccd7a9f5906501e` |
| Seeds | algorithm 11; validation split seed 2001 |

The frozen R4 provenance record stays at
`reports/rust-migration/native-build.json`. The O2 kernel build is recorded
separately in `reports/runtime-optimization/native-build-o2.json`; the shared
`Kernel`/`ScenarioStore` surfaces and their golden fixtures are unchanged by O2
(the O2 diff only adds `BatchKernel` and refactors summary/trace dict builders
into free functions).

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

### 1.3 Profile ranking (not a speed number)

TIMERS on, 4096-transition learn + 10-day eval. The 2-thread profile agrees
with the eval-inference hypothesis: unprofiled 100-day eval spent ~80% of wall
in `model.predict` (3.15 / 3.94 s), so O1 batched evaluation was the right
first stage.

O0 is **achieved**.

## 2. O1 — batched evaluation and pool reuse (opt-in)

Shipped defaults remain the scalar evaluator: `runtime.eval_batch_size = 1`,
`runtime.reuse_eval_pool = false`; training `n_envs=4`, `n_steps=256`,
`batch_size=256`, `n_epochs=4` unchanged. Unsupported combinations fail loudly
(no silent fallback).

### 2.1 Parity

`tests/backend_parity/test_runtime_o1.py` (16 passed): `batch=1` vs scalar,
batch 4/8/16/32 vs scalar including a partial last batch, mixed horizons, 20
consecutive evals without leaking/mixing days, pool invalidation on
order/flags/config/forecast/contract, no metric reuse across weight changes,
batch+forecast parity, random heuristic errors, callback tie rule, model
mode/RNG restoration. No action divergence; `python -m pytest
tests/backend_parity -q` → **130 passed, 1 skipped** at the O2 tree.

### 2.2 Ablation (100 validation days, TIMERS off)

Warm-up 1; **5 interleaved repetitions**; two process launches. Every variant
evaluated **100 days** (not a shrunk eval-limit). Mean `total_cost` was
**16100.7925** on every repetition of every variant.

Median wall, run 1 / run 2:

| Variant | Run 1 | Run 2 | vs scalar (run 2) | RSS median |
|---|---:|---:|---:|---:|
| scalar (default path) | 4.501 s | 4.190 s | 1.00× | 439 MB |
| batch-1 (batched path) | 4.979 s | 4.645 s | 0.90× | 439 MB |
| batch-4 | 2.110 s | 1.948 s | **2.15×** | 439 MB |
| batch-8 | 1.475 s | 1.431 s | **2.93×** | 439 MB |
| batch-16 | 1.180 s | 1.153 s | **3.63×** | 439 MB |
| batch-32 | 1.117 s | 1.016 s | **4.12×** | 439 MB |

Batching is the win; pool reuse adds no clear extra benefit on a **single**
100-day evaluation (env construction is cheap vs inference). `batch=1` on the
batched path is slightly slower than scalar (stacking overhead) and is kept for
compatibility only. O1 is **achieved** as opt-in.

### 2.3 O1 post-batching breakdown — where the wall moved

`o1-breakdown` mode, perf_counter segments, TIMERS off, 5 reps. This is the
measurement that chose the next stage. Median run 1 / run 2 (seconds):

| Batch | Wall | Inference | Native step + wrapper | Summary | Inference share | Step share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.487 / 4.971 | 3.525 / 3.920 | 0.578 / 0.624 | 0.265 / 0.287 | 78.6% / 78.9% | 12.9% / 12.6% |
| 4 | 1.923 / 2.080 | 1.137 / 1.250 | 0.435 / 0.472 | 0.256 / 0.278 | 59.1% / 60.1% | 22.6% / 22.7% |
| 8 | 1.379 / 1.521 | 0.657 / 0.739 | 0.398 / 0.439 | 0.250 / 0.276 | 47.7% / 48.6% | 28.8% |
| 16 | 1.085 / 1.225 | 0.417 / 0.470 | 0.376 / 0.421 | 0.240 / 0.274 | 38.4% | 34.6% / 34.4% |
| 32 | 0.969 / 1.066 | 0.302 / 0.333 | 0.373 / 0.407 | 0.245 / 0.268 | 31.1% / 31.2% | 38.5% / 38.2% |

Process peak RSS ~418 MB; mask/reset are negligible (<0.04 s). After batching,
inference is no longer dominant: at batch 32 the native step + wrapper
(38%) and the Python episode summary (~25%) together outweigh it. That is the
evidence for O2 and for the O3/PPO decision in §5.

## 3. O2 — native batch step (opt-in)

### 3.1 What shipped

`bus_sim.BatchKernel` owns N episode states over one packed `ScenarioStore` and
offers `reset_batch`, `mask_batch`, `step_batch`, `current_time_s_batch`,
`summary_inputs_batch` and `trace_snapshot_slot`. One `step_batch(slots,
actions)` advances exactly one control interval per active slot and returns
`(N, *single_env_shape)` observation arrays plus `mask`, `reward`,
`terminated`, `truncated`, `costs`. Shapes, slot ids and every action/mask are
validated **before** any mutation; a duplicate/unreset/masked action raises
`ValueError` and leaves every slot unchanged. An internal error mid-batch marks
the kernel poisoned until the slots are reset (no partial success).

Python adapters (`src/bus_rl/env/native_batch.py`):

- `NativeBatchVecEnv`: SB3 `VecEnv` for training, mirroring `DummyVecEnv` over
  `NativeBusDispatchEnv`. Each slot owns a seeded `np_random`, picks its next
  scenario on (auto-)reset, sets `terminal_observation`/`TimeLimit.truncated`,
  and honors `VecEnv.seed` exactly like `DummyVecEnv`.
- `NativeBatchEvalPool`: fixed-slot evaluator pool sharing the same
  `eval_pool_key` as `EvalEnvPool`, so a pool built for the Python path is
  refused by the native path and vice versa.
- `evaluation/runner.py::_evaluate_batched_native`: one `step_batch` per control
  step for all active slots; terminal summaries are captured before slot reuse.

Config: `runtime.native_batch` (default `false`, rust only, validated in
`RuntimeConfig.__post_init__`). CLI: `--native-batch`. Metadata records
`native_batch`.

### 3.2 Correctness

`tests/backend_parity/test_runtime_o2.py` (12 passed):

| Check | Result |
|---|---|
| `BatchKernel` vs scalar `Kernel` replay, 4 scenarios × 60 steps | obs/mask/reward/terminated/costs exactly equal |
| mixed horizons (short 480 s, long default) | short slot removed at its own step; long slot keeps stepping |
| retained outputs | earlier `step_batch` arrays unchanged by later steps and by `reset_batch` |
| invalid batch | masked action, out-of-range action, length mismatch, duplicate slot, unreset slot, mixed valid+unreset → `ValueError`; clock unchanged |
| partial reset | resetting slot 0 leaves slot 1's clock/state untouched |
| VecEnv vs `DummyVecEnv`, 300 steps | obs/reward/done/terminal info equal on masked-action trajectories |
| `VecEnv.seed` | construction + `reset()` + auto-reset scenario indices match `DummyVecEnv` |
| training parity, 2048 transitions | policy **and** optimizer tensors bit-identical (max abs diff 0.0) |
| native eval vs scalar, 10 days, batch 8 | frames equal (rtol 1e-12) and traces identical |
| native pool invalidation | order/flags/`native_batch`/forecast all refused |
| native batch + forecast | matches scalar |

`python -m pytest tests/backend_parity -q` → **130 passed, 1 skipped**.

### 3.3 Ablation (100 validation days, TIMERS off)

Warm-up 1; 5 interleaved repetitions; two launches. Same checkpoint/scenarios;
effective batch = min(configured, 100). Median run 1 / run 2 (seconds):

| Variant | Wall | Inference | Step | Summary | Step share | RSS median |
|---|---:|---:|---:|---:|---:|---:|
| o1-batch-8 | 1.369 / 1.446 | 0.657 / 0.688 | 0.390 / 0.410 | 0.262 / 0.285 | 28.5% | 424 MB |
| o2-native-8 | 1.149 / 1.211 | 0.636 / 0.663 | **0.180 / 0.189** | 0.262 / 0.285 | 15.6% | 424 MB |
| o1-batch-16 | 1.118 / 1.180 | 0.422 / 0.443 | 0.377 / 0.396 | 0.264 / 0.283 | 33.7% | 424 MB |
| o2-native-16 | 0.907 / 0.955 | 0.408 / 0.428 | **0.171 / 0.178** | 0.267 / 0.283 | 18.8% | 424 MB |
| o1-batch-32 | 0.984 / 1.042 | 0.302 / 0.316 | 0.368 / 0.388 | 0.264 / 0.283 | 37.3% | 424 MB |
| o2-native-32 | 0.779 / 0.821 | 0.293 / 0.306 | **0.168 / 0.176** | 0.262 / 0.280 | 21.5% | 424 MB |

The native step + wrapper segment shrinks **~2.2×** (0.37 → 0.17 s at batch
16); total eval wall drops ~1.2× at the same batch. Inference is unchanged
(F32→Float32 conversion dominates it, not the stepping). After O2, eval is
inference-bound again (~37–40%) with the episode summary as the second cost
(~34% at batch 32).

Training (isolated learn 12,288 transitions, 4 envs, one env-side change only):

| Variant | Learn wall | Rollout | PPO update | Transitions |
|---|---:|---:|---:|---:|
| `DummyVecEnv` (scalar) | 3.210 / 3.185 s | 2.122 / 2.095 s | 1.088 / 1.094 s | 12,288 |
| `NativeBatchVecEnv` | 2.970 / 2.900 s | 1.848 / 1.812 s | 1.122 / 1.088 s | 12,288 |

Rollout collection drops ~12–13%; the PPO update is untouched (as expected).
O2 is **achieved** as opt-in. Defaults stay scalar.

## 4. Full-workflow verification (245,760 transitions, 20 × 100 validation)

Single paired repetition, same session, seed 11, 2 Torch threads, in order
baseline → O1 → O2. Raw: `o2-full-workflow.json`; logs/checkpoints under
`runs/runtime-optimization/full/`.

| Variant | Flags | Total wall | Learn wall | Peak RSS | Best val cost |
|---|---|---:|---:|---:|---:|
| baseline | (defaults) | 143.53 s | 141.56 s | 446,952 KB | 13199.1325 |
| O1 | `--eval-batch-size 16 --reuse-eval-pool` | 87.79 s | 85.73 s | 455,272 KB | 13199.1325 |
| O2 | `--native-batch --eval-batch-size 16 --reuse-eval-pool` | 77.61 s | 75.58 s | 450,304 KB | 13199.1325 |

| Gate | Result |
|---|---|
| Full workflow ≥10% | O1 **1.635×**, O2 **1.849×** total wall vs baseline |
| Memory ≤1.20× baseline | O1 1.019×, O2 1.008× |
| Validation curve | 20 points, schedule identical, max abs cost diff **0.0** (O1 and O2) |
| Policy/optimizer tensors | `last.zip`/`best.zip` max abs diff **0.0** (57 leaves incl. optimizer) |
| Episodes / transitions | 2,048 / 245,760 for every variant |

ZIP hashes differ across variants (timestamps); the comparison is tensor
content, per §4.2. This is one paired repetition; the multi-repetition protocol
in §10.1 of the spec is still required before flipping defaults.

## 5. Re-profile and decisions

### 5.1 PPO update profile (explanatory; sub-step timers on)

One update = 1024 transitions, 4 envs × n_steps 256, 4 epochs, 16 minibatches.
Median of 3 repetitions; timer overhead present, so not a speed number.

| Sub-step | Seconds | Share |
|---|---:|---:|
| forward (`policy.evaluate_actions`) | 0.0295 | 35.7% |
| backward (`loss.backward`) | 0.0279 | 33.8% |
| optimizer (`Adam.step`) | 0.0141 | 17.1% |
| grad clip | 0.0035 | 4.2% |
| other (loss math, minibatch gather) | 0.0078 | 9.4% |
| total | 0.0826 | 100% |

Forward + backward are ~70% and the optimizer ~17%; there is no single
dominant, safe drop-in. `torch.compile`/AMP would risk the bit-identical
requirement, and changing epochs/minibatches changes the algorithm. **PPO
update optimization is deferred/rejected with evidence**, not chosen.

### 5.2 O3 decision — deferred

After O2 the native step is 0.17 s of a 0.78–0.82 s 100-day eval (≈21%) and
validation is only ~23% of the full workflow (O2 total 77.6 s vs O1 87.8 s).
Parallel native workers would add thread oversubscription on top of the 2
Torch threads and Python-side construction, and the component gate asks for a
≥10% median win on the targeted part that exceeds noise. The measured eval
breakdown says the larger remaining costs are inference (~37%) and the Python
episode summary (~34%), neither of which is fixed by parallel env workers.

**Decision: keep `native_workers=1`; O3 deferred.** The next candidate with the
strongest evidence is the per-episode summary construction
(`summary_inputs_batch` builds a full state snapshot plus dict/list per day:
0.26 s / 100 days, ~2.6 ms per episode), followed by further batched-inference
work if it survives a larger-batch ablation.

## 6. What was not done

- O3 native workers / `SubprocVecEnv`
- O4 extra caches, prefetch, multi-seed launcher
- O5 multi-repetition full-workflow protocol, default flips
- PPO update optimization (deferred with the profile in §5.1)
- Episode-summary native array path (next candidate, not implemented)
- R0–R4 / M1 / M3 rewrites; deferred M2/M4/M5; training hyperparameter changes

## 7. How to use O1 and O2

Defaults stay scalar. In a run TOML (does not change
`configs/experiments/core-threads2.toml`):

```toml
[runtime]
backend = "rust"
eval_batch_size = 16
reuse_eval_pool = true
native_batch = true
```

or on the CLI:

```bash
python -m bus_rl.cli train --config configs/experiments/core-threads2.toml \
  --manifest data/generated/base/manifest.json --seed 11 \
  --output runs/<name> --backend rust --torch-threads 2 --eval-limit 100 \
  --native-batch --eval-batch-size 16 --reuse-eval-pool
```

Invalid combinations raise `ValueError`: `eval_batch_size>1` with a heuristic,
native batch without a fitted per-slot forecaster, and
`native_batch=true` with `backend="python"`. `BestValidationCallback` owns and
releases the pool in `finalize()`.

Rollback: omit the fields (scalar evaluator + `DummyVecEnv`).

## 8. Commands and artifacts

| Command | Exit | Artifact |
|---|---:|---|
| `python -m pytest tests/backend_parity -q --tb=line` | 0 | 130 passed, 1 skipped |
| `python scripts/build_native.py --output reports/runtime-optimization/native-build-o2.json` | 0 | O2 kernel sha256 `74b3a8fa…` |
| `python scripts/benchmark_runtime.py --mode o1-breakdown --repetitions 5 --warmup 1 --eval-days 100 --output …/o1-breakdown-run{1,2}.json` | 0 | §2.3 breakdown |
| `python scripts/benchmark_runtime.py --mode o2-ablation --repetitions 5 --warmup 1 --eval-days 100 --output …/o2-ablation-run{1,2}.json` | 0 | §3.3 eval + training |
| `python scripts/benchmark_runtime.py --mode ppo-update-profile --repetitions 3 --output …/ppo-update-profile.json` | 0 | §5.1 |
| `python -m bus_rl.cli train … --backend rust --torch-threads 2 --eval-limit 100` (baseline) | 0 | `runs/runtime-optimization/full/baseline` |
| `… --eval-batch-size 16 --reuse-eval-pool` (O1) | 0 | `runs/runtime-optimization/full/o1` |
| `… --native-batch --eval-batch-size 16 --reuse-eval-pool` (O2) | 0 | `runs/runtime-optimization/full/o2` |

Exact argv: `reports/runtime-optimization/commands.json`. Code:
`evaluation/runner.py`, `evaluation/pool.py`, `env/native_batch.py`,
`training/train.py`, `training/callbacks.py`, `training/checkpoint.py`,
`config.py`, `cli.py`, `crates/bus-sim-py/src/lib.rs`,
`scripts/benchmark_runtime.py`, `scripts/build_native.py`.
