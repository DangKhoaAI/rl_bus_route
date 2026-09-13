# Runtime optimization report (O0–O2)

Date: **2026-09-12**. Spec: [docs/spec/runtime_optimize.md](../docs/spec/runtime_optimize.md).

Status: **O0 achieved. O1 achieved (opt-in). O2 achieved (opt-in). O3 deferred.
PPO update kept as an open candidate. O4/O5 not implemented. Defaults
unchanged.**

This report records measured evidence only. The comparison baseline is the Rust
runtime after R4 and memory M1/M3, at 2 Torch threads
(`configs/experiments/core-threads2.toml`). Historical full-workflow walls
597.99 / 156.37 s from [rust-migration.md](rust-migration.md) are **not** the
denominator for these candidates.

Machine: Linux x86_64, 24 CPUs, Python 3.11.14, torch 2.14.0+cpu, numpy 2.4.6,
rustc/cargo 1.97.1, Torch threads=2, TIMERS and profiler off unless a row says
otherwise. Host speed drifts between sessions; relative numbers inside one
session are the comparison, absolute walls are not.

## 1. O0 — locked 2-thread Rust baseline

### 1.1 Provenance

| Item | Value |
|---|---|
| Git revision | `2d00b5c` + working tree for O2 |
| Native library sha256 (frozen R4/O0/O1) | `aedc4faa4c434fd1f9565f08586332da0560e09d6349eef7dfebf961eb75bab2` |
| Native library sha256 (O2, installed) | `74b3a8fabbbabb5d4b167b92e56ac30af6f0957c180db78b6c501b21c50622ae` |
| Lock (`uv.lock`) | sha256 `5ed56349910166b5fe6232c3d79c8421c6714e8cc8504cc5ac785d68f4ddb530` |
| Config | `configs/experiments/core-threads2.toml` + `runtime.backend=rust` |
| Checkpoint | `tests/backend_parity/reference/diagnose-after/last.zip` sha256 `98f99ebf151cdf6dde914df95ea780c25f4856e6a36f90e30ccd7a9f5906501e` |
| Seeds | algorithm 11; validation split seed 2001 |

**Provenance fix.** Checkpoint metadata used to report the hash from the frozen
`reports/rust-migration/native-build.json` (`aedc4faa…`) even after the kernel
was rebuilt, so the O2 full run's metadata named a binary that was not loaded.
`native_build_info()` now hashes `src/bus_sim.so` at run time, records every
available build record, and names which record matches
(`build_record_matches_runtime`). Every full-workflow run below now records
`74b3a8fa…` and `["runtime-optimization"]`. The R4 build record stays frozen;
the O2 build is recorded separately in `native-build-o2.json`.

Baseline and candidates all ran the same installed binary, so the comparisons
are same-binary.

### 1.2 Unprofiled component timings (TIMERS off, profiler off)

Two process launches. Nested child spans are **not** summed into a published
total. Isolated learn is 12,288 transitions, 4 envs, n_steps=256, batch=256,
n_epochs=4, no periodic eval. Validation is the 100-day fixed checkpoint.

| Segment | Run 1 median | Run 2 median |
|---|---:|---:|
| load/pack (100 val days + store) | 1.755 s | 1.635 s |
| env reset (16 scenarios) | 0.0006 s | 0.0004 s |
| isolated learn wall | **3.077 s** | **3.029 s** |
| ↳ rollout collect | 2.006 s | 1.974 s |
| ↳ PPO update | 1.070 s | 1.054 s |
| validation 100-day wall | **3.936 s** | **3.819 s** |
| ↳ inference | 3.149 s | 3.056 s |
| ↳ native step + wrapper | 0.563 s | 0.544 s |
| ↳ summary | 0.185 s | 0.180 s |
| peak RSS | 448 MB | 449 MB |

O0 is **achieved**.

## 2. O1 — batched evaluation and pool reuse (opt-in)

Defaults stay scalar (`eval_batch_size=1`, `reuse_eval_pool=false`); training
hyperparameters unchanged. `tests/backend_parity/test_runtime_o1.py` covers
parity, pool invalidation, forecast, tie rule and RNG restoration.

Ablation (100 days, 5 interleaved reps, two launches), median wall run 1 / run 2:
scalar 4.501 / 4.190 s, batch-4 2.110 / 1.948, batch-8 1.475 / 1.431,
batch-16 1.180 / 1.153, batch-32 1.117 / 1.016. `batch=1` on the batched path
is slower than scalar (stacking overhead) and exists for compatibility only.

### 2.1 O1 post-batching breakdown — where the wall moved

`o1-breakdown`, perf_counter segments, TIMERS off, 5 reps, median run 1 / run 2:

| Batch | Wall | Inference | Native step + wrapper | Summary | Infer % | Step % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.487 / 4.971 | 3.525 / 3.920 | 0.578 / 0.624 | 0.265 / 0.287 | 78.6% / 78.9% | 12.9% / 12.6% |
| 4 | 1.923 / 2.080 | 1.137 / 1.250 | 0.435 / 0.472 | 0.256 / 0.278 | 59.1% / 60.1% | 22.6% / 22.7% |
| 8 | 1.379 / 1.521 | 0.657 / 0.739 | 0.398 / 0.439 | 0.250 / 0.276 | 47.7% / 48.6% | 28.8% |
| 16 | 1.085 / 1.225 | 0.417 / 0.470 | 0.376 / 0.421 | 0.240 / 0.274 | 38.4% | 34.6% / 34.4% |
| 32 | 0.969 / 1.066 | 0.302 / 0.333 | 0.373 / 0.407 | 0.245 / 0.268 | 31.1% / 31.2% | 38.5% / 38.2% |

After batching, inference stops dominating; at batch 32 native step (~38%) and
the Python episode summary (~25%) outweigh it. That chose O2. Peak RSS ~418 MB;
mask/reset <0.04 s.

## 3. O2 — native batch step (opt-in)

### 3.1 What shipped

`bus_sim.BatchKernel` owns N episode slots over one packed `ScenarioStore`
(`reset_batch`, `mask_batch`, `step_batch`, `current_time_s_batch`,
`summary_inputs_batch`, `trace_snapshot_slot`). One `step_batch` advances one
control interval per active slot and returns `(N, *single_env_shape)` arrays.
Shapes, slot ids and all actions/masks are validated before mutation; an
invalid batch raises `ValueError` and leaves every slot unchanged; a mid-batch
internal error poisons the kernel until reset (no partial success).

Python: `NativeBatchVecEnv` (SB3 `VecEnv`, mirrors `DummyVecEnv` auto-reset,
seeding and terminal info), `NativeBatchEvalPool` (shares `eval_pool_key` so the
Python and native pools cannot be mixed), and
`evaluation/runner.py::_evaluate_batched_native`. Opt-in via
`runtime.native_batch` / `--native-batch`; recorded in metadata.

### 3.2 Correctness

`tests/backend_parity/test_runtime_o2.py` (12 passed): scalar `Kernel` vs
`BatchKernel` exact replay (4 scenarios × 60 steps); mixed horizons; retained
outputs unchanged by later steps/resets; invalid batch rejected before mutation;
partial reset; `VecEnv` vs `DummyVecEnv` over 300 masked-action steps incl.
terminal info; `VecEnv.seed` scenario parity; **2048-transition training parity
(policy and optimizer tensors bit-identical)**; native eval vs scalar frames
(rtol 1e-12) and traces; pool invalidation; native batch + forecast.
`python -m pytest -q` → **178 passed, 1 skipped**.

### 3.3 Ablation (100 validation days, TIMERS off)

Warm-up 1; 5 interleaved repetitions; two launches. Median run 1 / run 2 (s):

| Variant | Wall | Inference | Step | Summary | Infer % of wall | Step % |
|---|---:|---:|---:|---:|---:|---:|
| o1-batch-8 | 1.535 / 1.398 | 0.749 / 0.678 | 0.445 / 0.404 | 0.283 / 0.255 | 49% | 29% |
| o2-native-8 | 1.252 / 1.235 | 0.704 / 0.662 | **0.196 / 0.188** | 0.297 / 0.317 | 56% | 16% |
| o1-batch-16 | 1.250 / 1.128 | 0.461 / 0.432 | 0.412 / 0.384 | 0.277 / 0.252 | 38% | 33% |
| o2-native-16 | 1.003 / 0.925 | 0.448 / 0.421 | **0.186 / 0.177** | 0.279 / 0.256 | 45% | 19% |
| o1-batch-32 | 1.064 / 0.985 | 0.328 / 0.308 | 0.401 / 0.374 | 0.267 / 0.252 | 31% | 38% |
| o2-native-32 | 0.870 / 0.835 | 0.319 / 0.298 | **0.180 / 0.172** | 0.293 / 0.308 | 37% | 21% |

Native step drops ~2.2× (0.41 → 0.18 s at batch 16). **Correction:** the full
workflow uses batch 16, where eval is inference ~45%, summary ~29%, step ~19% —
the "inference 37% / summary 34%" reading was batch 32.

**Summary split** (`summary_export_s` = native `summary_inputs_batch`;
`summary_convert_s` = `from_native_payload`; `summary_metrics_s` =
`summarize_inputs`), native-16 run 1 / run 2:

| Part | Seconds | Share of summary |
|---|---:|---:|
| native export | 0.0055 / 0.0051 | ~2% |
| `from_native_payload` | 0.1949 / 0.1801 | ~70% |
| `summarize_inputs` | 0.0742 / 0.0704 | ~27% |
| total | 0.279 / 0.256 | 100% |

So the native export is negligible; the cost is Python conversion — one
`CohortView` dataclass per cohort row in `from_native_payload`. That is a
narrower target than the earlier "dict/list construction" wording, and the
whole summary is only ~5–6 s of the full run.

Training (isolated learn 12,288 transitions, one env-side change):

| Variant | Learn wall | Rollout | PPO update | Transitions |
|---|---:|---:|---:|---:|
| `DummyVecEnv` | 3.411 / 3.197 s | 2.233 / 2.114 s | 1.177 / 1.087 s | 12,288 |
| `NativeBatchVecEnv` | 2.852 / 2.858 s | 1.771 / 1.771 s | 1.081 / 1.085 s | 12,288 |

Rollout drops ~16–20%; update unchanged. O2 is **achieved** as opt-in.

## 4. Full-workflow verification (245,760 transitions, 20 × 100 validation)

Single sequential session, seed 11, 2 Torch threads. Raw:
`o2-full-workflow.json`; logs/checkpoints under `runs/runtime-optimization/full/`.

| Variant | Flags | Total wall | Learn wall | Peak RSS | Best val cost |
|---|---|---:|---:|---:|---:|
| baseline | (defaults) | 151.76 s | 149.72 s | 449,996 KB | 13199.1325 |
| O1 | `--eval-batch-size 16 --reuse-eval-pool` | 94.21 s | 92.02 s | 452,152 KB | 13199.1325 |
| O2 | `--native-batch --eval-batch-size 16 --reuse-eval-pool` | 85.96 s | 83.74 s | 451,660 KB | 13199.1325 |
| O2 batch 32 | `--native-batch --eval-batch-size 32 --reuse-eval-pool` | 85.87 s | 83.58 s | 457,880 KB | 13199.1325 |

| Gate | Result |
|---|---|
| Full workflow ≥10% | O1 **1.611×**, O2 **1.765×**, O2-batch32 **1.767×** |
| Memory ≤1.20× baseline | 1.005× / 1.004× / 1.018× |
| Validation curve | 20 points, schedule identical, max abs cost diff **0.0** (all) |
| Policy/optimizer tensors | `last.zip`/`best.zip` max abs diff **0.0** (all) |

A previous session measured baseline 143.53 s, O1 87.79 s (1.635×) and O2
77.61 s (1.849×); the absolute drift is host noise, the relative picture is the
same.

**Batch 32 vs 16:** isolated eval says ~0.13 s saved per 100-day validation
(~2.7 s/run), but the full run shows 85.96 vs 85.87 s (**1.001×**, i.e. inside
noise). The spec's tie rule applies: keep the **smaller batch 16**. No
memory/perf reason to move to 32.

`learn_wall_s` includes validation and finalize/checkpoint. Approximate
component budget for O2 batch 16, from the component benchmarks (not a direct
breakdown of 85.96 s): rollout **~34–36 s**, PPO update **~22 s**, validation
**~18–19 s**, setup + other + unmeasured overhead **~9 s**. Ordering:
**rollout → PPO update → validation**. These are estimates across runs and must
not be summed as an exact total.

## 5. Re-profile and decisions

### 5.1 Rollout profile (priority 1; explanatory, sub-step timers on)

One rollout = 1024 transitions (4 envs × n_steps 256), 3 repetitions, median:

| Sub-step | Native O2 (s) | Share of rollout | Dummy (s) |
|---|---:|---:|---:|
| policy forward (total) | 0.0862 | 60.3% | 0.0851 |
| ↳ feature extraction | 0.0385 | 26.9% | 0.0381 |
| ↳ action distribution | 0.0256 | 17.9% | 0.0257 |
| ↳ sampling (`get_actions`) | 0.0131 | 9.2% | 0.0129 |
| ↳ forward other | 0.0090 | 6.3% | 0.0085 |
| env step (native step + obs conversion) | 0.0345 | 24.1% | 0.0552 |
| buffer add | 0.0069 | 4.8% | 0.0071 |
| `obs_as_tensor` | 0.0035 | 2.4% | 0.0035 |
| mask retrieval | 0.0032 | 2.2% | 0.0044 |
| GAE | 0.0008 | 0.6% | 0.0008 |
| other | 0.0076 | 5.3% | 0.0069 |
| **rollout total** | **0.1430** | 100% | 0.1630 |

The rollout bottleneck is the **policy forward (~60%)**, not the environment.
Native batch already cut env step from 34% to 24.1%. Tensor conversion, mask
and buffer are small; GAE is negligible. The forward is the same network
compute as eval inference, so optimizing it further risks numerical parity and
needs a dedicated plan.

### 5.2 PPO update profile (explanatory)

One update = 1024 transitions, 4 epochs, 16 minibatches, median of 3:

| Sub-step | Seconds | Share |
|---|---:|---:|
| forward (`policy.evaluate_actions`) | 0.0295 | 35.7% |
| backward (`loss.backward`) | 0.0279 | 33.8% |
| optimizer (`Adam.step`) | 0.0141 | 17.1% |
| grad clip | 0.0035 | 4.2% |
| other (loss math, minibatch gather) | 0.0078 | 9.4% |
| total | 0.0826 | 100% |

Forward + backward ≈ 70% and Adam ≈ 17%. The update is a real second bottleneck
(~22 s/run); it is **kept as an open candidate**, not rejected. Choosing a
change needs a follow-up profile and a numerical-parity check
(`torch.compile`/AMP are not free under the bit-identical requirement).

### 5.3 Decisions

| Item | Decision | Evidence |
|---|---|---|
| O3 parallel native workers | **deferred** | native step already 0.17–0.18 s/eval (19–21%); validation ~18–19 s of an ~86 s run; remaining cost is policy forward and Python summary/metadata, which workers do not fix; thread oversubscription on top of 2 Torch threads |
| Full-run eval batch 32 | **not adopted** | 1.001× vs batch 16; spec tie rule keeps batch 16 |
| Episode summary native array path | **candidate, not yet chosen** | summary is ~5–6 s/run; `from_native_payload` is ~70% of it (per-cohort dataclass loop), native export ~2% |
| PPO update optimization | **open candidate** | update ~22 s/run; forward+backward ~70%, Adam ~17%; no safe drop-in yet |
| Rollout policy-forward optimization | **top open candidate by size** | rollout ~34–36 s/run; forward ~60% of it (env step already cut to ~24% by O2) |

Next step by evidence: profile the policy forward deeper (or evaluate a
parity-safe torch compile/AMP path) and the `from_native_payload` cohort loop,
then re-run the O2 probe. Do not add threads or default a native summary yet.

## 6. What was not done

- O3 native workers / `SubprocVecEnv`
- O4 extra caches, prefetch, multi-seed launcher
- O5 multi-repetition full-workflow protocol, default flips
- PPO update / policy-forward optimization (open candidates, §5)
- `from_native_payload` fast path (candidate, §5.3)
- R0–R4 / M1 / M3 rewrites; deferred M2/M4/M5; hyperparameter changes

## 7. How to use O1 and O2

Defaults stay scalar. TOML:

```toml
[runtime]
backend = "rust"
eval_batch_size = 16
reuse_eval_pool = true
native_batch = true
```

CLI:

```bash
python -m bus_rl.cli train --config configs/experiments/core-threads2.toml \
  --manifest data/generated/base/manifest.json --seed 11 \
  --output runs/<name> --backend rust --torch-threads 2 --eval-limit 100 \
  --native-batch --eval-batch-size 16 --reuse-eval-pool
```

Invalid combinations raise `ValueError` (heuristic + batch>1; native batch
without a fitted per-slot forecaster; `native_batch` with `backend="python"`).
`BestValidationCallback` owns and releases the pool in `finalize()`.
Rollback: omit the fields (scalar evaluator + `DummyVecEnv`).

## 8. Commands and artifacts

| Command | Exit | Artifact |
|---|---:|---|
| `python -m pytest -q` | 0 | 178 passed, 1 skipped |
| `python scripts/build_native.py --output reports/runtime-optimization/native-build-o2.json` | 0 | O2 kernel sha256 `74b3a8fa…` |
| `python scripts/benchmark_runtime.py --mode o1-breakdown --repetitions 5 --warmup 1 --eval-days 100 …` | 0 | §2.1 |
| `python scripts/benchmark_runtime.py --mode o2-ablation --repetitions 5 --warmup 1 --eval-days 100 …` | 0 | §3.3 eval + training + summary split |
| `python scripts/benchmark_runtime.py --mode rollout-profile --repetitions 3 …` | 0 | §5.1 |
| `python scripts/benchmark_runtime.py --mode ppo-update-profile --repetitions 3 …` | 0 | §5.2 |
| `python -m bus_rl.cli train … --eval-batch-size 16` / `--native-batch` / `… 32` | 0 | §4 full workflow (4 variants) |

Exact argv: `reports/runtime-optimization/commands.json`. Code:
`evaluation/runner.py`, `evaluation/pool.py`, `env/native_batch.py`,
`training/{train,callbacks,checkpoint}.py`, `backend/native.py`, `config.py`,
`cli.py`, `crates/bus-sim-py/src/lib.rs`, `scripts/{benchmark_runtime,build_native}.py`.
