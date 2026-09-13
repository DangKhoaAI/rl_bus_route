# Runtime optimization report (O0–O2)

Date: **2026-09-12**. Spec: [docs/spec/runtime_optimize.md](../docs/spec/runtime_optimize.md).

Status: **O0 achieved. O1 achieved (opt-in). O2 achieved (opt-in). O3 deferred.
PPO update and rollout-forward work are open candidates. O4/O5 not implemented.
Defaults unchanged.**

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

| Item | Value |
|---|---|
| Git revision | `dff4fc9` + working tree for the forward-toggle work |
| Native library sha256 (frozen R4/O0/O1) | `aedc4faa4c434fd1f9565f08586332da0560e09d6349eef7dfebf961eb75bab2` |
| Native library sha256 (O2, installed) | `74b3a8fabbbabb5d4b167b92e56ac30af6f0957c180db78b6c501b21c50622ae` |
| Config | `configs/experiments/core-threads2.toml` + `runtime.backend=rust` |
| Checkpoint | `tests/backend_parity/reference/diagnose-after/last.zip` sha256 `98f99ebf151cdf6dde914df95ea780c25f4856e6a36f90e30ccd7a9f5906501e` |
| Seeds | algorithm 11; validation split seed 2001 |

Provenance: `native_build_info()` hashes `src/bus_sim.so` at run time and names
the matching build record. Every run below records `74b3a8fa…` and
`build_record_matches_runtime=["runtime-optimization"]`. The R4 record stays
frozen; the O2 build is in `native-build-o2.json`. Baseline and candidates all
ran the same installed binary.

O0 unprofiled baseline (12,288-transition isolated learn + 100-day validation,
two launches): learn 3.077 / 3.029 s, validation 3.936 / 3.819 s, peak RSS
448 / 449 MB. O0 is **achieved**.

## 2. O1 — batched evaluation and pool reuse (opt-in)

Defaults stay scalar (`eval_batch_size=1`, `reuse_eval_pool=false`). Ablation
(100 days, 5 interleaved reps): scalar 4.19–4.50 s, batch-4 ~1.95, batch-8
~1.43, batch-16 ~1.15, batch-32 ~1.02. `batch=1` on the batched path is slower
than scalar (stacking) and exists for compatibility only.

### 2.1 O1 post-batching breakdown — where the wall moved

`o1-breakdown`, perf_counter segments, TIMERS off, 5 reps, median run 1 / run 2:

| Batch | Wall | Inference | Native step + wrapper | Summary | Infer % | Step % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.151 / 4.159 | 3.158 / 3.192 | 0.562 / 0.561 | 0.271 / 0.270 | 76.1% / 76.7% | 13.5% |
| 4 | 1.798 / 1.802 | 1.034 / 1.036 | 0.430 / 0.433 | 0.259 / 0.258 | 57.5% | 23.9% |
| 8 | 1.335 / 1.335 | 0.616 / 0.615 | 0.402 / 0.402 | 0.258 / 0.255 | 46.1% / 46.0% | 30.1% |
| 16 | 1.081 / 1.079 | 0.391 / 0.393 | 0.383 / 0.383 | 0.254 / 0.250 | 36.2% / 36.4% | 35.4% |
| 32 | 0.954 / 0.953 | 0.279 / 0.281 | 0.374 / 0.372 | 0.253 / 0.250 | 29.3% / 29.5% | 39.1% |

After batching, inference stops dominating; native step and the episode summary
grow in share. That chose O2.

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

`tests/backend_parity/test_runtime_o2.py`: scalar `Kernel` vs `BatchKernel`
exact replay (4 scenarios × 60 steps); mixed horizons; retained outputs;
invalid batch rejected before mutation; partial reset; `VecEnv` vs `DummyVecEnv`
over 300 masked-action steps incl. terminal info; `VecEnv.seed` scenario parity;
**2048-transition training parity (policy + optimizer tensors bit-identical)**;
native eval vs scalar frames (rtol 1e-12) and traces; pool invalidation; native
batch + forecast; and the forward validation toggle bit-identical. `pytest -q`
→ **179 passed, 1 skipped**.

### 3.3 Ablation (100 validation days, TIMERS off)

Warm-up 1; 5 interleaved repetitions; two launches. Median run 1 / run 2 (s):

| Variant | Wall | Inference | Step | Summary | Infer % | Step % |
|---|---:|---:|---:|---:|---:|---:|
| o1-batch-8 | 1.332 / 1.331 | 0.613 / 0.612 | 0.404 / 0.404 | 0.254 / 0.255 | 46% | 30% |
| o2-native-8 | 1.168 / 1.165 | 0.594 / 0.596 | **0.186 / 0.187** | 0.315 / 0.316 | 51% | 16% |
| o1-batch-16 | 1.089 / 1.132 | 0.394 / 0.391 | 0.386 / 0.386 | 0.253 / 0.274 | 36% | 35% |
| o2-native-16 | 0.878 / 0.929 | 0.380 / 0.382 | **0.175 / 0.177** | 0.255 / 0.275 | 43% | 20% |
| o1-batch-32 | 0.953 / 1.010 | 0.280 / 0.281 | 0.372 / 0.377 | 0.250 / 0.271 | 29% | 38% |
| o2-native-32 | 0.807 / 0.805 | 0.272 / 0.271 | **0.172 / 0.172** | 0.306 / 0.308 | 34% | 21% |

Native step drops ~2.2× (0.386 → 0.175 s at batch 16). The full workflow uses
batch 16, where eval is inference ~43%, summary ~29%, step ~20%.

**Summary split** (`summary_export_s` = native `summary_inputs_batch`;
`summary_convert_s` = `from_native_payload`; `summary_metrics_s` =
`summarize_inputs`), native-16 run 1 / run 2:

| Part | Seconds | Share of summary |
|---|---:|---:|
| native export | 0.0051 / 0.0053 | ~2% |
| `from_native_payload` | 0.1787 / 0.1929 | ~70% |
| `summarize_inputs` | 0.0709 / 0.0713 | ~27% |
| total | 0.255 / 0.275 | ~5–6 s/run |

The native export is negligible; the cost is the per-cohort Python dataclass
loop in `from_native_payload`. The whole summary is only ~5–6 s of the run.

Training (isolated learn 12,288 transitions, one env-side change):

| Variant | Learn wall | Rollout | PPO update | Transitions |
|---|---:|---:|---:|---:|
| `DummyVecEnv` | 2.967 / 2.950 s | 1.913 / 1.902 s | 1.055 / 1.046 s | 12,288 |
| `NativeBatchVecEnv` | 2.780 / 2.613 s | 1.665 / 1.566 s | 1.120 / 1.045 s | 12,288 |

Rollout drops ~12–18%; update unchanged. O2 is **achieved** as opt-in.

## 4. Full-workflow verification (245,760 transitions, 20 × 100 validation)

Single sequential session, seed 11, 2 Torch threads, distribution validation
off. Raw: `o2-full-workflow.json`; logs/checkpoints under
`runs/runtime-optimization/full/`.

| Variant | Flags | Total wall | Learn wall | Peak RSS | Best val cost |
|---|---|---:|---:|---:|---:|
| baseline | (defaults) | 139.41 s | 137.38 s | 447,856 KB | 13199.1325 |
| O1 | `--eval-batch-size 16 --reuse-eval-pool` | 85.89 s | 83.77 s | 452,196 KB | 13199.1325 |
| O2 | `--native-batch --eval-batch-size 16 --reuse-eval-pool` | 74.94 s | 72.78 s | 450,436 KB | 13199.1325 |
| O2 batch 32 | `--native-batch --eval-batch-size 32 --reuse-eval-pool` | 73.26 s | 71.20 s | 456,312 KB | 13199.1325 |

| Gate | Result |
|---|---|
| Full workflow ≥10% | O1 **1.623×**, O2 **1.860×**, O2-batch32 **1.903×** |
| Memory ≤1.20× baseline | 1.010× / 1.006× / 1.019× |
| Validation curve | 20 points, schedule identical, max abs cost diff **0.0** (all) |
| Policy/optimizer tensors | `last.zip`/`best.zip` max abs diff **0.0** (all) |

**Batch 32 vs 16:** this session 74.94 vs 73.26 s (**1.023×**, ~1.7 s); the
previous session showed 1.001×. Isolated eval predicts up to ~2 s/run. That is
≤2.7% and inside run-to-run noise, far below the 10% component gate; the spec
tie rule keeps the smaller **batch 16**.

`learn_wall_s` includes validation and finalize/checkpoint. Approximate O2
component budget (from component benchmarks, not a direct breakdown of 74.94 s):
rollout **~32–34 s**, PPO update **~21 s**, validation **~17–18 s**, setup +
other + unmeasured overhead **~6–8 s**. Ordering: **rollout → PPO update →
validation**.

## 5. Re-profile and decisions

### 5.1 Policy-forward overhead — a cheap win adopted (priority 1)

The rollout profile (§5.2) showed the policy forward is the largest single
block. Profiling the forward at the sub-method level (batch 4, action dim 221,
native path) attributes ~254–260 µs/call:

| Part | µs/call | Share |
|---|---:|---:|
| `extract_features` (9-key flatten + MLP) | 73.0 | 28% |
| `mlp_extractor` | 23.0 | 9% |
| first action-distribution construction | 59.0 | 23% |
| `apply_masking` (second construction) | 33.6 | 13% |
| `sample` | 39.9 | 15% |
| `log_prob` | 25.6 | 10% |
| `value_net` | 3.2 | 1% |

So ~60% of the forward is **maskable-distribution bookkeeping**, not the MLP.
Torch's `Distribution._validate_args` defaults to `True`, and
`MaskableCategorical` constructs `Categorical` **twice per forward**
(`proba_distribution` then `apply_masking`), each running simplex `all`/`eq`
validation. Disabling validation only (it checks inputs and raises; it does not
change math) gives:

| Measurement | Validation on | off | Speedup |
|---|---:|---:|---:|
| policy forward (µs/call) | 253.6 | 206.8 | 1.227× |
| rollout (s) | 0.143 | 0.133 | 1.11× |
| eval inference (native-16, s) | 0.421 | 0.388 | 1.08× |
| PPO update (native, s) | 1.085 | 1.048 | 1.03× |

Outputs are **bit-identical**: sampled actions, values, log-probs and
deterministic argmax all compare equal (new test
`test_disabling_distribution_validation_is_bit_identical`). The toggle lives in
`bus_rl.runtime.apply_torch_threads` / `configure_torch_distributions`, so
train/eval/bench share it.

Also tested and rejected: a **pre-made tensor mask** (0.6% — `as_tensor` is
cheap), **`torch.set_flush_denormal(True)`** (no effect), **`torch_threads=1`**
(slower, 271.9 µs; 2 threads is faster and resulting logits are equal),
**`torch.inference_mode()`** (~8% but inference tensors are unsafe to reuse in
the later PPO update).

Left on the table: eliminating the **double distribution construction** (single
`MaskableCategorical(logits=..., masks=...)`) is bit-identical in a micro-test
(sample/log-prob equal) and ~19% of the distribution work, i.e. ~13% of the
forward. It needs a custom maskable distribution/policy class plus
checkpoint-compatibility and parity handling, so it is **deferred** as a
follow-up, not adopted now.

### 5.2 Rollout profile (explanatory)

One rollout = 1024 transitions (4 envs × n_steps 256), 3 reps, median:

| Sub-step | Native O2 (s) | Share | Dummy (s) |
|---|---:|---:|---:|
| policy forward (total) | 0.0747 | 56.0% | 0.0758 |
| ↳ feature extraction | 0.0398 | 29.8% | 0.0384 |
| ↳ action distribution | 0.0172 | 12.9% | — |
| ↳ sampling | 0.0144 | 10.8% | — |
| ↳ forward other | 0.0033 | 2.5% | — |
| env step | 0.0361 | 27.1% | 0.0585 |
| buffer add | 0.0071 | 5.3% | — |
| `obs_as_tensor` + mask | 0.0068 | 5.1% | — |
| GAE | 0.0008 | 0.6% | — |
| other | 0.0078 | 5.9% | — |
| **rollout total** | **0.1335** | 100% | 0.1584 |

After the §5.1 toggle, the forward share fell from ~60% to 56% and the
distribution part from 0.0256 to 0.0172 s; env step is now the clear second
block at 27%. Native batch already cut env step from 0.0585 (dummy) to 0.0361.

### 5.3 PPO update profile (explanatory)

One update = 1024 transitions, 4 epochs, 16 minibatches, median of 3: total
0.0882 s; forward 0.0293 (33%), backward 0.0305 (35%), Adam 0.0157 (18%), clip
0.0038 (4%), other 0.0089 (10%). Forward + backward ≈ 68%. The update is ~21 s
of the run and is **kept as an open candidate**.

### 5.4 Decisions

| Item | Decision | Evidence |
|---|---|---|
| Skip Torch distribution validation | **adopted** | forward 1.23×, rollout 1.11×, eval infer 1.08×, bit-identical |
| O3 parallel native workers | **deferred** | native step 0.17 s/eval; rollout is forward-bound, workers do not fix it |
| Full-run eval batch 32 | **not adopted** | 1.023× (≤2.7%, in noise); tie rule keeps batch 16 |
| Fast maskable distribution (no double construction) | **deferred** | bit-identical, ~13% of forward; needs custom policy + checkpoint handling |
| `from_native_payload` fast path | **candidate** | ~70% of a ~5–6 s/run summary |
| PPO update optimization | **open candidate** | ~21 s/run; forward+backward ~68% |

## 6. What was not done

- O3 native workers / `SubprocVecEnv`
- O4 extra caches, prefetch, multi-seed launcher
- O5 multi-repetition full-workflow protocol, default flips
- Fast maskable-distribution / policy-forward optimization (deferred, §5.1)
- `from_native_payload` fast path; PPO update optimization
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

Invalid combinations raise `ValueError`. Rollback: omit the fields (scalar
evaluator + `DummyVecEnv`).

## 8. Commands and artifacts

| Command | Exit | Artifact |
|---|---:|---|
| `python -m pytest -q` | 0 | 179 passed, 1 skipped |
| `python scripts/build_native.py --output reports/runtime-optimization/native-build-o2.json` | 0 | O2 kernel sha256 `74b3a8fa…` |
| `python scripts/benchmark_runtime.py --mode o1-breakdown …` | 0 | §2.1 |
| `python scripts/benchmark_runtime.py --mode o2-ablation …` | 0 | §3.3 eval + training + summary split |
| `python scripts/benchmark_runtime.py --mode rollout-profile …` | 0 | §5.2 |
| `python scripts/benchmark_runtime.py --mode ppo-update-profile …` | 0 | §5.3 |
| `python -m bus_rl.cli train …` (4 variants) | 0 | §4 |

Exact argv: `reports/runtime-optimization/commands.json`. Code:
`evaluation/runner.py`, `evaluation/pool.py`, `env/native_batch.py`,
`training/{train,callbacks,checkpoint}.py`, `backend/native.py`, `runtime.py`,
`config.py`, `cli.py`, `crates/bus-sim-py/src/lib.rs`,
`scripts/{benchmark_runtime,build_native}.py`.
