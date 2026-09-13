# Runtime optimization report (O0–O2, O5 repetitions)

Date: **2026-09-12**. Spec: [docs/spec/runtime_optimize.md](../docs/spec/runtime_optimize.md).

Status: **O0 achieved. O1 achieved (opt-in). O2 achieved (opt-in). O3 deferred.
Torch distribution validation is configurable (on by default; opt-in off).
O5 has paired multi-repetition evidence but defaults are not flipped. PPO
update and rollout-forward work remain open candidates.**

This report records measured evidence only. The comparison baseline is the Rust
runtime after R4 and memory M1/M3, at 2 Torch threads
(`configs/experiments/core-threads2.toml`), i.e. the shipped defaults
(including distribution validation **on**). Historical walls 597.99 / 156.37 s
from [rust-migration.md](rust-migration.md) are **not** the denominator.

Machine: Linux x86_64, 24 CPUs, Python 3.11.14, torch 2.14.0+cpu, numpy 2.4.6,
rustc/cargo 1.97.1, Torch threads=2, TIMERS and profiler off unless stated.
Host speed drifts between (and within) sessions; paired ratios inside a session
are the comparison, absolute walls are not.

## 1. O0 — locked 2-thread Rust baseline

| Item | Value |
|---|---|
| Git revision | `ae15622` + working tree for the O5/config work |
| Native library sha256 (frozen R4/O0/O1) | `aedc4faa4c434fd1f9565f08586332da0560e09d6349eef7dfebf961eb75bab2` |
| Native library sha256 (O2, installed) | `74b3a8fabbbabb5d4b167b92e56ac30af6f0957c180db78b6c501b21c50622ae` |
| Config | `configs/experiments/core-threads2.toml` + `runtime.backend=rust` |
| Checkpoint | `tests/backend_parity/reference/diagnose-after/last.zip` sha256 `98f99ebf151cdf6dde914df95ea780c25f4856e6a36f90e30ccd7a9f5906501e` |
| Seeds | algorithm 11; validation split seed 2001 |

Provenance: `native_build_info()` hashes `src/bus_sim.so` at run time and names
the matching build record (`build_record_matches_runtime`). The R4 record stays
frozen; the O2 build is `native-build-o2.json`. Baseline and candidates ran the
same installed binary.

O0 unprofiled baseline (12,288-transition isolated learn + 100-day validation,
two launches): learn 3.077 / 3.029 s, validation 3.936 / 3.819 s, peak RSS
448 / 449 MB. O0 is **achieved**.

## 2. O1 — batched evaluation and pool reuse (opt-in)

Ablation (100 days, 5 interleaved reps): scalar ~4.2–4.5 s, batch-4 ~1.95,
batch-8 ~1.43, batch-16 ~1.15, batch-32 ~1.02. `batch=1` on the batched path is
slower than scalar (stacking) and exists for compatibility only.

### 2.1 O1 post-batching breakdown

`o1-breakdown`, perf_counter segments, TIMERS off, 5 reps, median run 1 / run 2:

| Batch | Wall | Inference | Native step + wrapper | Summary | Infer % | Step % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.274 / 4.405 | 3.296 / 3.370 | 0.582 / 0.602 | 0.283 / 0.288 | 77.1% / 76.5% | 13.6% |
| 4 | 1.895 / 1.915 | 1.075 / 1.103 | 0.445 / 0.459 | 0.272 / 0.275 | 56.7% / 57.6% | 23.5% |
| 8 | 1.375 / 1.437 | 0.638 / 0.660 | 0.413 / 0.427 | 0.266 / 0.275 | 46.4% / 45.9% | 30.0% |
| 16 | 1.135 / 1.155 | 0.416 / 0.422 | 0.402 / 0.407 | 0.259 / 0.271 | 36.6% / 36.5% | 35.4% |
| 32 | 0.987 / 1.022 | 0.293 / 0.303 | 0.387 / 0.397 | 0.263 / 0.269 | 29.7% / 29.6% | 39.2% |

## 3. O2 — native batch step (opt-in)

### 3.1 What shipped

`bus_sim.BatchKernel` owns N episode slots over one packed `ScenarioStore`
(`reset_batch`, `mask_batch`, `step_batch`, `current_time_s_batch`,
`summary_inputs_batch`, `trace_snapshot_slot`). One `step_batch` advances one
control interval per active slot and returns `(N, *single_env_shape)` arrays.
Shapes, slot ids and all actions/masks are validated before mutation; an invalid
batch raises `ValueError` and leaves every slot unchanged; a mid-batch internal
error poisons the kernel until reset (no partial success).

Python: `NativeBatchVecEnv`, `NativeBatchEvalPool`, and
`evaluation/runner.py::_evaluate_batched_native`. Opt-in via
`runtime.native_batch` / `--native-batch`; recorded in metadata.

### 3.2 Correctness

`tests/backend_parity/test_runtime_o2.py` (15 passed): scalar `Kernel` vs
`BatchKernel` exact replay (4 scenarios × 60 steps); mixed horizons; retained
outputs; invalid batch rejected before mutation; partial reset; `VecEnv` vs
`DummyVecEnv` over 300 masked-action steps incl. terminal info; `VecEnv.seed`
scenario parity; 2048-transition training parity (policy + optimizer tensors
bit-identical); native eval vs scalar frames (rtol 1e-12) and traces; pool
invalidation; native batch + forecast; distribution-validation on/off
(actions, values, log-probs, deterministic argmax and `model.predict` all
bit-identical); 2048-transition training parity on vs off; metadata records the
effective setting. `pytest -q` → **181 passed, 1 skipped**.

### 3.3 Ablation (100 validation days, TIMERS off)

Warm-up 1; 5 interleaved repetitions; two launches. Median run 1 / run 2 (s):

| Variant | Wall | Inference | Step | Summary |
|---|---:|---:|---:|---:|
| o1-batch-8 | 1.329 / 1.385 | 0.615 / 0.637 | 0.398 / 0.421 | 0.255 / 0.262 |
| o2-native-8 | 1.216 / 1.211 | 0.624 / 0.620 | **0.193 / 0.194** | 0.314 / 0.323 |
| o1-batch-16 | 1.123 / 1.131 | 0.400 / 0.408 | 0.386 / 0.402 | 0.259 / 0.261 |
| o2-native-16 | 0.897 / 0.916 | 0.378 / 0.395 | **0.172 / 0.182** | 0.258 / 0.265 |
| o1-batch-32 | 0.948 / 0.994 | 0.281 / 0.291 | 0.369 / 0.390 | 0.248 / 0.260 |
| o2-native-32 | 0.758 / 0.834 | 0.271 / 0.282 | **0.169 / 0.178** | 0.278 / 0.315 |

Native step drops ~2.2×. The full workflow uses batch 16.

**Summary split** (native-16): native export 0.0051 / 0.0054 s (~2%),
`from_native_payload` 0.1813 / 0.1850 s (~70%), `summarize_inputs`
0.0693 / 0.0740 s (~27%). The native export is negligible; the cost is the
per-cohort Python dataclass loop, and the whole summary is only ~5–6 s/run.

Training (isolated learn 12,288 transitions): `DummyVecEnv` 3.234 / 3.045 s
(rollout 2.075 / 1.966, update 1.133 / 1.077); `NativeBatchVecEnv`
2.783 / 2.755 s (rollout 1.676 / 1.658, update 1.106 / 1.096). Rollout drops
~15–19%; update unchanged.

## 4. Full-workflow multi-repetition (O5)

`scripts/benchmark_full_workflow.py`: each run is a separate process, the
variant order rotates every repetition, seed 11, 2 Torch threads, 245,760
transitions with 20×100 validation. Raw: `full-workflow-reps.json`.

`baseline` uses the shipped defaults (distribution validation **on**). The
opt-in variants use `--no-validate-distributions`; `o2b16-on` reruns the accepted
O2 config with validation on to isolate that toggle.

Per-run total wall (s):

| Variant | r0 | r1 | r2 | median | paired ratio vs baseline (r0/r1/r2) |
|---|---:|---:|---:|---:|---|
| baseline | 145.71 | 183.86 | 179.60 | 179.60 | 1.00 |
| o2b16 | 74.45 | 85.04 | 83.92 | 83.92 | 1.957 / 2.162 / 2.140 |
| o2b32 | 71.84 | 85.67 | 87.18 | 85.67 | 2.028 / 2.146 / 2.060 |
| o2b16-on | 87.49 | — | — | 87.49 | 1.666 |

The host drifted upward during the session (baseline r0 145.7 s vs r1/r2
~180 s; candidates also slowed), so **paired per-repetition ratios are the
primary number**: O2-off is 1.96–2.16× faster than the shipped defaults, and
O2-on (validation enabled) is 1.67×. Even the most conservative paired estimate
clears the 10% gate. The median-based ratios (2.10–2.14×) are inflated by the
baseline running later and longer.

**Batch 16 vs 32**, paired: 1.036× / 0.993× / 0.963× (32/16 wall). The sign
flips across repetitions, so the difference is noise; the tie rule keeps
**batch 16** (also the smaller memory footprint). This confirms the earlier
single-session 1.001×.

Peak RSS medians: baseline 447,196 KB; o2b16 450,516 (1.007×); o2b32 455,684
(1.019×); o2b16-on 453,412 (1.014×). Memory gate (≤1.20×) passes.

**Parity**: every one of the 10 runs has validation curve max abs diff **0.0**
and last/best tensor max abs diff **0.0** vs baseline-r0. Because `o2b16-on`
(validation on) and `o2b16` (off) are each bit-identical to the same baseline,
the toggle is bit-identical at the full seed — the direct `o2b16-on` vs
`o2b16` tensor diff is also 0.0. Best validation cost is 13199.1325 in every run.

An earlier single-repetition session (`o2-full-workflow.json`, validation off
for every variant) measured baseline 139.41 s, O1 85.89 s, O2 74.94 s,
O2-batch32 73.26 s; it is superseded by the paired multi-repetition evidence
above, but matches its ordering.

## 5. Re-profile and decisions

### 5.1 Policy-forward overhead — a configurable, bit-identical win

Sub-method profile of one `policy(obs, action_masks)` (batch 4, 221 actions):
~254–260 µs, of which `extract_features` 73 µs (28%), `mlp_extractor` 23 µs
(9%), first distribution construction 59 µs (23%), `apply_masking` 34 µs (13%),
`sample` 40 µs (15%), `log_prob` 26 µs (10%), `value_net` 3 µs. So ~60% is
maskable-distribution bookkeeping, and Torch constructs the categorical twice
per forward with `validate_args=True`.

`runtime.validate_distributions` (default **true**) controls the Torch
distribution argument validation. It is applied in
`bus_rl.runtime.apply_runtime_settings`, exposed as
`--validate-distributions/--no-validate-distributions`, and recorded in
metadata as both the requested value and the effective
`torch_distribution_validate_args`. The accepted opt-in config sets it **off**.

Controlled A/B (`distribution-validation.json`, same process/model/observations):

| Measurement | on | off | Speedup |
|---|---:|---:|---:|
| forward (µs/call) | 258.86 | 208.12 | **1.24×** |
| rollout (s) | 0.1568 | 0.1410 | 1.112× (10.1% less) |

`bit_identical = {actions, values, log_prob, argmax: true}` in the artifact, and
the unit tests add `model.predict` and 2048-transition training parity.

**Caveat, stated explicitly:** a valid action mask does **not** guarantee finite
logits/probabilities. Disabling validation does not change the math on valid
inputs, but it can surface arithmetic problems later or silently instead of
raising. That is why the toggle is configurable and defaults on; the fast path
is an explicit opt-in, not a hidden default.

Tested and rejected (no gain): pre-made tensor mask (0.6%), `flush_denormal`
(none), `torch_threads=1` (slower; 2 threads faster, equal logits),
`torch.inference_mode()` (~8% but unsafe tensors for the later PPO update).
Deferred: eliminating the double distribution construction (~13% of forward)
needs a custom policy + checkpoint handling.

### 5.2 Rollout profile (explanatory)

One rollout = 1024 transitions, 3 reps, median:

| Sub-step | Native O2 (s) | Share | Dummy (s) |
|---|---:|---:|---:|
| policy forward (total) | 0.0837 | 55.8% | 0.0825 |
| ↳ feature extraction | 0.0445 | 29.7% | 0.0431 |
| ↳ action distribution | 0.0194 | 12.9% | — |
| ↳ sampling | 0.0159 | 10.6% | — |
| ↳ forward other | 0.0040 | 2.6% | — |
| env step | 0.0407 | 27.1% | 0.0641 |
| buffer add | 0.0082 | 5.4% | — |
| `obs_as_tensor` + mask | 0.0079 | 5.3% | — |
| GAE | 0.0009 | 0.6% | — |
| other | 0.0087 | 5.8% | — |
| **rollout total** | **0.1501** | 100% | 0.1729 |

Rollout is forward-bound; native batch already cut env step from 0.0641 (dummy)
to 0.0407. Buffer, tensor conversion and GAE are small.

### 5.3 PPO update profile (explanatory)

One update = 1024 transitions, 4 epochs, 16 minibatches, median of 3: total
0.0981 s; forward 0.0329 (33.5%), backward 0.0343 (35.0%), Adam 0.0172 (17.5%),
clip 0.0041 (4.2%), other 0.0096 (9.8%). Forward + backward ≈ 68%. The update
is ~21 s/run and is **kept as an open candidate**.

### 5.4 Decisions

| Item | Decision | Evidence |
|---|---|---|
| Skip Torch distribution validation (configurable) | **adopted, opt-in** | forward 1.24×, rollout 1.11×, bit-identical; default stays on |
| O3 parallel native workers | **deferred** | native step 0.17 s/eval; rollout is forward-bound |
| Full-run eval batch 32 | **not adopted** | paired 1.036/0.993/0.963 (sign flips); keep 16 |
| Fast maskable distribution (no double construction) | **deferred** | bit-identical, ~13% of forward; needs custom policy |
| `from_native_payload` fast path | **candidate** | ~70% of a ~5–6 s/run summary |
| PPO update optimization | **open candidate** | ~21 s/run; forward+backward ~68% |
| Flip defaults to the opt-in config | **not yet** | O5 repetitions done, but one seed and one host; keep opt-in |

## 6. What was not done

- O3 native workers / `SubprocVecEnv`
- O4 extra caches, prefetch, multi-seed launcher
- O5 default flip (multi-rep evidence exists; more seeds/hosts and a decision
  on the accepted config are still open)
- Fast maskable-distribution / policy-forward optimization (deferred)
- `from_native_payload` fast path; PPO update optimization
- R0–R4 / M1 / M3 rewrites; deferred M2/M4/M5; hyperparameter changes

## 7. How to use O1 and O2

Defaults stay scalar with distribution validation on. Accepted opt-in config:

```toml
[runtime]
backend = "rust"
eval_batch_size = 16
reuse_eval_pool = true
native_batch = true
validate_distributions = false
```

CLI:

```bash
python -m bus_rl.cli train --config configs/experiments/core-threads2.toml \
  --manifest data/generated/base/manifest.json --seed 11 \
  --output runs/<name> --backend rust --torch-threads 2 --eval-limit 100 \
  --native-batch --eval-batch-size 16 --reuse-eval-pool \
  --no-validate-distributions
```

Debug/rollback: drop `--native-batch` / set `eval_batch_size=1` /
`reuse_eval_pool=false`, and use `--validate-distributions` to restore the
argument checks. Unsupported combinations raise `ValueError`.

## 8. Commands and artifacts

| Command | Exit | Artifact |
|---|---:|---|
| `python -m pytest -q` | 0 | 181 passed, 1 skipped |
| `python scripts/build_native.py --output reports/runtime-optimization/native-build-o2.json` | 0 | O2 kernel sha256 `74b3a8fa…` |
| `python scripts/benchmark_runtime.py --mode o1-breakdown …` | 0 | §2.1 |
| `python scripts/benchmark_runtime.py --mode o2-ablation …` | 0 | §3.3 eval + training + summary split |
| `python scripts/benchmark_runtime.py --mode rollout-profile …` | 0 | §5.2 |
| `python scripts/benchmark_runtime.py --mode ppo-update-profile …` | 0 | §5.3 |
| `python scripts/benchmark_runtime.py --mode distribution-validation …` | 0 | §5.1 A/B |
| `python scripts/benchmark_full_workflow.py --repetitions 3` | 0 | §4 paired multi-rep |

Exact argv: `reports/runtime-optimization/commands.json`. Code:
`evaluation/{runner,pool}.py`, `env/native_batch.py`,
`training/{train,callbacks,checkpoint,diagnose}.py`, `backend/native.py`,
`runtime.py`, `config.py`, `cli.py`, `crates/bus-sim-py/src/lib.rs`,
`scripts/{benchmark_runtime,benchmark_full_workflow,build_native}.py`.
