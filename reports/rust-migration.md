# Rust migration report

Status: **R0–R4 accepted; the Rust backend is frozen for the RL L0 baseline.**
Date: 2026-09-12.
Spec: [docs/spec/rust_improve.md](spec/rust_improve.md). Plan: [docs/plan/rust_improve.md](plan/rust_improve.md).

This report records measured evidence only. R0–R3 are accepted. R4.1
correctness, the R4.2 light speed pass, and the R4.3 full L0-protocol workflow
(245,760 transitions x 100 validation days per backend) are recorded; R4 is
**accepted** and task L0.1 in the RL plan is unlocked.

## 0. Why R0 first

The RL plan's **L0 baseline** is gated on an *accepted Rust R4* (parity + >=2x
simulation/learn gates). No Rust crate existed when this work started, so the
first executable stage in the referenced documents was **R0 — freeze the Python
oracle**. R0.1–R0.3 are implemented and accepted here; R1–R3 followed and are
also accepted (sections 4–6). R4 is accepted as well (section 7), so RL L0 is
unlocked.

## 1. R0.1 — Inventory and freeze the reference

The oracle manifest is `reports/rust-migration/oracle-manifest.json`
(machine-checkable, self-hashed). Everything is regenerated with:

```text
python scripts/export_oracle.py build                 # manifest + summary (~1 s, fast default)
python scripts/export_oracle.py build --fixtures      # + verify the 11 fixtures (~8 s)
python scripts/export_oracle.py build --force         # (re)write fixtures from the oracle
python scripts/export_oracle.py verify                # hashes + fixtures + reference (~13 s)
python scripts/export_oracle.py verify --hashes-only  # manifest hashes only (~1 s)
python scripts/export_oracle.py verify --deep         # + regenerate 1,200 scenario seeds (~35 s)
```

Fixtures are recorded in a single ``BusDispatchEnv.step`` pass via the ``on_tick``
trace hook (no mirror pass), and ``verify`` only regenerates the 1,200 scenario
seeds with ``--deep``.

Frozen reference:

| Item | Value |
|---|---|
| Git revision | `1c34457abe78e488df92c240cce8a236dca36af7` (dirty: R0 artifacts being added) |
| Lock | `uv.lock` sha256 `ab9e23d3b3cf0dafac664a0ebaca4bc10d6bee43125597ee6f66b59308756004` |
| Python / torch / numpy | 3.11.14 / 2.14.0+cpu / 2.4.6 |
| Threads / device | 16 / cpu |
| Physical config hash | `fad7a03b35b999d33b672357b439a9bf34899e8bf814bbe98077446ef4a5d4b2` |
| Action schema hash | `ff62ae3cfa7478758a5370e575dbfc3c15729c0ea10a3ccbb2bd7a207cab761a` |
| Observation schema hash | `fa07e9254e5f5474db32935332bea7a40ddbadf7c6538defdedbdc51322397e2` |

Split seeds and counts are fixed in `bus_rl/data/scenario.py`
(`train=1001`, `validation=2001`, `test_id=3001`, `test_ood_burst=4001`,
`test_ood_traffic=5001`; `500/100/200/200/200`). Scenario order is the
`np.random.SeedSequence(split_seed).spawn(count)` order. The committed
`data/generated/base/manifest.json` is git-ignored but fully regenerable; the
manifest records its hash when present and verifies the regenerated split.

Algorithm seeds are fixed at **11/22/33**; the core budget is **245,760
transitions** with validation every **12,288** — recorded for L0 but not run
here.

### Checkpoint inventory (git-ignored historical runs)

| Checkpoint | Steps | Metadata | Role |
|---|---:|---|---|
| `runs/diagnose-after/last.zip` | 2,048 | yes | **selected fixed reference** |
| `runs/diagnose/last.zip` | 2,048 | yes | pre-optimization historical |
| `runs/pilot-11/best.zip`, `last.zip` | 12,288 | yes | pilot historical |
| `runs/core-11/best.zip` | — | no | incomplete (checkpoint without metadata) |

`runs/core-11/best.zip` is classified **incomplete** (no `metadata.json`), so it
is not reusable. Historical reports are preserved unchanged and hashed in the
manifest.

### Contract summary

The full action/observation/cost/terminal contract is embedded in
`oracle-manifest.json` under `contract`. Key points:

- `Discrete(221)`: 1 NOOP + 64 DISPATCH + 64 REASSIGN + 64 SHORT_TURN +
  16 RECALL + 12 SET_HEADWAY; NOOP index 0; headways `{360, 600, 900}` s.
- Observation: `obs_version=2`, all `float32`, keys `stops[4,2,8,7]`,
  `arrival_history[4,2,8,5]`, `forecast[4,2,8]`, `vehicles[16,27]`,
  `routes[4,8]`, `stop_valid[4,2,8]`, `vehicle_valid[16]`, `route_valid[4]`,
  `context[3]`, padding zeroed and masked.
- Terminal: horizon means `terminated=True, truncated=False`; unfinished
  settlement `60*(waiting+onboard)` is charged once at `t=H`.
- Iteration order: waiting cohorts in insertion order; vehicles by id;
  dispatch tie-break by lowest id; boarding FIFO by `(arrival_tick, cohort_id)`.
- Environment RNG selects a scenario on `reset(seed)` only;
  `options.scenario_index` overrides; demand/travel tapes are pre-generated and
  action-independent.

### Source-vs-spec discrepancies

Documented with reproducers and a port decision in the manifest. Summary:

| ID | Area | Impact |
|---|---|---|
| D1 | Action applied before tick 1 completion/arrivals (spec §4.1 orders completion/arrivals first) | Port follows the code |
| D2 | Terminal boarding happens every `TERMINAL_IDLE` tick, not at departure (spec §4.1) | Port follows the code |
| D3 | Departure `pattern` is `str` for extra and `int` for scheduled; `runner` headway filter keeps only extras | Headway metrics are a subset; reproduce for evaluator parity |
| D4 | `tick_s=30` hardcoded for boarding/completion/abandonment tick stamps | Only `tick_s=30` supported |
| D5 | Excessive-wait 900 s and comfort 30 hardcoded; `comfort_capacity` unused | Replicate literals |
| D6 | `routes[:,7]` counts all `DEPOT_IDLE`, not cooldown-aware reserve (spec §8.1) | Port follows the code |
| D7 | `vehicles[:,26] == vehicles[:,23]`; no distinct time-to-terminal | Port follows the code |
| D8 | `arrival_history` bin uses `(t-1-s)//Δ` | Replicate exactly |
| D9 | `observe(forecast=...)` argument is unused; env injects forecast | Port the env contract |

No unresolved discrepancy blocks the port contract: D1/D2/D6/D7/D8/D9 are
resolved by "follow the code"; D3 is reproduced by the evaluator; D4/D5 restrict
supported configs. Each has a `port_decision` in the manifest.

## 2. R0.2 — Golden fixtures and differential harness

Fixtures live in `tests/backend_parity/fixtures/<name>/` (committed,
~1.8 MB total): `input.json` (scenario spec + action trace + tolerances),
`expected.npz` (oracle outputs), `expected_events.json` (departures/actions).

| Fixture | Scenario hash (12) | Control | Steps | Families seen |
|---|---|---|---:|---|
| `zero_m3` | `c3ec70da81cb` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |
| `normal_m1` | `4d82951a29e7` | M1 | 120 | DISPATCH, RECALL, SET_HEADWAY, NOOP |
| `normal_m2` | `4d82951a29e7` | M2 | 120 | DISPATCH, REASSIGN, SET_HEADWAY, NOOP |
| `normal_m3` | `4d82951a29e7` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |
| `normal_m3_random` | `4d82951a29e7` | M3 | 120 | all mission families + NOOP |
| `peak_m3` | `701c0bf7016e` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |
| `burst_m3` | `4e469eca6a2a` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |
| `traffic_m3` | `53f50a04f3fc` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |
| `capacity_m3` | `00a737ba92b9` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |
| `backlog_m3` | `638003a2bad9` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |
| `abandon_m3` | `c9629857b4de` | M3 | 120 | DISPATCH, SET_HEADWAY, SHORT_TURN, NOOP |

Coverage:

- Environments: zero, normal, peak, burst, traffic demand; multiple seeds.
- Stages: M1/M2/M3 (plus a random-valid controller for broad states).
- Every action family is exercised across the catalog; invalid actions,
  donor/cooldown guards, partial boarding/capacity, abandonment, backlog
  terminal settlement, boarded/completed channel semantics, and
  finished-passengers-in-history are asserted in
  `tests/backend_parity/test_kernel_coverage.py`.
- Records: per-boundary observation/mask/reward/cost components/state plus
  per-tick vehicle state, counters and events. The tick pass mirrors
  `BusDispatchEnv.step` and is cross-checked against it on every fixture.
- Action-driven replay: `replay_fixture` regenerates the trace from
  `input.json`; `test_differential.py` also replays the explicit action list.

Tolerances (from the spec): counters/status/IDs/masks exact; observation
`rtol=atol=1e-6`; cost/reward `rtol=atol=1e-9`. A deliberately corrupted
fixture produces a first-divergence report naming the field, index, expected and
actual values (`test_corrupted_fixture_reports_first_divergence`).

### Fixed reference checkpoint

`tests/backend_parity/reference/diagnose-after/` is a committed copy of the
historical `runs/diagnose-after/last.zip` + `metadata.json`
(checkpoint sha256 `98f99ebf151cdf6dde914df95ea780c25f4856e6a36f90e30ccd7a9f5906501e`).
The exact config (`configs/experiments/core.toml`), 10-day validation split and
seed are recoverable, so the historical reference is reused rather than
replaced.

Paired evaluation reproduces the historical regression anchor exactly:

```text
python scripts/export_oracle.py verify
# fixed-checkpoint eval: mean total_cost = 15471.25 (historical 15471.25)
```

The scenarios are regenerated with
`bus_rl.data.scenario.generate_manifest('validation', 100)[:10]`, which is
byte-identical to the git-ignored `data/generated/base` validation split.

## 3. R0.3 — Unprofiled Python benchmark

Entry point: `scripts/benchmark_python.py` (cProfile never imported;
`TIMERS.enabled=False`; conservation checks off; warm-up separated; setup
reported separately). Raw data: `reports/rust-migration/python-benchmark.json`.

Protocol: 5 measured repetitions, 1 warm-up, CPU, 16 torch threads, same
machine, scenarios regenerated deterministically.

### Simulation (fixed action traces, steady-state only)

| Workload | Median wall | Decisions/s | Ticks/s |
|---|---:|---:|---:|
| `zero_m3` | 0.0446 s | 2690.7 | 10762.9 |
| `normal_m3` | 0.0900 s | 1333.6 | 5334.2 |
| `peak_m3` | 0.0914 s | 1313.2 | 5252.8 |
| `burst_m3` | 0.0963 s | 1245.9 | 4983.5 |
| `traffic_m3` | 0.0912 s | 1315.8 | 5263.2 |
| **total (600 decisions)** | **0.4135 s** | — | — |

### Isolated learn (core hyperparameters, periodic eval disabled)

| Metric | Value |
|---|---:|
| Transitions per rep | 12,288 (actual, all 5 reps) |
| Median learn wall | **34.047 s** |
| Throughput | **360.9 transitions/s** |
| Median setup (env+model) | 0.024 s |
| Config | 4 envs, n_steps=256, batch_size=256, n_epochs=4, seed 11 |

### Fixed-checkpoint evaluation

| Metric | Value |
|---|---:|
| Days | 10 |
| Median wall (5 reps) | **3.529 s** |
| Mean `total_cost` | **15471.25** (matches historical) |

Peak RSS during the whole benchmark: **457 MB**.

### Note on the old 266 decisions/s figure

`reports/training-diagnosis-v1.1-python-improve.md` measured 266 decisions/s **with cProfile
and nested timers enabled**. That number is diagnostic only and is explicitly
not the production reference. The unprofiled numbers above are the R0.3
reference that R4 will remeasure interleaved with the Rust backend.

## 4. R1 — native kernel

Layout (spec section 3.1): `crates/bus-sim/` (rlib `bus_sim_core`: domain,
engine, passengers, vehicles, dispatcher, travel, guards, costs, snapshots) and
`crates/bus-sim-py/` (cdylib `bus_sim`: thin PyO3 debug bridge). Build/install:
`python scripts/build_native.py` (documented in `crates/README.md`); the Python
CLI is unchanged.

What was ported (field-for-field from the oracle):

- Enums, IDs, `SimConfig`, network/fleet, cohort storage, cached `Vehicle.load`
  and incremental totals (`i64` with conservation assertions). Immutable tapes
  are copied once into the scenario; `reset` only rebuilds episode state and
  never reads files or re-packs.
- Passenger lifecycle: arrivals, eligibility, capacity/partial boarding with
  lineage-preserving splits, single first-denial, alighting/completion,
  abandonment boundary, and the waiting hot list vs finished archive.
- Vehicle lifecycle: phase completion, travel rounding from the packed traffic
  field, dwell/layover, short-turn turnpoint and terminal-idle behavior.
- Actions and guards: the frozen 221-slot table and the donor/cooldown/floor
  guards; invalid actions are rejected before any mutation.
- Tick order: action before the interval, then per tick complete-phase ->
  arrivals -> abandon -> terminal boarding -> dispatch -> integrate costs ->
  decay timers -> advance clock -> conservation -> event log -> one-time
  settlement. Ticks per interval come from config.
- Costs/reward: every raw component, mission changes, one-time terminal
  unfinished settlement and the weighted `-cost/n_ref` reward.

### R1 parity result

All 11 R0 golden fixtures were replayed through `bus_sim.Kernel` and compared
to the oracle with the frozen tolerances. **Zero divergences** on:

- state counters, vehicle arrays, full cohort table (kind/bus/lineage/status),
  headway/dispatch clocks and `terminal_settled` at every boundary;
- every per-tick vehicle/counter/event record and every per-tick cost;
- every per-step cost component, reward and termination flag;
- departure events and accepted mission/headway events (including the D3
  `str` vs `int` pattern distinction).

Tests: `cargo test -p bus-sim` (12 native unit tests: reset, invalid
dimensions/IDs, capacity/conservation, config-driven ticks, rejected invalid
action, observation shapes/windows/future-tape independence) and
`python -m pytest tests/backend_parity/test_rust_kernel.py` (26 tests: 11
fixtures x 2 surfaces plus guard/settlement/reset cases). Native build
provenance (toolchain, revision, `.so` hash, import check) is in
`reports/rust-migration/native-build.json`.

## 5. R2 — observations and masks

`crates/bus-sim/src/observation.rs` ports `bus_rl.env.observation.observe`
channel for channel:

- queue count, mean age (normalized), max-age and excessive-wait channels;
- five-lag `arrival_history` plus the last-interval arrival channel;
- the boarded channel restricted to currently ONBOARD cohorts whose boarding
  time lies in the window, and the completion channel;
- fleet/route encodings, validity tensors, node encoding, `context`, and the
  zero forecast placeholder.

Incrementality (R2.1): waiting cohorts come from the hot list, onboard cohorts
from `Vehicle.passengers`, and finished cohorts from a bounded
`recent_finished` ring (`WorldState::recent_finished`) that retains entries while
either their arrival is inside the history window or their completion is inside
the last interval. The full `finished` archive is never scanned by observation,
yet the waiting -> onboard -> recent-finished order preserves the oracle's
`iter_cohorts` accumulation order (float32).

Mask cache (R2.2): `Kernel` computes the guard mask once at `reset` and again
after every `step`/`debug_step`, and `action_mask()` returns a fresh copy of the
cached buffer. `Kernel.mask_computations` proves repeated reads do not
recompute, and `debug_validate_mask()` re-derives the mask from the current
state to detect a stale cache. A masked, out-of-range or stale action is
rejected before any mutation.

### R2 parity result

All 11 fixtures replay through the native kernel with:

- every observation channel equal at `rtol=atol=1e-6` at reset and after every
  step (shapes, `float32` dtype and validity tensors included);
- all 221 mask bits exactly equal at reset and after every step;
- the mask cache validating after each step and stable across repeated reads.

Tests: 15 in `tests/backend_parity/test_rust_observation.py` (11 fixture
differentials plus cache/copy/reset/rejection cases). Native window-boundary,
future-tape-independence and recent-finished eviction are covered by the core
unit tests. Native unit tests in total: 12.

## 6. R3 — backend integration

R3.1 (Gym contract): the PyO3 bridge exposes `Kernel.reset_contract()`
(`{obs, mask}`) and `Kernel.step_contract(action)`
(`{obs, reward, terminated, truncated, mask, costs}`), rejecting masked or
out-of-range actions before any mutation. `bus_rl.env.native_bus_dispatch`
wraps it in a `gym.Env` with the same spaces, seeding, scenario selection,
`terminated=True/truncated=False` horizon behavior and `StepCosts` conversion as
the oracle. Every kernel owns its packed tapes; no Python `WorldState` is built
per training step, and returned obs/mask arrays are fresh copies.

R3.2 (evaluator/traces/forecast): `bus_rl.evaluation.summary` defines
`SummaryInputs`/`CohortView` and the shared metric math, with adapters for the
Python `WorldState` and for `Kernel.episode_summary_inputs()`. `evaluation/runner`
now consumes `env.summary_inputs()` and `env.trace_snapshot()`, so cohort/lineage
data, departures, actions, totals and vehicle snapshots flow through one
interface. Statistics and plots stay in Python; detailed data is exported once
per episode (or when tracing is enabled). Forecast remains Python: the native
wrapper predicts from the current obs and time and sets the context flag exactly
like the oracle.

R3.3 (selection/lineage): `runtime.backend = "python" | "rust"` is read from
config, overridable with `--backend` on train/diagnose/evaluate/baseline, and
honored by the env factory used by training, the validation callback, diagnose,
evaluation and baselines. Requesting `rust` without the extension raises a clear
`RuntimeError`; Python fallback requires explicit selection. Checkpoint metadata
records `backend`, `native_build` (crate version, `.so` hash, rustc, git sha) and
`backend_parity_verified`; cross-backend loading is allowed only when the
physical/action/obs hashes match and parity is marked verified.

### R3 verification result

21 tests in `tests/backend_parity/test_rust_integration.py`:

- native/py env step parity for six fixtures (obs tolerance `1e-6`, mask
exact, reward/costs `1e-9`), Gym contract and masked-action rejection, output
ownership across steps/resets, interleaved envs, `DummyVecEnv` auto-reset;
- evaluator + trace parity for fixed/threshold/proportional/random controllers,
forecast on/off, zero-demand censoring, and the normal plotting pipeline;
- backend default/override, missing-extension failure, metadata provenance and
cross-backend checkpoint rules;
- a CLI end-to-end run: `train --backend rust` (32 transitions, checkpoint +
metadata) and `evaluate` with both backends producing identical
`total_cost` (1e-9).

## 7. R4 — acceptance, full workflow, and optimization profile

R4.1 correctness is covered by `tests/backend_parity/test_rust_acceptance.py`:
a 2048-transition MaskablePPO smoke on both backends (finite obs/reward/loss,
checkpoint save/load) and fixed-checkpoint per-day parity over 10 validation
days (per-day cost and full trace identical).

R4.2 light pass, measured interleaved on one machine (24 CPUs, **2 torch
threads for both backends**, CPU-only) with profiler and TIMERS disabled,
conservation off, 5 repetitions and 1 warm-up per backend. "Simulation" is the
median of per-repetition totals (sum over the 5 workloads), not a sum of
per-workload medians. Raw times are in
`reports/rust-migration/speed-acceptance.json`.

| Workload | Python median | Rust median | Speedup |
|---|---:|---:|---:|
| Simulation, 5 workloads / 600 decisions | 0.357 s | 0.0142 s | **25.11x** |
| Isolated learn, 12,288 transitions | 13.159 s (934 t/s) | 3.153 s (3897 t/s) | **4.17x** |
| Fixed-checkpoint eval, 10 days | 1.418 s | 0.412 s | **3.44x** |

- Per-workload simulation speedups range 17.4x (zero demand) to 26.6x (burst).
- Isolated learn matches the core protocol (4 envs, n_steps 256, batch 256,
  n_epochs 4, seed 11, no periodic eval); setup cost is reported separately.
- Fixed-checkpoint evaluation reproduces mean `total_cost` **15471.25** on both
  backends with identical per-day costs and traces.
- Learn is now stable across repetitions (Python 13.11–13.31 s, Rust
  3.13–3.16 s), so the 4.17x ratio is not noise.
- Peak RSS **491 MB** (from the earlier 16-thread run) is the peak of the process
  running **both** backends; it is not a per-backend figure.

### R4.3 — full workflow benchmark (acceptance)

One full L0-protocol seed per backend with identical schedules:
`configs/experiments/core-threads2.toml` (4 envs, n_steps 256, batch 256,
_epochs 4, seed 11, **2 torch threads**), budget **245,760 transitions**,
validation every **12,288** transitions (20 evaluations) on **100 validation
days**. Python ran first so Rust was measured on the warmer machine; the host
slowed ~16% over the session (an earlier Python seed took 523 s), so this table
is the same-session pair and the earlier pair measured 3.60x. Raw evidence: `reports/rust-migration/full-workflow.json`, with logs and
checkpoints under `runs/rust-migration/full-workflow/`.

| Metric | Python | Rust | Ratio |
|---|---:|---:|---:|
| Learn + validation (`wall_time_s`) | 595.35 s | 154.23 s | **3.86x** |
| Total wall incl. setup (`/usr/bin/time`) | 597.99 s | 156.37 s | **3.82x** |
| Setup (reported separately) | 2.64 s | 2.14 s | — |
| Transitions / episodes | 245,760 / 2,048 | 245,760 / 2,048 | — |
| Evaluations x validation days | 20 x 100 | 20 x 100 | — |
| Best validation cost | 13199.1325 | 13199.1325 | equal |
| Peak RSS | 479 MB | 445 MB | **0.93x** |

**Parity.** All 20 validation costs are bit-identical (max abs diff 0.0), and
the trained artifacts are byte-identical: `policy.pth`,
`policy.optimizer.pth` and `pytorch_variables.pth` match exactly for both
`best.zip` and `last.zip` (state-dict max abs diff 0.0). The checkpoint `.zip`
sha256 values differ only because `policy_class`/`rollout_buffer_class`
serialize as function object addresses and `start_time` is a wall-clock value;
`_last_obs`, `_last_original_obs`, `_last_episode_starts` and `num_timesteps`
are equal. This closes the R4.3 parity question.

**Validation dominates the workflow.** Rerunning the same seed with
`--eval-limit 10` isolates the cost: Python 280.31 s, Rust 76.21 s. The extra 90
validation days x 20 evaluations cost Python 240.8 s (46% of its total) and Rust
66.8 s (47%), so evaluation, not simulation, is now the largest remaining Python
cost. R4.3 is accepted: the full Rust workflow is **3.82x** faster, uses **less
memory than Python**, and every preceding gate passes.

**Memory.** Rust peak RSS is now **below** Python's: **445 MB vs 479 MB (0.93x)**.
Four storage issues were found and fixed; each was verified byte-identical:

1. **Eager kernels.** `NativeBusDispatchEnv` built one `Kernel` per scenario, and
   each kernel keeps ~243 KB of per-episode scratch after its first episode
   (bounded, reused later). With 500 train scenarios x 4 envs + 100 validation
   scenarios that was ~500 MB of idle kernels. Fixed by creating the kernel
   lazily in `reset()`: the 4-env DummyVecEnv retained heap dropped from 232 MB
   to 0.5 MB (430x) with no throughput loss.
2. **Dense arrival tapes in the store.** The native store kept the oracle
   arrival tape `(480, 3, 2, 6, 6)` = **405 KB/scenario**, but it is **98.5%
   zeros** (1,509 nonzero of 103,680). Native `SparseArrivals` now stores only
   nonzero cells in a CSR-by-tick layout that preserves the exact scan order
   `add_arrivals` uses: the store for all 600 scenarios fell **245.6 MB ->
   30.1 MB**. An interleaved A/B of the two `.so` files at 61,440 steps /
   100-day evals measured sparse 40.0/38.5 s vs dense 40.0/39.1 s, so sparse is
   not slower.
3. **`int32` arrival tapes.** The dense tape was `int32` although the largest
   count is **5** (OOD `flood` x20 = 100, `capacity` = 45). Both backends now use
   `int8`: the 600 scenarios fell **249 MB -> 67 MB** (tape 405 -> 101 KB each).
   `scenario_digest` normalizes to `int32` before hashing, so `scenario_hash` and
   every committed fixture are unchanged. Helpers validate the int8 range.
4. **Rust-run scenario ownership.** A Rust run loaded all 600 scenarios (with the
   dense tapes) into Python only to hand them to the native store. `load_split`
   now supports metadata-only scenarios (`with_tapes=False`) and the CLI uses
   them for Rust runs without forecast; `NativeScenarioStore` reads each
   `tapes.npz` once, verifies `scenario_digest`, and keeps only the sparse
   packing. Python never holds the 600 tapes in a Rust run.

Rust peak RSS is now **below** Python's (445 MB vs 479 MB, 0.93x): the store is
30 MB and both backends share the small metadata. Python never paid for issues
1-2 because it builds `WorldState` on demand; issues 3-4 help both or Rust only.
Overall Rust peak RSS went **1236 MB -> 445 MB** across this work.

### Torch threads (chosen configuration)

`scripts/tune_runtime.py` -> `reports/rust-migration/runtime-tuning.json`,
same-condition comparison, median isolated learn of 12,288 transitions
(2 repetitions + 1 warm-up) and fixed-checkpoint eval of 10 days:

| torch threads | Python learn | Rust learn | Python eval | Rust eval |
|---:|---:|---:|---:|---:|
| **2** | **13.15 s** | **3.15 s** | **1.51 s** | **0.41 s** |
| 4 | 14.32 s | 3.19 s | 1.55 s | 0.44 s |
| 16 (old default) | 28.55 s | 8.03 s | 3.67 s | 1.32 s |

2 threads wins for both backends; the environment default of 16 was ~2.2x
slower on Python learn and ~2.5x slower on Rust learn. `torch_threads` is now an
`AlgorithmConfig` field, applied in train/diagnose/evaluate via
`bus_rl.runtime.apply_torch_threads`, settable with `--torch-threads` or
`configs/experiments/core-threads2.toml`, and recorded in metadata as
`torch_threads` / `torch_threads_actual`.

### Observation validation (opt-in)

`runtime.validate_observation` defaults to True. When False the wrappers skip
only `validate_observation`; forecast still runs in Python and simulator
semantics are unchanged. End-to-end learn at 2 threads: Python 13.086 ->
12.988 s (-0.7%), Rust 3.126 -> 2.974 s (-4.9%). Env-level step cost: Python
0.0917 -> 0.0817 s, Rust 0.0032 -> 0.0021 s. Validation is cheap in absolute
terms, so it stays on by default and off is only for measured experiments.

### Where the remaining time goes (Rust, 16-thread profile)

Short Rust profile (4096 transitions, 4 envs, 1024/rollout; TIMERS on, so
absolute times are inflated but shares are informative) in
`reports/rust-migration/profile-native.json`:

- Learn is **PPO-update bound**: rollout collect 0.44 s vs PPO update 0.60 s
  (~57% of learn).
- Inside a rollout, `env.step` is 0.37 s (84%); inference + buffer overhead is
  only 0.06 s. Within `env.step`, Python-side observation post-processing
  (`np.asarray` + `validate_observation`) is 0.19 s, about half of the env step.
- Eval is **inference bound**: over 10 days, `env.step` 0.135 s vs PPO inference
  1.07 s (82%); metric summarization 0.064 s.

### Optimizations applied in this pass

- **Shared native scenario store**: `crates/bus-sim-py` `ScenarioStore` +
  `Kernel.from_store` pack each scenario once per process and share immutable
  tapes via `Arc`; `src/bus_rl/backend/native.py` caches stores by
  `(scenario_hash, M-flags)`. The arrival tape is stored sparse
  (`SparseArrivals`, nonzero cells only in CSR-by-tick order) because the dense
  oracle tape is 98.5% zeros: the store for all 600 scenarios measures 30.1 MB.
- **int8 arrival tapes**: the dense tape's largest count is 5, so both backends
  use `int8` (tape 405 -> 101 KB/scenario; 600 scenarios 249 -> 67 MB) with
  `scenario_digest` normalizing to int32 so `scenario_hash` is unchanged.
- **Lazy kernels**: kernels are built in `reset()` for the selected scenario
  rather than one per scenario up front, because each kernel keeps ~243 KB of
  episode scratch; see the R4.3 memory note above.
- **Wrapper mask reuse**: `reset_contract`/`step_contract` already return the
  next mask; `NativeBusDispatchEnv` keeps it and `action_masks()` serves a copy,
  dropping the extra native `action_mask()` FFI call per step (verified by
  `mask_computations` in the integration tests).

Gate status: at the chosen 2 threads (same condition for both backends)
simulation is 25.1x, isolated learn 4.17x (above the 4x stretch) and
fixed-checkpoint eval 3.44x; all far above the 2x gate. The R4.3 full workflow
is 3.82x with bit-identical training and **lower peak RSS than Python**, so
**R4 is accepted** and the native revision `aedc4faa4c43` is frozen for RL.
Optimization should now target PPO
update and evaluation inference, not the simulation core; batched multi-scenario
inference is the next candidate for eval.

## 8. Evidence

| Command | Exit | Artifacts |
|---|---:|---|
| `python scripts/export_oracle.py build` | 0 | manifest, 11 fixtures, summary |
| `python scripts/export_oracle.py verify` | 0 | all hashes + 11 replays + reference 15471.25 |
| `python scripts/build_native.py` | 0 | native build + `native-build.json` |
| `cargo test -p bus-sim` | 0 | 12 native unit tests |
| `python -m pytest tests/backend_parity/test_rust_kernel.py -q` | 0 | 26 kernel parity tests |
| `python -m pytest tests/backend_parity/test_rust_observation.py -q` | 0 | 15 observation/mask parity tests |
| `python -m pytest tests/backend_parity/test_rust_integration.py -q` | 0 | 24 integration tests |
| `python -m pytest tests/backend_parity/test_rust_acceptance.py -q` | 0 | 4 R4.1 smoke/parity tests |
| `python scripts/benchmark_backends.py --repetitions 5 --transitions 12288` | 0 | `speed-acceptance.json` (light R4.2) |
| `python scripts/profile_native_training.py` | 0 | `profile-native.json` (segments/threads/eval/memory) |
| `python scripts/tune_runtime.py` | 0 | `runtime-tuning.json` (threads + validation) |
| `python -m pytest -q` | 0 | 149 passed (1 skipped deep)
| `python -m bus_rl.cli train ... --backend python --eval-limit 100` | 0 | `full-workflow.json` (Python full seed) |
| `python -m bus_rl.cli train ... --backend rust --eval-limit 100` | 0 | `full-workflow.json` (Rust full seed) |
| `python scripts/benchmark_python.py --repetitions 5 --transitions 12288` | 0 | `python-benchmark.json` |

Artifacts:

- `reports/rust-migration/oracle-manifest.json` — frozen inventory + contract.
- `reports/rust-migration/python-benchmark.json` — raw repetitions.
- `reports/rust-migration/speed-acceptance.json` — interleaved light R4.2 raw times.
- `reports/rust-migration/profile-native.json` — Rust training segments, torch
  thread sweep, eval segments and shared-store memory.
- `reports/rust-migration/runtime-tuning.json` — same-condition thread matrix and
  observation-validation cost for both backends.
- `reports/rust-migration/full-workflow.json` — R4.3 full-seed wall times, parity,
  validation-cost ablation and memory breakdown.
- `reports/rust-migration/full-workflow-memory.json` — native `ScenarioStore`
  memory measurement (600 scenarios).
- `runs/rust-migration/full-workflow/{python,rust}/` — git-ignored logs,
  `metadata.json`, `evaluations.csv` and checkpoints for both full seeds.
- `reports/rust-migration/native-build.json` — native toolchain/revision/hash.
- `crates/bus-sim/`, `crates/bus-sim-py/` — native kernel + PyO3 bridge.
- `crates/bus-sim/src/observation.rs` — incremental observation + ring.
- `src/bus_rl/backend/native.py` — scenario packing + native provenance.
- `src/bus_rl/env/native_bus_dispatch.py` — native Gym wrapper.
- `src/bus_rl/env/factory.py` — `runtime.backend` selection.
- `src/bus_rl/evaluation/summary.py` — shared summary/trace inputs + metrics.
- `tests/backend_parity/fixtures/` — golden fixtures (retired after acceptance).
- `tests/backend_parity/reference/` — reference checkpoint (now read from the
  git-ignored `runs/diagnose-after/`).

## 9. Limitations and next steps

- R0–R4 are accepted. R4.2 light gates pass (simulation ~24.9x, isolated
  learn ~3.9–4.17x) and R4.3 full workflow is 3.82x with bit-identical training
  (section 7). The remaining opportunity is evaluation, which is ~46% of the
  Python full-seed wall; batched multi-scenario inference is the next candidate.
- Rust peak RSS is now **0.93x** Python's (445 MB vs 479 MB) for a full seed —
  below Python. Four storage bugs were found and fixed: eager per-scenario
  kernels (~243 KB each), a dense ~405 KB arrival tape that is 98.5% zeros (now
  sparse), int32 tapes whose max value is 5 (now int8), and a Rust run holding all
  600 Python tapes (now metadata-only, store reads from disk). Rust peak went
  1236 -> 445 MB. See [docs/spec/memory_optimize.md](spec/memory_optimize.md).
- The manifest records `git.sha`/`git.dirty` for the repo HEAD at build time and
  the reference checkpoint's origin revision separately; the frozen oracle
  contract is the physical/action/observation hashes and the fixture hashes, not
  the git revision. The reference checkpoint was created at `4f041433`.
  The full-workflow metadata also records `git_dirty=true`; the only uncommitted
  file at run time was the regenerated `native-build.json` (same library hash).
- The git-ignored `runs/` and `data/generated/` inventories are documented but
  optional for verification; the committed reference checkpoint and
  regeneration commands make R0 reproducible from a fresh checkout.
- `src/bus_sim.so` and `target/` are git-ignored; rebuild with
  `python scripts/build_native.py`. The core unit tests run without Python; the
  parity tests skip when the extension is absent.
- `recent_finished` duplicates recently finished cohorts by design; its size is
  bounded by the arrival/completion windows, not by episode length.
- R4 remeasured Python on the same machine as Rust (both full seeds); the R0.3
  numbers remain the historical Python baseline for the >=2x gates.

## 10. Stage checklist

- [x] R0: immutable oracle, golden coverage, and unprofiled reference accepted.
- [x] R1: native domain/lifecycle/actions/ticks/costs at oracle parity.
- [x] R2: observation/history and mask parity accepted.
- [x] R3: wrapper/evaluator/forecast/backend/provenance accepted.
- [x] R4: correctness, 2x simulation and learn gates, and 3.82x full workflow
  (bit-identical training, lower RSS than Python) accepted.
- [x] Historical reports preserved; measured results are separated from projections.
- [x] Frozen native revision `aedc4faa4c43` recorded; task L0.1 in the RL plan is unlocked.
