# Rust Backend Implementation Plan

Date: 2026-09-12. Status: **R0–R4 accepted; Rust backend frozen for RL L0.**

## 1. Scope and execution contract

Implement the [Rust specification](../spec/rust_improve.md) before starting the [RL improvement plan](improve_RL.md). Preserve the [base specification](../spec/spec_v1.0.md). The [original implementation plan](plan_v1.0.md) remains historical context and does not establish completion of new work.

Sequence: **R0 Oracle -> R1 Kernel -> R2 Observation/masks -> R3 Integration -> R4 Acceptance -> RL L0-L3**.

The specification defines behavior and acceptance thresholds; this plan defines executable work packages. If implementation exposes a contradiction, record it with a reproducer and resolve the contract before accepting dependent work. Do not silently change physics, reward, observation semantics, or acceptance thresholds.

Each task starts unchecked. Mark it complete only after its acceptance criteria pass and evidence is recorded. Record task ID, source revision, exact command, exit status, artifact paths/hashes, and any remaining limitations in `reports/rust-migration.md`. Failed gates remain open. Proposed files and interfaces below are not claims that those files or commands already exist.

## 2. Task dependencies

| Task | Stage | Depends on | Deliverable |
|---|---|---|---|
| R0.1 | Oracle | None | Frozen contract and provenance manifest |
| R0.2 | Oracle | R0.1 | Golden fixtures and replay harness |
| R0.3 | Oracle | R0.1, R0.2 | Unprofiled Python reference measurements |
| R1.1 | Kernel | R0 accepted | Buildable native domain and scenario loader |
| R1.2 | Kernel | R1.1 | Passenger and vehicle lifecycle |
| R1.3 | Kernel | R1.2 | Actions, ticks, travel, and costs |
| R2.1 | Observation | R1.3 | Equivalent incremental observation |
| R2.2 | Masks | R1.3 | Validated mask cache |
| R3.1 | Integration | R2.1, R2.2 | PyO3 and Gym wrapper |
| R3.2 | Integration | R3.1 | Evaluator, traces, and forecast compatibility |
| R3.3 | Integration | R3.2 | Backend selection and checkpoint provenance |
| R4.1 | Acceptance | R3.3 | Correctness and training verification |
| R4.2 | Acceptance | R4.1, R0.3 | Unprofiled speed acceptance |
| R4.3 | Handoff | R4.2 | Full workflow benchmark and frozen backend |

## 3. R0: freeze the Python oracle

### R0.1 - Inventory and freeze the reference

- [x] Record the Python after revision, dirty state, dependency lock, config hashes, manifests, scenario order, seeds, and machine/thread settings.
- [x] Inventory available checkpoints and reports without overwriting historical artifacts.
- [x] Document action IDs, observation keys/shapes/dtypes/scales, cost components, terminal behavior, RNG selection, and deterministic iteration order.
- [x] Record any source/base-spec discrepancy, especially time thresholds, rounding, boarded-channel semantics, and terminal settlement.

**Implementation surface:** `domain.py`, `env/`, `sim/`, `control/`, `rewards/`, `training/checkpoint.py`; proposed `reports/rust-migration/oracle-manifest.json`.

**Verification:** resolve every manifest reference and hash; reset identical scenarios repeatedly and compare initial state/observation/mask; review discrepancies against source and base spec.

**Acceptance:** the exact reference can be reconstructed, no unresolved discrepancy affects the port contract, and reference artifacts are immutable. Python performance improvements are not a prerequisite.

### R0.2 - Build golden fixtures and a differential harness

- [x] Export per-tick state/counters/events and per-control-step action, observation, mask, reward, and cost components.
- [x] Store input tapes, action traces, scenario/config hashes, and expected outputs separately from native outputs.
- [x] Cover zero, normal, peak, burst, and traffic cases; multiple seeds; M1/M2/M3; every action family; invalid actions; partial boarding/splits; capacity; abandonment; donor/cooldown guards; short turns; and terminal settlement.
- [x] Add fixtures for history boundaries, immediate completion after boarding, repeated splits, and finished passengers remaining in recent arrival history.
- [x] Select a fixed checkpoint for paired evaluation. Reuse the historical 15471.25 reference only if its exact checkpoint/config/days are recoverable; otherwise create and label a new Python reference.

**Implementation surface:** proposed `tests/backend_parity/`, fixture exporter under `scripts/`, raw `runs/rust-migration/oracle/`.

**Verification:** replay exported actions on Python and reproduce the fixture; deliberately alter a fixture value to demonstrate useful first-divergence reporting.

**Acceptance:** all coverage categories have named fixtures; counters/status/IDs/masks compare exactly; observations use `rtol=1e-6, atol=1e-6`, costs/rewards use `rtol=1e-9, atol=1e-9`. Reports identify scenario, tick/step, field, expected value, and actual value. Equal mean reward alone never passes parity.

### R0.3 - Establish an unprofiled Python benchmark

- [x] Build a benchmark entry point for fixed-action simulation, isolated learn, and fixed-checkpoint evaluation.
- [x] Disable cProfile and TIMERS, use matching trace/conservation settings, and separate setup from steady-state execution.
- [x] Warm up separately; collect at least five repetitions, raw wall times, actual transitions, thread settings, and peak RSS.
- [x] Use at least 12,288 transitions for isolated learn with the core hyperparameters, periodic evaluation disabled, and a reproducible initial model.

**Outputs:** proposed `reports/rust-migration/python-benchmark.json` and exact reproduction commands in the report.

**Verification:** Audit benchmark config and profiler flags; recompute median/min/max from raw repetitions and verify actual transition counts.

**Acceptance:** raw repetitions and median/min/max are available; no comparison uses the profiled 266 decisions/s as the production reference. R4 will remeasure Python interleaved with Rust on the same machine.

**R0 gate:** R0.1-R0.3 accepted. Freeze the oracle before native implementation.

**Evidence (2026-09-12, accepted):** revision `1c34457`; commands and raw
results in [reports/rust-migration.md](../reports/rust-migration.md) and
`reports/rust-migration/`. `python scripts/export_oracle.py verify` reproduces
11 golden fixtures and the fixed-checkpoint mean `15471.25`;
`python scripts/benchmark_python.py` records the unprofiled reference
(simulation total 0.4135 s / 600 decisions; isolated learn 34.05 s / 12,288
transitions). Freeze the oracle before native implementation.

## 4. R1: implement the native kernel

### R1.1 - Native build, domain, and scenario ownership

- [x] Create `crates/bus-sim/` and the chosen PyO3 packaging layout; document the release build/install procedure while retaining the existing Python CLI.
- [x] Implement enums, IDs, vehicles, cohort storage, cached loads, and incremental totals with checked bounds.
- [x] Pack immutable network/tapes/traffic/config once; define native ownership or a valid retained Python owner.
- [x] Implement reset and debug snapshots with deterministic vehicle/cohort ordering.

**Verification:** native unit tests for reset, invalid dimensions/IDs, capacity bounds, ownership/lifetime, and initial snapshots against R0 fixtures.

**Acceptance:** release build succeeds from documented commands; initial state matches the oracle; reset does not read files or repack all scenarios; no dangling borrowed buffers or overflow-prone unchecked counters.

### R1.2 - Passenger and vehicle lifecycle

- [x] Implement arrivals, eligibility, partial boarding, lineage-preserving splits, alighting/completion, abandonment, and load/counter transitions.
- [x] Implement phase completion, travel duration/rounding, and terminal-idle behavior.
- [x] Retain finished records needed for metrics; remove them from active scans without losing historical information.

**Verification:** targeted native tests and differential fixtures for full/partial buses, repeated splits, expiration boundaries, same-tick transitions, and zero demand. Enable conservation in tests.

**Acceptance:** passenger mass, lineage, vehicle loads, statuses, event counts, and transition timing match Python at every tested tick. No cohort is duplicated or lost.

### R1.3 - Actions, tick order, and costs

- [x] Implement NOOP, DISPATCH, RECALL, REASSIGN, SHORT_TURN, and SET_HEADWAY with unchanged mappings and guards.
- [x] Preserve action-before-interval and the complete tick order in Rust spec section 3.3.
- [x] Implement raw cost integration, weighted reward, mission changes, and one-time terminal unfinished settlement.
- [x] Derive ticks per interval from config; preserve reference semantics for supported configs.

**Verification:** replay every R0 action trace; compare per-tick events/state and every cost component; test invalid action rejection before mutation.

**Acceptance:** all kernel fixtures pass with the specified tolerances, terminal cost is charged exactly once, and no physics/reward change is bundled into migration.

**R1 gate:** R1.1-R1.3 accepted; kernel behavior is equivalent before observation optimization.

**Evidence (2026-09-12, accepted):** crate layout `crates/bus-sim`
(`bus_sim_core`) + `crates/bus-sim-py` (`bus_sim`); build via
`python scripts/build_native.py`. All 11 R0 fixtures replay through the native
kernel with zero divergences on state/counters/IDs/lineage, every cost
component, per-tick state, departure/action events and terminal settlement;
`cargo test -p bus-sim` (7 tests) and
`python -m pytest tests/backend_parity/test_rust_kernel.py` (26 tests) pass.
Details in [reports/rust-migration.md](../reports/rust-migration.md) and
`reports/rust-migration/native-build.json`. Observation/mask parity remains R2.

## 5. R2: observations and masks

### R2.1 - Incremental observation with history parity

- [x] Implement current queue/age statistics and bounded history/completion statistics.
- [x] Preserve recent arrivals for completed/abandoned passengers without scanning the full finished list.
- [x] Preserve the current boarded-channel rule: currently ONBOARD cohorts whose boarding time lies in the window, not all boarding events.
- [x] Preserve all shapes, float32 outputs, normalization, validity tensors, context, and forecast placeholders.

**Verification:** differential tests for every observation channel at reset and every step; exact window boundaries; repeated splits; board-then-complete; abandonment; final episode state; tapes identical in the past but different in the future.

**Acceptance:** all observation fixtures meet tolerance and no feature leaks future data. A simple waiting-plus-onboard scan that drops finished-derived information fails acceptance.

### R2.2 - Cache masks safely

- [x] Compute the initial mask at reset, validate against the current-state mask, and compute the next-state mask after step.
- [x] Remove redundant computation across collector/step/dispatcher where safe; retain guards for independently callable APIs.
- [x] Protect cache ownership and invalidate after any supported debug mutation.

**Verification:** compare all 221 bits across fixtures; repeatedly call `action_masks()` without mutation; test masked/out-of-range actions, reset, cooldown boundaries, and caller attempts to modify returned arrays.

**Acceptance:** masks match exactly, invalid actions leave state unchanged, NOOP stays valid, and repeated reads do not recompute or corrupt native cache.

**R2 gate:** R2.1 and R2.2 accepted; observation and action semantics remain unchanged.

**Evidence (2026-09-12, accepted):** `crates/bus-sim/src/observation.rs`
reproduces all nine channels at oracle tolerance; `Kernel.mask_cache` is
computed at reset and after each step and returns copies. All 11 fixtures match
every observation channel within `rtol=atol=1e-6` and all 221 mask bits
exactly at reset and after every step (15 tests in
`tests/backend_parity/test_rust_observation.py`), plus 12 native unit tests.
Details in [reports/rust-migration.md](../reports/rust-migration.md).

## 6. R3: integrate the backend

### R3.1 - PyO3 and Gym contract

- [x] Expose reset and one native step per control decision using the contract in Rust spec section 5.
- [x] Preserve Gym seeding/scenario selection, terminated/truncated behavior, terminal observations, and VecEnv auto-reset.
- [x] Convert native cost output into the existing `StepCosts` contract.
- [x] Return arrays with safe lifetime/ownership; do not rebuild Python WorldState each training step.

**Verification:** Gym environment checks plus project environment tests; retain old obs/mask arrays across later steps/resets and prove they are unchanged; interleave two env instances to detect shared mutable state; test DummyVecEnv auto-reset.

**Acceptance:** existing observation/action spaces and step/reset tuple contracts remain compatible; saved rollout inputs cannot be overwritten by subsequent native operations.

### R3.2 - Evaluation, traces, and forecast

- [x] Add a common terminal-summary/trace interface with adapters for both backends.
- [x] Supply cohort/lineage data, departures, actions, totals, and vehicle snapshots needed by `evaluation/runner.py`.
- [x] Keep statistics and plots in Python; export detailed snapshots only at episode end or when tracing is enabled.
- [x] Keep forecast causal and reproduce context flags and episode reset behavior.

**Verification:** fixed-checkpoint and heuristic paired evaluation; compare every metric, categorical/None output, raw component, and trace; exercise zero-demand/censored episodes and forecast on/off; render report outputs through the normal reporting pipeline.

**Acceptance:** metrics and traces are equivalent within the frozen tolerances; forecast has no future access; no routine train step requires full Python state reconstruction.

### R3.3 - Backend selection and lineage

- [x] Implement proposed `runtime.backend` selection throughout CLI/config, train, validation callbacks, diagnose, evaluate, and baselines.
- [x] Fail explicitly when Rust is requested but unavailable; retain Python fallback through explicit selection.
- [x] Record native build/version, backend, hashes, dependency lock, and source revision in metadata.
- [x] Preserve checkpoint compatibility checks; permit cross-backend loading only under matching contracts and verified parity.

**Verification:** run all entry points for each backend; inspect validation metadata for accidental Python fallback; test missing extension, mismatched physics/schema, and matching cross-backend checkpoint load.

**Acceptance:** backend choice is honored end to end and provenance checks remain effective.

**R3 gate:** R3.1-R3.3 accepted; the Rust backend supports the complete workflow.

**Evidence (2026-09-12, accepted):** `Kernel.reset_contract`/
`step_contract` plus `bus_rl.env.native_bus_dispatch.NativeBusDispatchEnv`
mirror the oracle Gym contract; `bus_rl.evaluation.summary` is the shared
summary/trace interface with Python and native adapters; `runtime.backend`
selects the backend across config/CLI/train/validation/evaluate/baselines and
fails loudly when Rust is missing. 21 integration tests
(`tests/backend_parity/test_rust_integration.py`) cover the Gym contract, VecEnv
auto-reset, output ownership, interleaved envs, evaluator/trace parity for four
controllers, forecast on/off, zero-demand censoring, plotting, backend
selection, metadata and cross-backend checkpoint rules. Details in
[reports/rust-migration.md](../reports/rust-migration.md).

## 7. R4: acceptance and handoff

### R4.1 - Correctness and train smoke

- [x] Run the full differential suite, existing Python regression suite, native tests, Python lint/format checks, and native format/lint checks applicable to the selected layout.
- [x] Run a 2048-transition MaskablePPO smoke with finite obs/reward/loss, no masked invalid actions, and checkpoint save/load.
- [x] Compare fixed-checkpoint trajectories and per-day evaluation, not only means.

**Evidence:** exact commands and logs, test counts, parity coverage matrix, checkpoint hashes.

**Acceptance:** no required test fails; unexpected policy divergence is investigated. Newly trained weights or smoke cost are not required to equal the historical 15471.25 result.

### R4.2 - Speed acceptance

- [x] Build release; warm up; rerun both backends in interleaved order, at least five repetitions each, with cProfile/TIMERS disabled.
- [x] Match hardware, threads, dependencies, scenario/action traces, model initialization, logging, and conservation settings.
- [x] Measure simulation-only and isolated learn of at least 12,288 transitions; measure fixed-checkpoint eval separately.
- [x] Report raw times, median/min/max, per-workload behavior, setup cost, and peak RSS. Investigate unstable measurements before deciding.

**Evidence (light pass, 2026-09-12):** `reports/rust-migration/speed-acceptance.json`,
`runtime-tuning.json` and `profile-native.json`. At the chosen 2 torch threads
(same condition both backends) median totals: simulation 0.357 s vs 0.0142 s
(**25.11x**); isolated learn 12,288 transitions 13.159 s vs 3.153 s (**4.17x**);
fixed-checkpoint eval 10 days 1.418 s vs 0.412 s (3.44x), mean cost 15471.25 and
per-day costs identical. Thread matrix: 2/4/16 threads gave Python learn 13.15 /
14.32 / 28.55 s and Rust 3.15 / 3.19 / 8.03 s, so 2 wins for both. Observation
validation is opt-in (`runtime.validate_observation`) and costs at most ~5%
end-to-end. Peak RSS 491 MB is the combined two-backend process. R4.3 (full
workflow) remains open.

**Verification:** Recompute both speedup ratios from raw paired benchmark files; inspect workload-level results and verify all protocol settings against R0.3.

**Acceptance:** median total simulation wall over the trace suite is at most 50% of Python after; median isolated learn wall is at most 50% of Python after. Both require **at least 2x speedup**. **4x is a stretch goal**, not a completion requirement. Native-only speedup cannot substitute for end-to-end learn speedup.

**On failure:** profile, optimize, rerun affected correctness checks, then repeat benchmarks. Keep R4 open; do not start RL tuning or silently reduce the threshold.

### R4.3 - Full workflow and release decision

- [x] Run one full 245,760-transition seed on each backend with identical validation/checkpoint schedules, preferably the L0 protocol to permit reuse.
- [x] Measure actual total wall including validation/checkpoint, with setup reported separately; explain any evaluation or memory regression.
- [x] Complete `reports/rust-migration.md`, link raw evidence, and freeze the native revision/build used by RL experiments.
- [x] Document clean build/install, backend selection, fallback, and reproduction commands.

**Verification:** Compare full-run timing and validation schedules from metadata; reproduce the documented release installation and a saved-checkpoint evaluation.

**Acceptance:** full Rust workflow is faster than Python; all preceding gates pass; evidence is reproducible; the report explicitly marks R4 accepted. Reuse this seed for RL only if every L0 setting and artifact matches.

**Evidence (2026-09-12, accepted):** one full L0-protocol seed per backend
(`configs/experiments/core-threads2.toml`, seed 11, 4 envs, 2 torch threads,
245,760 transitions, validation every 12,288 on 100 validation days) in
`reports/rust-migration/full-workflow.json`. Python 595.35 s vs Rust 161.61 s
learn+validation (**3.68x**); total wall incl. setup 597.99 s vs 164.28 s
(**3.64x**, setup 2.6/2.7 s; same-session pair, the host drifts over the session
so earlier pairs read 3.5-3.7x). All 20 validation costs are bit-identical
(max abs diff 0.0) and `policy.pth`, `policy.optimizer.pth` and
`pytorch_variables.pth` are byte-identical for both `best.zip` and `last.zip`.
Validation dominates the workflow (46% of Python wall); rerunning with
`--eval-limit 10` isolates that cost. Three storage bugs were found and fixed so
Rust peak RSS is now 508 MB vs Python 479 MB (**1.06x**):
`NativeBusDispatchEnv` eagerly built one kernel per scenario (each retains
~243 KB of episode scratch; now lazy in `reset()`), the store kept a dense
~405 KB arrival tape that is 98.5% zeros (now sparse, store 245.6 -> 30.1 MB),
and the tape was int32 with max value 5 (now int8; 600 scenarios 249 -> 67 MB).
Rust peak went 1236 -> 508 MB. Native revision `aedc4faa4c43` is frozen for RL;
if the L0 seed reuses these exact settings and artifacts, no rerun is needed.

## 8. Final acceptance checklist

- [x] R0: immutable oracle, golden coverage, and unprofiled reference accepted.
- [x] R1: domain/lifecycle/actions/ticks/costs accepted.
- [x] R2: observation/history and mask parity accepted.
- [x] R3: wrapper/evaluator/forecast/backend/provenance accepted.
- [x] R4: correctness, 2x simulation and learn gates, and faster full workflow accepted.
- [x] Historical reports preserved; new report distinguishes measured results from projections.
- [x] Handoff records the frozen backend revision (`aedc4faa4c43`) and unlocks task L0.1 in the [RL plan](improve_RL.md).

All items pass; this document now records an accepted Rust backend.
