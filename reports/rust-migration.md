# Rust migration report

Status: **R0 accepted (Python oracle frozen). R1–R4 not started.**
Date: 2026-09-12.
Spec: [docs/spec/rust_improve.md](spec/rust_improve.md). Plan: [docs/plan/rust_improve.md](plan/rust_improve.md).

This report records measured evidence only. It does not mark any R1–R4 task
complete and does not claim a native backend exists.

## 0. Why R0 and not RL L0

The RL plan's **L0 baseline** is gated on an *accepted Rust R4* (parity + ≥2×
simulation/learn gates). No Rust crate exists in this revision, so RL L0 cannot
run without violating the plan. The first executable stage in the documents
referenced for this task is **R0 — freeze the Python oracle**. R0.1–R0.3 are
implemented and accepted here; R1 is the next task.

## 1. R0.1 — Inventory and freeze the reference

The oracle manifest is `reports/rust-migration/oracle-manifest.json`
(machine-checkable, self-hashed). Everything is regenerated with:

```text
python scripts/export_oracle.py build     # reference + fixtures + manifest
python scripts/export_oracle.py verify    # re-derive every hash and replay
```

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

`reports/training-diagnosis-after.md` measured 266 decisions/s **with cProfile
and nested timers enabled**. That number is diagnostic only and is explicitly
not the production reference. The unprofiled numbers above are the R0.3
reference that R4 will remeasure interleaved with the Rust backend.

## 4. Evidence

| Command | Exit | Artifacts |
|---|---:|---|
| `python scripts/export_oracle.py build` | 0 | manifest, 11 fixtures, summary |
| `python scripts/export_oracle.py verify` | 0 | all hashes + 11 replays + reference 15471.25 |
| `python scripts/benchmark_python.py --repetitions 5 --transitions 12288` | 0 | `python-benchmark.json` |
| `python -m pytest tests/backend_parity -q` | 0 | 32 passed |
| `python -m pytest -q` | 0 | full existing suite + parity |

Artifacts:

- `reports/rust-migration/oracle-manifest.json` — frozen inventory + contract.
- `reports/rust-migration/fixtures-summary.json` — fixture hashes/coverage.
- `reports/rust-migration/python-benchmark.json` — raw repetitions.
- `tests/backend_parity/fixtures/` — golden fixtures (committed).
- `tests/backend_parity/reference/` — committed reference checkpoint.

## 5. Limitations and next steps

- R0 records the Python oracle; it does not accept any Rust task. R1 (native
  build/domain/reset) is next and must consume these fixtures.
- The manifest records `git_dirty=true` because R0 artifacts were added in the
  same working tree; the revision field pins `1c34457`.
- The git-ignored `runs/` and `data/generated/` inventories are documented but
  optional for verification; the committed reference checkpoint and
  regeneration commands make R0 reproducible from a fresh checkout.
- R4 must remeasure Python interleaved with Rust on one machine; the R0.3
  numbers are the Python baseline for the ≥2× simulation and isolated-learn
  gates.

## 6. R0 checklist

- [x] R0.1 immutable oracle, contract documented, discrepancies recorded.
- [x] R0.2 named golden fixtures, differential harness, corrupted-fixture demo,
      reusable fixed reference checkpoint.
- [x] R0.3 unprofiled Python benchmark with raw repetitions and median/min/max.
- [x] R0 gate accepted: oracle frozen before native implementation.
- [ ] R1 native kernel.
