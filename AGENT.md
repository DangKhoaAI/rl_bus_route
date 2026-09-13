# AGENT.md — working in `bus-rl`

CPU simulator + MaskablePPO controllers for dynamic bus fleet control on a
fixed 3-route network. Python 3.11, Linux, CPU PyTorch. This is a synthetic
operational-control study, not a deployment tool. Specs live in `docs/spec/`,
plans in `docs/plan/`.

## Backend model (read this first)

- **Rust is the default and maintained simulator.** `RuntimeConfig.backend`
  defaults to `"rust"`.
- **The Python oracle is deprecated.** `src/bus_sim/oracle/` and
  `src/bus_rl/execution/environments/python/` receive no further fixes. Selecting
  them needs an explicit opt-in:
  `--backend python --legacy-python` (or `runtime.legacy_python = true`), and it
  emits a `DeprecationWarning`. Requesting `backend="python"` without the flag is
  a hard error.
- **No silent fallback.** A `rust` run without the extension fails loudly with a
  build hint.

## Layout

```text
src/
├── bus_sim/
│   ├── oracle/          # DEPRECATED Python simulator kernel
│   └── parity/          # scenario catalog, state snapshots, controllers (test/bench infra)
└── bus_rl/
    ├── execution/
    │   ├── environments/{factory,python,rust}  # backend selection, Gym wrappers, FFI bridge
    │   ├── scenarios/   # generation + persistence (manifest, tapes)
    │   ├── runtime.py   # thread/device settings
    │   └── timing.py    # TIMERS instrumentation
    ├── learning/        # baselines, forecasting, policy, training/{train,callbacks,checkpoint,diagnose}
    ├── evaluation/      # runner, pool, summary, statistics, plots, profile
    ├── config.py        # RunConfig / RuntimeConfig / dataclass TOML mapping
    ├── cli.py           # generate | baseline | profile | train | diagnose | evaluate | report
    └── provenance.py    # hashes + fresh-output guards
crates/
├── bus-sim-core/        # pure Rust kernel (no PyO3); `cargo test -p bus-sim-core`
└── bus-sim-python/      # PyO3 cdylib exposing `bus_sim_native` (Kernel/ScenarioStore/BatchKernel)
tests/                   # see tests/README.md for the naming rules
configs/                 # TOML runs; `base.toml` is merged under every other config
scripts/                 # build_native, benchmarks, profiling, tune
docs/                    # specs, plans, research
reports/                 # frozen reproducible evidence — do not rewrite
```

**Live-path caveat:** the Rust wrapper still imports a few oracle modules as the
shared vocabulary — `bus_sim.oracle.{domain,costs,observation}` and the
`Scenario`/`ScenarioDigest` types. Changing those affects both backends. Only the
oracle *engine* (`engine`, `dispatcher`, `guards`, `passengers`, `vehicles`,
`travel`, `actions`, and the Python `environment.py`) is deprecated.

## Setup and build

```bash
uv sync --locked --extra dev
uv run python scripts/build_native.py    # required: builds src/bus_sim_native.so
```

`scripts/build_native.py` runs `cargo build -p bus-sim-python --release`, copies
`target/release/libbus_sim_native.so` to `src/bus_sim_native.so`, verifies import,
and writes provenance to `reports/rust-migration/native-build.json`.

`src/bus_sim_native.so`, `data/generated/`, `runs/`, `target/`, `.venv/` are
gitignored — never commit them.

## Quality gates

```bash
uv run ruff check .                  # line-length 100, target py311
uv run ruff format --check .
uv run pytest -q                     # native default; legacy tests skipped (-n 4)
uv run pytest -q --legacy-python     # include deprecated-oracle tests
uv run pytest -q -n 0                # serial when debugging
uv run pytest -m native -q           # Python↔Rust FFI contract only
cargo test -p bus-sim-core           # kernel unit tests
```

Never commit red gates. Fix or revert; do not weaken tests to pass.

## Change → verify matrix

| You changed | Run |
|---|---|
| `crates/bus-sim-core/**` | `cargo test -p bus-sim-core`, then rebuild `.so`, then `uv run pytest -m native -q` and `uv run pytest -q` |
| `crates/bus-sim-python/**` (FFI) | rebuild `.so`, `uv run pytest -m native -q`, `uv run pytest -q` |
| `src/bus_rl/execution/environments/rust/**` | rebuild if the Rust side changed, `uv run pytest -q` |
| `src/bus_rl/config.py`, `cli.py`, `factory.py` | `uv run pytest -q`, and exercise the CLI smoke |
| `src/bus_sim/oracle/**` | `uv run pytest -q --legacy-python` (and accept it is deprecated) |
| docs only | nothing, but keep commands runnable |

Targeted runs while iterating, full `uv run pytest -q` before declaring done.

## Invariants

- **Physics/semantics changes need evidence.** Keep the action table (221 slots),
  NOOP validity, masks/guards, cost terms, and conservation intact unless the
  change is intended; speed work must not alter semantics. Accepted parity/speed
  evidence is frozen in `reports/rust-migration/`.
- **Determinism matters.** Scenario/seed/order affect results; config and
  provenance hashes are recorded in checkpoints and run metadata. Do not add
  nondeterministic iteration order.
- **Cross-backend checkpoints** load only when contract hashes match and parity is
  marked verified (`checkpoint.py`).
- **Conservation checks** (`bus_sim.oracle.domain.CONSERVATION_CHECKS`) are on in
  tests and must be off for speed references.
- **`reports/` is evidence.** Add new reports; do not edit accepted ones.
- **No new tests for deprecated code** expecting fixes. Follow `tests/README.md`:
  unit mirrors `src` as `test_<module>.py`; integration uses
  `test_<subject>[_<aspect>].py` under the entry-point path.

## Common tasks

- **Change a run option:** add to `RuntimeConfig`/`RunConfig` in `config.py`, wire
  it in `cli.py` (`_run_config`) if it needs a flag, and update `configs/base.toml`.
- **Change the observation contract:** `crates/bus-sim-core/src/observation.rs`
  plus the Python consumer; shapes/keys are pinned by
  `tests/integration/.../rust/test_kernel_contract.py`.
- **Add a Rust binding surface:** `crates/bus-sim-python/src/lib.rs` (`#[pymethods]`
  / `#[pyclass]` / `#[pymodule]`), then a `tests/integration/.../rust/` contract
  test; the consumer is `.../environments/rust/bridge.py`.
- **Add a CLI subcommand:** `src/bus_rl/cli.py`, then extend
  `tests/integration/test_cli_pipeline.py` if it belongs to the smoke flow.

## Where to look first

- Backend selection: `src/bus_rl/execution/environments/factory.py`
- FFI bridge / payload schema: `src/bus_rl/execution/environments/rust/bridge.py`
- Batch stepping: `.../environments/rust/batch.py` + `BatchKernel` in `lib.rs`
- Training entry: `src/bus_rl/learning/training/train.py`
- Evaluator/metrics: `src/bus_rl/evaluation/runner.py`, `summary.py`
- Native tests: `reports/rust-migration/` (frozen), `crates/README.md`
