# bus-rl

CPU simulator and MaskablePPO controllers for dynamic bus fleet control
(reserve dispatch, headway targets, reassign, short-turn) on a fixed 3-route
network. Specification: `docs/spec/spec_v1.0.md`. Plan:
`docs/plan/plan_v1.0.md`.

Python 3.11, Linux, CPU PyTorch. This is a synthetic operational-control study,
not a city deployment or a new-route design tool.

## Repository layout

```text
src/
├── bus_sim/
│   ├── oracle/          # Reference Python simulator
│   └── parity/          # Python/Rust fixtures, snapshots, and contracts
└── bus_rl/
    ├── execution/
    │   ├── environments/
    │   │   ├── python/  # Gym wrapper for `bus_sim.oracle`
    │   │   └── rust/    # Gym wrappers and bridge for `bus_sim_native`
    │   ├── scenarios/   # Scenario generation and persistence
    │   ├── runtime.py   # Runtime and thread settings
    │   └── timing.py    # Runtime timing instrumentation
    ├── learning/        # Baselines, forecasting, policy, and training
    └── evaluation/      # Controller metrics, plots, and profiling
crates/
├── bus-sim-core/        # Optimized Rust simulator
└── bus-sim-python/      # PyO3 bridge (`bus_sim_native`)
tests/
├── support/             # Shared factories, paths and comparison helpers
├── unit/                # Python unit tests (sim/, rl/)
└── integration/         # CLI and training workflow tests
configs/                 # Run and experiment configurations
scripts/                 # Build, benchmark, and profiling utilities
docs/                    # Specifications, plans, and research notes
reports/                 # Selected reproducible results and evidence
```

Python packages follow the standard `src/` layout, while Rust packages follow
the Cargo workspace convention under `crates/`. Historical PPO checkpoints
that reference `bus_rl.models.features` are translated transparently by the
checkpoint loader; the obsolete package is not retained in the source tree.

## Setup

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Tiny smoke (2 train / 1 val / 1 test days, 32 PPO steps) lives in
`tests/integration/test_cli_pipeline.py`.

## Reproduce

Generate the frozen 500/100/200/200/200-day manifests (not committed):

```bash
uv run bus-rl generate --config configs/base.toml --output data/generated/base
```

Baselines, profiler, pilot (12,288 transitions), and evaluation:

```bash
uv run bus-rl baseline --config configs/eval.toml --manifest data/generated/base/manifest.json --split validation --methods fixed,threshold,proportional --output runs/baselines
uv run bus-rl profile --config configs/pilot.toml --decisions 120 --output runs/profile
uv run bus-rl train --config configs/pilot.toml --seed 11 --output runs/pilot-11
uv run bus-rl evaluate --config configs/eval.toml --checkpoint runs/pilot-11/best.zip --manifest data/generated/base/manifest.json --split validation --output runs/pilot-11/eval
uv run bus-rl report --results runs/pilot-11/eval/results.csv --output reports/core-11
```

Core M3 (245,760 transitions, seeds 11/22/33):

```bash
uv run bus-rl train --config configs/experiments/core.toml --seed 11 --output runs/core-11
uv run bus-rl train --config configs/experiments/core.toml --seed 22 --output runs/core-22
uv run bus-rl train --config configs/experiments/core.toml --seed 33 --output runs/core-33
```

Ablations use `configs/experiments/{no_reassign,no_short,fairness_zero}.toml`
with the same tapes, 221-slot table, and compute. Forecast experiment:
`configs/experiments/forecast.toml` (historical time-bin mean + recent
correction; no future tape).

Large artifacts (`data/generated/`, `runs/*.zip`) are gitignored and must be
recreated. Selected reports live in `reports/`.

## Native kernel (Rust)

The Rust migration (`docs/spec/rust_improve.md`, `docs/plan/rust_improve.md`)
is complete: R0-R4 ported the kernel and the native backend is accepted, while
Python remains the default reference. The one-off R0 golden fixtures and the
native Python-level parity suite were retired after acceptance; their evidence
is archived under `reports/rust-migration/`, and the Rust core keeps its own
`cargo test` coverage.

```bash
python scripts/build_native.py       # build and install src/bus_sim_native.so
cargo test -p bus-sim-core           # native unit tests
```

Select the backend per run (default `python`, configurable via `[runtime]
backend` or `--backend`):

```bash
uv run bus-rl evaluate --config configs/eval.toml --manifest data/generated/base/manifest.json \
  --split validation --method threshold --output runs/eval-rust --backend rust
```

Requesting `--backend rust` without the extension fails loudly; Python is kept
as an explicit fallback. Run metadata records the backend and native build hash.
See `crates/README.md` and `reports/rust-migration.md`.

### Fast development loop

```bash
python -m pytest -q                              # parallel -n 4 (~20 s)
python -m pytest -q -n 0                         # serial, for debugging
```

`pytest` runs 4 xdist workers by default (2 torch threads each); use
`python -m pytest -q -n 0` for a serial run when debugging.

## What is in the observation

Controllers see queues, ages, loads, past arrivals, headways, and fleet
status. They do not see future demand tapes, scenario seeds, or latent
passenger destinations. Forecasts, when enabled, are fit on train-day arrival
logs only.
