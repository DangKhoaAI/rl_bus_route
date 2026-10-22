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
│   ├── oracle/          # DEPRECATED reference Python simulator
│   └── parity/          # Scenario catalog, snapshots, controllers
└── bus_rl/
    ├── execution/
    │   ├── environments/
    │   │   ├── python/  # DEPRECATED Gym wrapper for `bus_sim.oracle`
    │   │   └── rust/    # Gym wrappers and bridge for `bus_sim_native` (default)
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
├── unit/                # Isolated tests, mirroring `src` (`bus_sim/oracle`, `bus_rl/…`)
└── integration/         # Cross-module tests, mirroring `src/bus_rl` (Python env,
                         # Rust FFI gate, RL) plus the CLI pipeline
configs/                 # Run and experiment configurations
scripts/                 # Build, benchmark, and profiling utilities
docs/                    # Specifications, plans, and research notes
reports/                 # Curated evidence; start at reports/README.md
```

Python packages follow the standard `src/` layout, while Rust packages follow
the Cargo workspace convention under `crates/`. The Rust kernel is the default
and maintained simulator; `src/bus_sim/oracle/` and
`src/bus_rl/execution/environments/python/` are deprecated and kept only as a
reference. Historical PPO checkpoints that reference `bus_rl.models.features`
are translated transparently by the checkpoint loader; the obsolete package is
not retained in the source tree.

## Setup

```bash
uv sync --locked --extra dev
python scripts/build_native.py       # required: the Rust backend is the default
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

`pytest` runs the native backend by default and skips deprecated Python tests;
pass `--legacy-python` to include them.

Tiny smoke (2 train / 1 val / 1 test days, 32 PPO steps) lives in
`tests/integration/test_cli_pipeline.py`.

## CLI

Commands use `bus-rl <noun> <action>`:

| Command | Purpose |
|---|---|
| `data generate` | Generate scenario data and a manifest |
| `model train` | Train a policy and save best/last checkpoints |
| `model evaluate` | Evaluate a checkpoint and write results/traces |
| `baseline run` | Evaluate fixed/threshold/proportional controllers |
| `report generate` | Generate plots and a summary from results |
| `test diagnose` | Diagnose training and evaluation timings |
| `test profile` | Run the historical simulator microprofiler |

Use `bus-rl model --help` or `bus-rl model train --help` for a group or action.
Flat legacy commands such as `train`, `evaluate` and `generate` remain accepted
as aliases so archived command lines still run. New documentation uses the
grouped names. Flags, config precedence and output artifacts are unchanged.

To collect a complete run under `runs/result-gru10/`:

```bash
uv run bus-rl model train --manifest data/generated/base/manifest.json --output runs/result-gru10/train
uv run bus-rl model evaluate --manifest data/generated/base/manifest.json --checkpoint runs/result-gru10/train/best.zip --split test_id --trace --output runs/result-gru10/eval
uv run bus-rl report generate --results runs/result-gru10/eval/results.csv --history runs/result-gru10/train/evaluations.csv --output runs/result-gru10/report
```

Each `--output` must be a fresh path; parent folders are created automatically.

## Policy selection

Training now defaults to **MaskablePPO + GRU, 10 minutes of arrival history,
without belief features**. This was the best supported compact choice in the
98,304-step encoder ablations; it has not replaced the frozen full-budget MLP
benchmark in `reports/README.md`. `--config` is optional for `model train`, `test diagnose`
and `model evaluate` (defaults: `configs/train.toml`, `configs/train.toml`,
`configs/eval.toml`). The dataset manifest and output directory remain explicit.

```bash
# Default: GRU10
uv run bus-rl model train --manifest data/generated/base/manifest.json --output runs/gru10

# Override with flags
uv run bus-rl model train --manifest data/generated/base/manifest.json --output runs/mlp --encoder mlp
uv run bus-rl model train --manifest data/generated/base/manifest.json --output runs/gru30 --encoder gru --history-minutes 30

# Or use a ready-to-run TOML preset
uv run bus-rl model train --manifest data/generated/base/manifest.json --output runs/belief20-fade --config configs/policies/belief20-fade.toml

# Evaluation restores encoder/history from the chosen checkpoint
uv run bus-rl model evaluate --manifest data/generated/base/manifest.json --checkpoint runs/gru10/best.zip --output runs/gru10/eval
```

Presets under `configs/policies/`: `mlp`, `gru10`, `gru20`, `gru30`, `belief20`,
`belief20-weak`, `belief20-fade`, `gru20-belief-fade`. They vary only the encoder,
history and belief parameters; common settings are inherited from `base.toml`
and configuration defaults. These presets use the normal training budget;
the historical ablation budget remains in `configs/experiments/`.

For `model train`/`test diagnose`, explicit flags override TOML, which overrides base
defaults: `--encoder {mlp,gru,belief,belief_gru}`, `--history-minutes`,
`--belief-prior-shape`, `--belief-prior-rate`, `--belief-decay`. The resolved
settings are saved to `metadata.json`. When evaluating a checkpoint, its saved
encoder/history take precedence over the evaluation config.

Historical experiment configs explicitly retain `encoder="mlp"` so rerunning
them preserves the original architecture. See the
[encoder study](reports/archive/belief-gru-ablation/README.md) and
[memory/belief follow-up](reports/archive/memory-belief-ablation/README.md).

## Reproduce the original study

Generate the frozen 500/100/200/200/200-day manifests (not committed):

```bash
uv run bus-rl data generate --config configs/base.toml --output data/generated/base
```

Baselines, profiler, pilot (12,288 transitions), and evaluation:

```bash
uv run bus-rl baseline run --config configs/eval.toml --manifest data/generated/base/manifest.json --split validation --methods fixed,threshold,proportional --output runs/baselines
uv run bus-rl test profile --config configs/pilot.toml --decisions 120 --output runs/profile
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/pilot.toml --seed 11 --output runs/pilot-11
uv run bus-rl model evaluate --config configs/eval.toml --checkpoint runs/pilot-11/best.zip --manifest data/generated/base/manifest.json --split validation --output runs/pilot-11/eval
uv run bus-rl report generate --results runs/pilot-11/eval/results.csv --output reports/core-11
```

Core M3 (245,760 transitions, seeds 11/22/33):

```bash
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/experiments/core.toml --seed 11 --output runs/core-11
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/experiments/core.toml --seed 22 --output runs/core-22
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/experiments/core.toml --seed 33 --output runs/core-33
```

Ablations use `configs/experiments/{no_reassign,no_short,fairness_zero}.toml`
with the same tapes, 221-slot table, and compute. Forecast experiment:
`configs/experiments/forecast.toml` (historical time-bin mean + recent
correction; no future tape).

Large artifacts (`data/generated/`, `runs/*.zip`) are gitignored and must be
recreated. Start at `reports/README.md`; the final result is
`reports/README.md`.

## Native kernel (Rust)

The Rust migration (`docs/spec/rust_improve.md`, `docs/plan/rust_improve.md`)
is complete: R0-R4 ported the kernel and the native backend is accepted. The
native Rust kernel is now the **default and maintained** backend; the Python
oracle is deprecated and receives no further fixes. The one-off R0 golden
fixtures and the native Python-level parity suite were retired after
acceptance; their evidence is archived under `reports/archive/infrastructure/rust-migration/`, and the
Rust core keeps its own `cargo test` coverage. The Python↔Rust **interface** is
still enforced by the Rust FFI suite in
`tests/integration/bus_rl/execution/environments/rust/` (payload schema, array
shapes/dtypes, error mapping, ownership, batch lifecycle); it skips when the
extension is not built.

```bash
python scripts/build_native.py       # build and install src/bus_sim_native.so
cargo test -p bus-sim-core           # native unit tests
uv run pytest -m native -q           # Python↔Rust interface contract
```

`backend` defaults to `rust` (via `[runtime] backend` or `--backend`):

```bash
uv run bus-rl model evaluate --config configs/eval.toml --manifest data/generated/base/manifest.json \
  --split validation --method threshold --output runs/eval-rust
```

Running without the extension fails loudly. The deprecated Python oracle needs
an explicit opt-in and warns when used:

```bash
uv run bus-rl model evaluate --config configs/eval.toml --manifest data/generated/base/manifest.json \
  --split validation --method threshold --output runs/eval-py \
  --backend python --legacy-python
```

Run metadata records the backend and native build hash.
See `crates/README.md` and `reports/archive/infrastructure/rust-migration/README.md`.

### Fast development loop

```bash
python -m pytest -q                              # native default, parallel -n 4 (~20 s)
python -m pytest -q --legacy-python              # also run the deprecated oracle tests
python -m pytest -q -n 0                         # serial, for debugging
```

`pytest` runs 4 xdist workers by default (2 torch threads each); use
`python -m pytest -q -n 0` for a serial run when debugging.

## What is in the observation

Controllers see queues, ages, loads, past arrivals, headways, and fleet
status. They do not see future demand tapes, scenario seeds, or latent
passenger destinations. Forecasts, when enabled, are fit on train-day arrival
logs only.
