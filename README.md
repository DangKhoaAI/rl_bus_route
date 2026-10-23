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
scripts/                 # Native build and study-report utilities; see scripts/README.md
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
uv run python scripts/build_native.py       # required: the Rust backend is the default
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

`pytest` runs the native backend by default and skips deprecated Python tests;
pass `--legacy-python` to include them.

Tiny smoke (2 train / 1 val / 1 test days, 32 PPO steps) lives in
`tests/integration/test_cli_pipeline.py`.

## CLI

Quick workflow: [data → train → evaluate → report](scripts/README.md).

Commands use `bus-rl <noun> <action>`:

| Command | Purpose |
|---|---|
| `data generate` | Generate scenario data and a manifest |
| `model train` | Train a policy and save best/last checkpoints |
| `model evaluate` | Evaluate a checkpoint and write results/traces |
| `baseline run` | Evaluate fixed/threshold/proportional controllers |
| `report generate` | Generate a report bundle: tables, figures and provenance |
| `test diagnose` | Diagnose training and evaluation timings |
| `test profile` | Run the historical simulator microprofiler |
| `experiment run` | Run a fixed policy/seed matrix and export Git evidence |
| `artifact export` | Export completed evidence without model state or logs |
| `artifact verify` | Verify the exported inventory and SHA256 checksums |

Use `bus-rl model --help` or `bus-rl model train --help` for a group or action.
Flat legacy commands such as `train`, `evaluate` and `generate` remain accepted
as aliases so archived command lines still run. New documentation uses the
grouped names. Flags and config precedence are unchanged.

To collect a complete run under `experiments/result-gru10/`:

```bash
uv run bus-rl model train --manifest data/generated/base/manifest.json --output experiments/result-gru10/train
uv run bus-rl model evaluate --manifest data/generated/base/manifest.json --checkpoint experiments/result-gru10/train/best.zip --split test_id --trace --output experiments/result-gru10/eval
uv run bus-rl baseline run --config configs/eval.toml --manifest data/generated/base/manifest.json --split test_id --methods fixed,proportional,threshold --output experiments/result-gru10/baselines
uv run bus-rl report generate --results experiments/result-gru10/eval/results.csv experiments/result-gru10/baselines/results.csv --output experiments/result-gru10/report
```

Each `--output` must be a fresh path; parent folders are created automatically.

`report generate` accepts multiple RL/baseline CSVs and creates an English,
presentation-ready report in one command:

```text
experiments/result-gru10/report/
└── report.html  # Run configuration, test conditions and final results; embedded figures
```

Each report is one portable HTML file. The main content shows the run setup,
each test condition and the final cost/service comparisons. Training curves,
full settings and input hashes are expandable inside the same document.
Original data remain in `train/` and `eval/`; no CSV, Markdown, JSON or figure
copies are emitted into the report folder. Study-wide `summary.csv` and
`comparisons.csv` remain at the study root for comparisons between policies.

The four main figures use short English titles, direct value labels and a
consistent RL accent color. Cost and waiting-time comparisons use point charts
with explicitly labelled data-scaled axes. Cost-reduction charts show both
positive and negative results with a zero reference and paired confidence
intervals. Tables and findings preserve service trade-offs, including worse P95
wait even when aggregate cost improves.

Training records a real step-zero validation measurement, the seed, encoder and
memory window, plus every subsequent validation score. Step zero is a reference
only; checkpoint selection retains the lowest **post-training** validation cost
with earliest-checkpoint tie breaking. Metadata preserves initial/final/best
scores, selected step, data counts/hashes and checkpoint/history hashes.

`model evaluate` saves `evaluation.json` and a portable `training_history.csv`
next to `results.csv`; report generation automatically loads and verifies them.
`baseline run` also records evaluation metadata. Explicit histories are supported
for older runs or separate archives:

```bash
uv run bus-rl report generate \
  --results experiments/core-11/eval/results.csv experiments/core-22/eval/results.csv experiments/baselines/results.csv \
  --history experiments/core-11/evaluations.csv experiments/core-22/evaluations.csv \
  --output reports/comparison
```

Pass one CSV per method/seed/split, or a combined CSV. `--rl-method` selects the
RL method when automatic detection (`ppo`, `maskable_ppo`, `core`) is unsuitable.
All compared models must cover identical scenario days with matching seeds and
tape hashes; duplicate or partially matched inputs fail before report output is
created. Missing scenario identity yields an explicitly unavailable contrast.

Savings are `100 × (baseline − RL) / baseline`. The report averages model seeds
within each day before a paired day bootstrap (2,000 resamples, seed 6001), and
recomputes the percentage ratio on each resample. Intervals exclude training-seed
uncertainty and are not adjusted for multiple comparisons. A single day cannot
produce a significance claim. Validation scores remain labelled as model-selection
data, rather than independent test evidence.

Older CSVs remain supported, with missing configuration explicitly marked.
Histories without step zero use their first recorded evaluation as reference;
no untrained score or seed is fabricated. Training/comparison panels explicitly
indicate missing inputs. Trace images are generated only when traces are supplied.
Report generation summarizes existing artifacts; it does not train or evaluate
models. Use `experiment run --artifacts runs/<study>` to export reproducible evidence; the final report under `reports/` is authored separately.

## Local experiments and Git artifacts

Three directories have distinct roles:

| Directory | Contents | Git policy |
|---|---|---|
| `experiments/` | Local checkpoints, intermediate evaluation data and exploratory runs | Ignored |
| `runs/` | Fixed, reproducible evidence exports: configs, metadata, CSVs, figures and technical summaries | Tracked |
| `reports/` | Human-authored final study assembled from the evidence | Tracked |

Run the standard matrix and publish its compact evidence locally:

```bash
uv run bus-rl experiment run \
  --config configs/studies/standard.toml \
  --manifest data/generated/base/manifest.json \
  --output experiments/standard-v1 \
  --artifacts runs/standard-v1
uv run --with reportlab python scripts/build_study_report.py --study runs/standard-v1
uv run bus-rl artifact verify --path runs/standard-v1
```

The fixed matrix uses MLP10, GRU10 and Belief20-fade, seeds 11/22/33,
245,760 transitions per seed, 500 training days and 100 validation days.
All training finishes before evaluating the three 200-day test splits. Shared
Fixed/Proportional/Threshold evaluations use exactly the same scenario tapes.
Three isolated training processes run concurrently; model RNG state is never shared.
This is a reproducibility snapshot on the established reference dataset, not a
newly unseen test set or an automatic replacement for the accepted final study.

Local runs follow `<study>/<policy>/<seed>/{train,eval}`; each policy has a
`report/` aggregating all seeds. Baselines use `baselines/shared/eval/`.
A frozen protocol includes source/config/data hashes and resolved settings.
Training diagnostics are disabled; only best checkpoints are retained after
successful completion. Git exports omit checkpoints, optimizer state, raw
validation telemetry and logs. After consolidation, the study root contains
`README.md`, `report.md`, `report.html`, `report.pdf` and `evidence/`. Original
per-day results, configs and metadata stay under `evidence/`; redundant per-policy
presentation copies are removed. The report builder checks its numbers against
per-day results and does not retrain models.

Export or verify a completed study independently:

```bash
uv run bus-rl artifact export --source experiments/standard-v1 --output runs/standard-v1-copy
uv run bus-rl artifact verify --path runs/standard-v1-copy
```

Every output must be fresh. `experiment run --resume` can reuse completed stages
only when the frozen configuration, dataset and implementation still match.
Tracked evidence contains no model weights; reevaluation requires the local
best checkpoint or retraining from the recorded configuration.

Historical raw runs were moved from `runs/` to `experiments/archive/`. Important
benchmark data and selected checkpoints remain available there. Diagnostics,
redundant final checkpoints, performance-probe weights and superseded demos were
removed; `experiments/cleanup-20261022.json` records the local cleanup inventory.
Accepted historical evidence under `reports/` remains frozen. See
[run artifacts](docs/run-artifacts.md) for the directory contract.

## Policy selection

Training now defaults to **MaskablePPO + GRU, 10 minutes of arrival history,
without belief features**. This was the best supported compact choice in the
98,304-step encoder ablations; it has not replaced the frozen full-budget MLP
benchmark in `reports/README.md`. `--config` is optional for `model train`, `test diagnose`
and `model evaluate` (defaults: `configs/train.toml`, `configs/train.toml`,
`configs/eval.toml`). The dataset manifest and output directory remain explicit.

```bash
# Default: GRU10
uv run bus-rl model train --manifest data/generated/base/manifest.json --output experiments/gru10

# Override with flags
uv run bus-rl model train --manifest data/generated/base/manifest.json --output experiments/mlp --encoder mlp
uv run bus-rl model train --manifest data/generated/base/manifest.json --output experiments/gru30 --encoder gru --history-minutes 30

# Or use a ready-to-run TOML preset
uv run bus-rl model train --manifest data/generated/base/manifest.json --output experiments/belief20-fade --config configs/policies/belief20-fade.toml

# Evaluation restores encoder/history from the chosen checkpoint
uv run bus-rl model evaluate --manifest data/generated/base/manifest.json --checkpoint experiments/gru10/best.zip --output experiments/gru10/eval
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
uv run bus-rl baseline run --config configs/eval.toml --manifest data/generated/base/manifest.json --split validation --methods fixed,threshold,proportional --output experiments/baselines
uv run bus-rl test profile --config configs/pilot.toml --decisions 120 --output experiments/profile
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/pilot.toml --seed 11 --output experiments/pilot-11
uv run bus-rl model evaluate --config configs/eval.toml --checkpoint experiments/pilot-11/best.zip --manifest data/generated/base/manifest.json --split validation --output experiments/pilot-11/eval
uv run bus-rl report generate --results experiments/pilot-11/eval/results.csv --output experiments/pilot-11/report
```

Core M3 (245,760 transitions, seeds 11/22/33):

```bash
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/experiments/core.toml --seed 11 --output experiments/core-11
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/experiments/core.toml --seed 22 --output experiments/core-22
uv run bus-rl model train --manifest data/generated/base/manifest.json --config configs/experiments/core.toml --seed 33 --output experiments/core-33
```

Ablations use `configs/experiments/{no_reassign,no_short,fairness_zero}.toml`
with the same tapes, 221-slot table, and compute. Forecast experiment:
`configs/experiments/forecast.toml` (historical time-bin mean + recent
correction; no future tape).

Large artifacts (`data/generated/`, `experiments/*.zip`) are gitignored and must be
recreated. Start at `reports/README.md`; the final result is
`reports/README.md`.

## Station API

`src/bus_rl/station/` serves live native simulations of one held-out day over
HTTP for the 3D dashboard (`web/station/`, served at `/visualize/` when present; `/` redirects to `/intro/`). Each
`POST /simulations/{id}/steps` advances N control intervals with a baseline
controller and returns per-route buses, the depot reserve queue and interval
costs. Settings come from `.env` (see `.env.example`); the manifest is required.

`POST /simulations` takes the controller, the split and the day
(`{"controller": "ppo", "split": "test_ood_burst", "scenario_index": 3}`);
`SIMULATION__SPLITS` lists the splits the station loads, and the first is the
default. The dashboard offers the same three choices in its top bar: picking
one starts a new simulation.

### Run locally

One process serves both the API and the dashboard; the page calls the API on its
own origin, so there is no separate frontend server to start.

```bash
uv sync --locked --extra dev --extra server
uv run python scripts/build_native.py        # once: builds the Rust kernel
SIMULATION__MANIFEST_PATH=data/generated/base/manifest.json \
  uv run uvicorn bus_rl.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/intro/ and /visualize/
```

### Run with Docker

The image builds the Rust kernel and regenerates the scenario days, so the demo
machine needs no Rust toolchain and no torch.

```bash
docker build -t bus-rl-station .
docker run --rm -p 8000:8000 bus-rl-station
# open http://localhost:8000/intro/ and /visualize/
```

### Serve a trained policy

Point the station at a checkpoint (`metadata.json` must sit beside it); the
dashboard opens on `fixed` and lists `ppo` beside the baselines. This needs torch, so the default Docker
image runs the baselines only.

```bash
SIMULATION__CHECKPOINT_PATH=runs/core-11/best.zip \
SIMULATION__MANIFEST_PATH=data/generated/base/manifest.json \
  uv run uvicorn bus_rl.main:app --host 127.0.0.1 --port 8000
```

Without Docker, download the checkpoint first:
`gh release download core-11 --repo DangKhoaAI/rl_bus_route --dir runs/core-11`.
The `ppo` image target adds torch and downloads that checkpoint itself:

```bash
docker build --target ppo -t bus-rl-station:ppo .
```

```bash
docker run --rm -p 8000:8000 bus-rl-station:ppo
```

The page loads three.js from a CDN, so the demo machine needs internet access.

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
uv run python scripts/build_native.py       # build and install src/bus_sim_native.so
cargo test -p bus-sim-core           # native unit tests
uv run pytest -m native -q           # Python↔Rust interface contract
```

`backend` defaults to `rust` (via `[runtime] backend` or `--backend`):

```bash
uv run bus-rl model evaluate --config configs/eval.toml --manifest data/generated/base/manifest.json \
  --split validation --method threshold --output experiments/eval-rust
```

Running without the extension fails loudly. The deprecated Python oracle needs
an explicit opt-in and warns when used:

```bash
uv run bus-rl model evaluate --config configs/eval.toml --manifest data/generated/base/manifest.json \
  --split validation --method threshold --output experiments/eval-py \
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
