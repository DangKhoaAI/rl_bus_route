# bus-rl

CPU simulator and MaskablePPO controllers for dynamic bus fleet control
(reserve dispatch, headway targets, reassign, short-turn) on a fixed 3-route
network. Specification: `docs/spec.md`. Plan: `docs/plan.md`.

Python 3.11, Linux, CPU PyTorch. This is a synthetic operational-control study,
not a city deployment or a new-route design tool.

## Setup

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Tiny smoke (2 train / 1 val / 1 test days, 32 PPO steps) lives in
`tests/test_pipeline.py`.

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

## What is in the observation

Controllers see queues, ages, loads, past arrivals, headways, and fleet
status. They do not see future demand tapes, scenario seeds, or latent
passenger destinations. Forecasts, when enabled, are fit on train-day arrival
logs only.
