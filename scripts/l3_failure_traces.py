#!/usr/bin/env python
"""Trace the reproducibly-selected highest-cost L3 held-out days for core.

Selection rule (fixed before looking at the traces): for each held-out split and
each core seed, take the three days with the highest ``total_cost_core`` at that
checkpoint, breaking ties by ascending ``scenario_index``.

Raw decision traces are large, so they are written to the gitignored
``runs/rl-improvement/l3-failure-traces/`` tree; the committed evidence is a
compact digest with action-family counts, peak queues and the trace path.

Run:  uv run python scripts/l3_failure_traces.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from bus_rl.config import ControlConfig, load_run_config
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.scenarios.io import load_split
from bus_rl.learning.training.checkpoint import load_metadata, load_model
from bus_sim.oracle.costs import RewardConfig

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "rl-improvement"
RAW = RUNS / "l3-failure-traces"
EVIDENCE = ROOT / "reports" / "rl-improvement" / "evidence"
MANIFEST = ROOT / "data" / "generated" / "base" / "manifest.json"
CONFIG = ROOT / "configs" / "experiments" / "rl-improvement" / "core.toml"

SPLITS = ["test_id", "test_ood_burst", "test_ood_traffic"]
SEEDS = [11, 22, 33]
N_DAYS = 3


def _reward_from_metadata(metadata: dict) -> RewardConfig:
    allowed = RewardConfig.__dataclass_fields__
    return RewardConfig(**{k: metadata["reward"][k] for k in allowed if k in metadata["reward"]})


def trace_seed(split: str, seed: int) -> dict:
    run = load_run_config(CONFIG)
    eval_dir = RUNS / "l3-eval-fixed" / "core" / str(seed) / split
    results = pd.read_csv(eval_dir / "results.csv")
    selected = results.sort_values(
        ["total_cost_core", "scenario_index"], ascending=[False, True]
    ).head(N_DAYS)
    scenarios = load_split(MANIFEST, split)

    checkpoint = RUNS / "l3-t9" / "core" / str(seed) / "train-20260913-r2" / "best.zip"
    metadata = load_metadata(checkpoint)
    run = replace(
        run,
        control=ControlConfig(**metadata["control"]),
        reward=_reward_from_metadata(metadata),
        forecast=replace(run.forecast, enabled=bool(metadata.get("forecast_enabled"))),
    )
    env = make_env_for_run(scenarios[:1], run)
    model, _ = load_model(checkpoint, env, run.physical, backend=run.runtime.backend)

    indices = [int(i) for i in selected.scenario_index]
    subset = [scenarios[i] for i in indices]
    frame, traces = evaluate_scenarios(
        subset, run, "ppo", model=model, model_seed=seed, split=split, trace_all=True
    )
    assert len(frame) == len(subset), "trace evaluation lost days"

    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{split}-seed{seed}.json"
    raw_path.write_text(json.dumps(traces, indent=2, default=str))

    digest = []
    for position, original_index in enumerate(indices):
        rows = traces.get(position, [])
        families: dict[str, int] = {}
        for row in rows:
            action = row.get("action", "unknown")
            families[action] = families.get(action, 0) + 1
        digest.append(
            {
                "scenario_index": original_index,
                "scenario_seed": int(selected.iloc[position].scenario_seed),
                "scenario_hash": str(selected.iloc[position].scenario_hash),
                "total_cost_core": float(selected.iloc[position].total_cost_core),
                "decisions": len(rows),
                "action_families": families,
                "peak_queue": int(max((max(r.get("queues", [0])) for r in rows), default=0)),
                "raw_trace": str(raw_path.relative_to(ROOT)),
            }
        )
    return {"split": split, "seed": seed, "days": digest}


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection": (
            "for each held-out split and core seed, the three highest total_cost_core days at "
            "that checkpoint; ties broken by ascending scenario_index"
        ),
        "checkpoint_timestep": "lowest validation total_cost_core; earliest tie",
        "raw_trace_root": str(RAW.relative_to(ROOT)),
        "traces": [],
    }
    for split in SPLITS:
        for seed in SEEDS:
            payload["traces"].append(trace_seed(split, seed))
    (EVIDENCE / "l3_failure_traces.json").write_text(json.dumps(payload, indent=2))
    total = sum(len(entry["days"]) for entry in payload["traces"])
    print(f"traced {total} days across {len(payload['traces'])} checkpoints -> {EVIDENCE}")


if __name__ == "__main__":
    main()
