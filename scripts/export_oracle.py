#!/usr/bin/env python
"""R0 oracle tooling: freeze the Python reference and export golden fixtures.

Usage:
    python scripts/export_oracle.py build [--force]
    python scripts/export_oracle.py verify

``build`` writes ``reports/rust-migration/oracle-manifest.json`` and exports the
named fixtures under ``tests/backend_parity/fixtures``. ``verify`` re-derives
every hash, replays every fixture, and re-checks the reference checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bus_rl.config import ControlConfig, load_run_config
from bus_rl.data.scenario import generate_manifest
from bus_rl.env.bus_dispatch import BusDispatchEnv
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.parity.fixtures import (
    fixtures_summary,
    flatten,
    load_fixture,
    record,
    replay_fixture,
    save_fixture,
)
from bus_rl.parity.manifest import build_manifest, verify_manifest
from bus_rl.parity.scenarios import CATALOG
from bus_rl.parity.snapshot import compare_records
from bus_rl.provenance import file_hash
from bus_rl.rewards.costs import RewardConfig
from bus_rl.timing import TIMERS
from bus_rl.training.checkpoint import load_metadata, load_model

REPORT_DIR = ROOT / "reports" / "rust-migration"
FIXTURES_DIR = ROOT / "tests" / "backend_parity" / "fixtures"
REFERENCE_DIR = ROOT / "tests" / "backend_parity" / "reference"
REFERENCE_SOURCE = ROOT / "runs" / "diagnose-after"


def build_reference(force: bool = False) -> dict:
    """Copy the historical diagnose-after checkpoint and re-measure 15471.25."""
    target = REFERENCE_DIR / "diagnose-after"
    if target.exists() and not force:
        existing = json.loads((REFERENCE_DIR / "reference.json").read_text())
        print(f"[reference] existing reference kept ({existing['mean_total_cost']})")
        return existing
    if not REFERENCE_SOURCE.exists():
        raise SystemExit(
            f"historical reference checkpoint not found: {REFERENCE_SOURCE}; "
            "run the diagnose-after training or pass a different source"
        )
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(REFERENCE_SOURCE / "last.zip", target / "last.zip")
    shutil.copy2(REFERENCE_SOURCE / "metadata.json", target / "metadata.json")

    TIMERS.enabled = False
    run = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    scenarios = generate_manifest("validation", 100)[:10]
    checkpoint = target / "last.zip"
    metadata = load_metadata(checkpoint)
    run = replace(
        run,
        control=ControlConfig(**metadata["control"]),
        reward=RewardConfig(
            **{
                key: metadata["reward"][key]
                for key in RewardConfig.__dataclass_fields__
                if key in metadata["reward"]
            }
        ),
    )
    env = BusDispatchEnv(scenarios[:1], run.physical, reward=run.reward, control=run.control)
    model, metadata = load_model(checkpoint, env, run.physical)
    frame, _ = evaluate_scenarios(
        scenarios, run, "ppo", model=model, model_seed=metadata.get("seed", 11)
    )
    payload = {
        "source": str(REFERENCE_SOURCE.relative_to(ROOT)),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": file_hash(checkpoint),
        "metadata_sha256": file_hash(target / "metadata.json"),
        "config": "configs/experiments/core.toml",
        "split": "validation",
        "days": 10,
        "scenario_source": "bus_rl.data.scenario.generate_manifest('validation', 100)[:10]",
        "seed": metadata.get("seed"),
        "total_timesteps_actual": metadata.get("total_timesteps_actual"),
        "historical_mean_total_cost": 15471.25,
        "mean_total_cost": float(frame["total_cost"].mean()),
        "mean_total_cost_core": float(frame["total_cost_core"].mean()),
        "git_sha": metadata.get("git_sha"),
        "git_dirty": metadata.get("git_dirty"),
        "note": "historical reference is reusable because checkpoint/config/days are recoverable",
    }
    if abs(payload["mean_total_cost"] - payload["historical_mean_total_cost"]) > 1e-6:
        raise SystemExit(
            f"reference checkpoint no longer reproduces 15471.25: {payload['mean_total_cost']}"
        )
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    (REFERENCE_DIR / "reference.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[reference] reproduced mean cost {payload['mean_total_cost']}")
    return payload


def coverage_matrix() -> dict:
    from bus_rl.control.actions import ACTION_TABLE

    matrix = {}
    for name in CATALOG:
        directory = FIXTURES_DIR / name
        payload = json.loads((directory / "input.json").read_text())
        events = json.loads((directory / "expected_events.json").read_text())
        kinds = [ACTION_TABLE[action].kind for action in payload["actions"]]
        counts: dict[str, int] = {}
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
        matrix[name] = {
            "steps": payload["num_steps"],
            "action_counts": counts,
            "families_seen": sorted(counts),
            "departures": len(events["departures"]),
            "control": payload["control"],
        }
    return matrix


def export_fixtures(force: bool) -> dict:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in CATALOG.items():
        directory = FIXTURES_DIR / name
        fresh = record(spec)
        fresh_flat = flatten(fresh)
        if directory.exists() and not force:
            _, expected, _ = load_fixture(directory)
            divergences = compare_records(expected, fresh_flat, limit=5)
            if divergences:
                raise SystemExit(f"fixture {name} no longer reproduces: {divergences[:3]}")
            results[name] = "verified"
            print(f"[fixture] {name:16s} verified")
            continue
        if directory.exists():
            shutil.rmtree(directory)
        save_fixture(fresh, directory)
        results[name] = "written"
        print(f"[fixture] {name:16s} written")
    return results


def write_summary(matrix: dict) -> None:
    payload = {
        "fixtures": fixtures_summary(FIXTURES_DIR),
        "coverage": matrix,
        "tolerances": {
            "obs_rtol": 1e-6,
            "obs_atol": 1e-6,
            "cost_rtol": 1e-9,
            "cost_atol": 1e-9,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "fixtures-summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def cmd_build(args) -> None:
    build_reference(force=args.force)
    export_fixtures(force=args.force)
    manifest = build_manifest(ROOT)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "oracle-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_summary(coverage_matrix())
    print(f"[manifest] written to {(REPORT_DIR / 'oracle-manifest.json').relative_to(ROOT)}")


def cmd_verify(args) -> None:
    manifest = json.loads((REPORT_DIR / "oracle-manifest.json").read_text())
    problems = verify_manifest(manifest, ROOT)
    failures = 0
    for name in CATALOG:
        directory = FIXTURES_DIR / name
        input_payload, expected, _ = load_fixture(directory)
        replayed = replay_fixture(input_payload)
        divergences = compare_records(expected, replayed, limit=5)
        if divergences:
            failures += 1
            problems.append(f"fixture {name} diverged: {divergences[0]}")
    reference = json.loads((REFERENCE_DIR / "reference.json").read_text())
    run = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    scenarios = generate_manifest("validation", 100)[:10]
    checkpoint = ROOT / reference["checkpoint"]
    metadata = load_metadata(checkpoint)
    run = replace(
        run,
        control=ControlConfig(**metadata["control"]),
        reward=RewardConfig(
            **{
                key: metadata["reward"][key]
                for key in RewardConfig.__dataclass_fields__
                if key in metadata["reward"]
            }
        ),
    )
    env = BusDispatchEnv(scenarios[:1], run.physical, reward=run.reward, control=run.control)
    model, metadata = load_model(checkpoint, env, run.physical)
    frame, _ = evaluate_scenarios(
        scenarios, run, "ppo", model=model, model_seed=metadata.get("seed", 11)
    )
    mean = float(frame["total_cost"].mean())
    if abs(mean - reference["historical_mean_total_cost"]) > 1e-6:
        problems.append(f"reference mean drifted: {mean}")
    if problems:
        print("VERIFY FAILED")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(1)
    print(f"VERIFY OK: {len(CATALOG)} fixtures, manifest hashes, reference mean={mean}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=cmd_build)
    verify = sub.add_parser("verify")
    verify.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
