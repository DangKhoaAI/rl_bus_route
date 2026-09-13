"""Oracle manifest (R0.1): freeze the Python reference before any native port.

The manifest is a machine-checkable inventory: hashes, versions, seeds, the
observation/action/cost contract, and every known source-vs-spec discrepancy.
``verify_manifest`` re-derives the hashes so the manifest cannot silently rot.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bus_rl.config import AlgorithmConfig, ControlConfig
from bus_rl.execution.scenarios.generation import DEFAULT_COUNTS, SPLIT_SEEDS, generate_manifest
from bus_rl.provenance import (
    action_schema_hash,
    canonical_hash,
    file_hash,
    git_status,
    lock_hash,
    observation_schema_hash,
    physical_config_hash,
)
from bus_sim.oracle.actions import ACTION_TABLE, HEADWAYS
from bus_sim.oracle.costs import RewardConfig
from bus_sim.oracle.domain import SimConfig
from bus_sim.parity.snapshot import COST_FIELDS

MANIFEST_SCHEMA_VERSION = 1
ALGORITHM_SEEDS = (11, 22, 33)

ACTION_FAMILIES = ("NOOP", "DISPATCH", "REASSIGN", "SHORT_TURN", "RECALL", "SET_HEADWAY")

OBSERVATION_CONTRACT: dict[str, dict] = {
    "stops": {
        "shape": [4, 2, 8, 7],
        "dtype": "float32",
        "channels": [
            "queue count/40",
            "mean waiting age/2700s",
            "max waiting age/2700s",
            "count with age>=900s/40",
            "count boarded in last control interval/40",
            "count completed in last control interval/40",
            "count arrived in last control interval/40",
        ],
    },
    "arrival_history": {
        "shape": [4, 2, 8, 5],
        "dtype": "float32",
        "note": "5 past control bins, bin 0 = [t-Delta, t), scaled /40",
    },
    "forecast": {"shape": [4, 2, 8], "dtype": "float32", "note": "zeros when disabled"},
    "vehicles": {
        "shape": [16, 27],
        "dtype": "float32",
        "note": "phase 0:6, route 6:11, pattern 11:13, direction 13:15, node 15:17, "
        "target 17:22, scalars 22:27",
    },
    "routes": {
        "shape": [4, 8],
        "dtype": "float32",
        "note": "headway target/1200, two full-departure gaps/1200, full/16, incoming/16, "
        "short/16, target cooldown/600, depot count/16",
    },
    "stop_valid": {"shape": [4, 2, 8], "dtype": "float32"},
    "vehicle_valid": {"shape": [16], "dtype": "float32"},
    "route_valid": {"shape": [4], "dtype": "float32"},
    "context": {"shape": [3], "dtype": "float32", "note": "time/H, remaining/H, forecast enabled"},
}

DISCREPANCIES: list[dict] = [
    {
        "id": "D1-action-before-tick-order",
        "area": "tick ordering",
        "description": (
            "sim/engine.advance_interval applies the action before the first tick, so the "
            "decision precedes that tick's phase completion and arrivals. Base spec section "
            "4.1 lists completion and arrivals before the action at the boundary."
        ),
        "reproducer": "advance_interval(state, scenario, DISPATCH) then inspect the first "
        "advance_tick: the new DEADHEAD vehicle exists before arrivals are appended.",
        "port_decision": "Follow the code (matches Rust spec section 3.3). Do not reorder.",
    },
    {
        "id": "D2-terminal-boarding-at-idle",
        "area": "boarding timing",
        "description": (
            "The engine boards every TERMINAL_IDLE bus on every tick, not at the dispatcher "
            "departure instant. Base spec section 4.1 says terminal boarding happens at "
            "departure; Rust spec section 3.3 says 'board terminal-idle', matching the code."
        ),
        "reproducer": "park an empty FULL bus at an origin terminal with waiting riders and "
        "call advance_tick once: load increases without a departure event.",
        "port_decision": "Follow the code.",
    },
    {
        "id": "D3-departure-pattern-type",
        "area": "evaluation metrics",
        "description": (
            "state.departures records pattern.name (str) for pending_extra departures and "
            "pattern.value (int) for scheduled full departures. evaluation/runner filters "
            "`event.get('pattern') != 'FULL'`, so only extra departures (dispatched/reassigned/"
            "short) enter the headway gap statistics; scheduled FULL departures are dropped."
        ),
        "reproducer": "run any controller: most departures have pattern=0 and are skipped by "
        "the filter, so mean_headway_s reflects a subset.",
        "port_decision": "Reproduce the exact filter during R3 evaluator parity; raise as a "
        "metric revision separately.",
    },
    {
        "id": "D4-hardcoded-tick-seconds",
        "area": "passenger timestamps",
        "description": (
            "sim/passengers computes boarding/completion/abandonment tick indices with the "
            "literal 30 instead of config.tick_s."
        ),
        "reproducer": "construct SimConfig(tick_s=60); boarding_tick is current_time_s//30.",
        "port_decision": "Only tick_s=30 is supported; replicate the literal for parity.",
    },
    {
        "id": "D5-hardcoded-time-and-comfort-thresholds",
        "area": "costs and observation",
        "description": (
            "Excessive-wait threshold 900s and comfort capacity 30 are literals in "
            "rewards/costs.integrate_tick_costs and env/observation.observe; "
            "SimConfig.comfort_capacity is unused."
        ),
        "reproducer": "set comfort_capacity=0; crowding cost is unchanged.",
        "port_decision": "Replicate literals; keep config unused for parity.",
    },
    {
        "id": "D6-depot-feature-counts-all-idle",
        "area": "observation",
        "description": (
            "routes[:,7] is state.depot_count/16 (all DEPOT_IDLE) while base spec section 8.1 "
            "describes a cooldown-aware reserve-compatible count."
        ),
        "reproducer": "dispatch a reserve bus: it leaves DEPOT_IDLE and the feature drops "
        "immediately even before the cooldown matters.",
        "port_decision": "Follow the code.",
    },
    {
        "id": "D7-duplicate-vehicle-scalars",
        "area": "observation",
        "description": (
            "vehicles[:,26] equals vehicles[:,23] (both remaining_s/horizon); the documented "
            "'nominal time-to-terminal' channel is not distinct."
        ),
        "reproducer": "compare columns 23 and 26 over any rollout.",
        "port_decision": "Follow the code.",
    },
    {
        "id": "D8-arrival-history-off-by-one",
        "area": "observation",
        "description": (
            "arrival_history uses lag=(t-1-arrival_s)//control_interval, so bin 0 holds "
            "arrivals in [t-Delta, t). This is consistent with the recent-arrivals window but "
            "is an undocumented -1 offset."
        ),
        "reproducer": "arrival at tick 0 appears in lag 0 at t=120 and lag 1 at t=240.",
        "port_decision": "Replicate the formula exactly.",
    },
    {
        "id": "D9-observe-forecast-parameter-unused",
        "area": "observation/forecast",
        "description": (
            "env/observation.observe accepts a forecast argument but ignores it; the env "
            "injects the forecast tensor and enables context[2] afterwards."
        ),
        "reproducer": "call observe(state, scenario, forecast=ones) and inspect the output.",
        "port_decision": "Port the env-level contract (reset/step), not the dead parameter.",
    },
]


def observation_schema_contract() -> dict:
    return {
        "obs_version": 2,
        "keys": OBSERVATION_CONTRACT,
        "schema_hash": observation_schema_hash(),
        "notes": {
            "dtype": "all float32",
            "padding": "entity padding is zero and masked by *_valid",
            "scale": "count/load /40; time/age /horizon or 2700; headway /1200; cooldown /600",
        },
    }


def action_schema_contract() -> dict:
    counts: dict[str, int] = {name: 0 for name in ACTION_FAMILIES}
    for action in ACTION_TABLE:
        counts[action.kind] += 1
    assert sum(counts.values()) == 221
    return {
        "discrete_size": 221,
        "noop_index": 0,
        "family_counts": counts,
        "headway_choices_s": list(HEADWAYS),
        "schema_hash": action_schema_hash(),
        "encoding": "NOOP; 64 DISPATCH b-major/r-minor; 64 REASSIGN; 64 SHORT_TURN; "
        "16 RECALL; 12 SET_HEADWAY r-major/choice-minor",
    }


def terminal_behavior() -> dict:
    return {
        "reset": "WorldState at t=0, no passengers, fleet from scenario",
        "horizon": "terminated=True, truncated=False when current_time_s >= horizon_s",
        "settlement": "terminal_unfinished_count = waiting + onboard charged once at t=H",
        "reward": "r_t = -interval_cost(costs_t, reward)/n_ref; terminal penalty inside costs",
        "rng": "env.reset(seed=...) seeds gym np_random for scenario choice; "
        "options.scenario_index overrides; environment tapes are pre-generated, action-independent",
        "iteration_order": "waiting cohorts in insertion order; vehicles sorted by vehicle_id; "
        "dispatch tie-break by lowest vehicle_id; boarding FIFO by (arrival_tick, cohort_id)",
        "conservation": "CONSERVATION_CHECKS is False in training and True under the pytest autouse fixture",
    }


def _hash_tree(root: Path, pattern: str, *, required: bool) -> dict[str, dict]:
    entries = {}
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            entries[str(path.relative_to(root))] = {
                "sha256": file_hash(path),
                "required": required,
            }
    return entries


def inventory_checkpoints(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "runs").rglob("*.zip")):
        metadata_path = path.parent / "metadata.json"
        row = {
            "path": str(path.relative_to(root)),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
            "required": False,
            "note": "git-ignored historical run; documented here, regenerable",
            "has_metadata": metadata_path.exists(),
        }
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            row["metadata_sha256"] = file_hash(metadata_path)
            row["git_sha"] = metadata.get("git_sha")
            row["git_dirty"] = metadata.get("git_dirty")
            row["config_source"] = metadata.get("config_source")
            row["seed"] = metadata.get("seed")
            row["total_timesteps_actual"] = metadata.get("total_timesteps_actual")
            row["best_val_cost"] = metadata.get("best_val_cost")
            row["physical_config_hash"] = metadata.get("physical_config_hash")
            row["obs_schema_hash"] = metadata.get("obs_schema_hash")
            row["action_schema_hash"] = metadata.get("action_schema_hash")
        rows.append(row)
    return rows


# Only the migration contract narrative is hash-frozen. Analysis reports,
# figures and benchmark outputs live under `reports/` for readers but must not
# churn the oracle manifest when they are regenerated.
CONTRACT_REPORTS = ("reports/rust-migration.md",)


def inventory_reports(root: Path) -> list[dict]:
    rows = []
    for relative in CONTRACT_REPORTS:
        path = root / relative
        if path.exists():
            rows.append({"path": relative, "sha256": file_hash(path), "required": True})
        else:
            rows.append({"path": relative, "sha256": None, "required": True})
    return rows


def _load_reference(root: Path) -> dict | None:
    path = Path(root) / "tests" / "parity" / "reference" / "reference.json"
    return json.loads(path.read_text()) if path.exists() else None


def build_manifest(root: Path) -> dict:
    root = Path(root)
    sha, dirty = git_status(root)
    manifest_path = root / "data" / "generated" / "base" / "manifest.json"
    payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "role": "python-oracle",
        "git": {"sha": sha, "dirty": dirty},
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "lock": {"uv_lock": str(root / "uv.lock"), "sha256": lock_hash(root)},
        "physical_config": asdict(SimConfig()),
        "physical_config_hash": physical_config_hash(SimConfig()),
        "control_defaults": asdict(ControlConfig()),
        "reward_defaults": asdict(RewardConfig()),
        "algorithm_defaults": asdict(AlgorithmConfig()),
        "hashes": {
            "action_schema_hash": action_schema_hash(),
            "observation_schema_hash": observation_schema_hash(),
            "manifests": _hash_tree(root, "data/generated/**/manifest.json", required=False),
        },
        "scenario_order": {
            "split_seeds": {name: int(seed) for name, seed in SPLIT_SEEDS.items()},
            "split_counts": dict(DEFAULT_COUNTS),
            "generator": "bus_rl.execution.scenarios.generation.generate_manifest (SeedSequence.spawn)",
            "committed_manifest": str(manifest_path.relative_to(root))
            if manifest_path.exists()
            else None,
            "committed_manifest_sha256": file_hash(manifest_path)
            if manifest_path.exists()
            else None,
            "regenerable": True,
        },
        "seeds": {
            "algorithm": list(ALGORITHM_SEEDS),
            "split_generator": {name: int(seed) for name, seed in SPLIT_SEEDS.items()},
        },
        "contract": {
            "action": action_schema_contract(),
            "observation": observation_schema_contract(),
            "cost_components": list(COST_FIELDS),
            "terminal": terminal_behavior(),
        },
        "checkpoints": inventory_checkpoints(root),
        "reports": inventory_reports(root),
        "reference_checkpoint": _load_reference(root),
        "discrepancies": DISCREPANCIES,
    }
    payload["self_hash"] = canonical_hash({k: v for k, v in payload.items() if k != "self_hash"})
    return payload


def _verify_scenario_regeneration(payload: dict) -> list[str]:
    problems = []
    for split, seed in payload["scenario_order"]["split_seeds"].items():
        count = payload["scenario_order"]["split_counts"][split]
        scenarios = generate_manifest(split, count, SimConfig())
        seeds = {scenario.seed for scenario in scenarios}
        if len(seeds) != count:
            problems.append(f"split {split}: regenerated {len(seeds)} unique seeds != {count}")
        expected = SPLIT_SEEDS[split]
        if expected != seed:
            problems.append(f"split {split}: seed constant drifted")
    return problems


def verify_manifest(payload: dict, root: Path, *, regenerate_scenarios: bool = False) -> list[str]:
    """Return a list of verification problems (empty means the manifest holds).

    ``regenerate_scenarios`` re-runs the 1,200-day scenario generator to prove
    deterministic ordering. That is the R0 acceptance check; it is opt-in
    because it dominates verify time and only needs to run when the generator or
    physical config changes.
    """
    root = Path(root)
    problems: list[str] = []
    if payload.get("physical_config_hash") != physical_config_hash(SimConfig()):
        problems.append("physical_config_hash drifted")
    if payload["hashes"]["action_schema_hash"] != action_schema_hash():
        problems.append("action_schema_hash drifted")
    if payload["hashes"]["observation_schema_hash"] != observation_schema_hash():
        problems.append("observation_schema_hash drifted")
    if payload["lock"]["sha256"] != lock_hash(root):
        problems.append("uv.lock hash drifted")
    for relative, entry in payload["hashes"]["manifests"].items():
        path = root / relative
        if not path.exists():
            if entry.get("required"):
                problems.append(f"missing manifest: {relative}")
        elif file_hash(path) != entry["sha256"]:
            problems.append(f"manifest hash drifted: {relative}")
    for row in payload["checkpoints"]:
        path = root / row["path"]
        if not path.exists():
            if row.get("required"):
                problems.append(f"missing checkpoint: {row['path']}")
        elif file_hash(path) != row["sha256"]:
            problems.append(f"checkpoint hash drifted: {row['path']}")
    for row in payload["reports"]:
        path = root / row["path"]
        if not path.exists():
            if row.get("required"):
                problems.append(f"missing report: {row['path']}")
        elif file_hash(path) != row["sha256"]:
            problems.append(f"report hash drifted: {row['path']}")
    reference = payload.get("reference_checkpoint")
    if reference:
        path = root / reference["checkpoint"]
        if not path.exists():
            problems.append(f"missing reference checkpoint: {reference['checkpoint']}")
        elif file_hash(path) != reference["checkpoint_sha256"]:
            problems.append("reference checkpoint hash drifted")
    if regenerate_scenarios:
        problems.extend(_verify_scenario_regeneration(payload))
    return problems
