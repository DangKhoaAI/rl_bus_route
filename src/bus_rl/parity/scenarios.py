"""Self-contained scenario catalog for oracle fixtures.

Fixtures must be reproducible from a fresh checkout, so every scenario here is
built from ``generate_scenario`` using only a seed/variant/physical override --
no dependence on the (git-ignored) ``data/generated`` manifests.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from bus_rl.config import AlgorithmConfig, ControlConfig, ForecastConfig, RunConfig
from bus_rl.data.scenario import generate_scenario
from bus_rl.domain import Scenario, SimConfig, scenario_digest
from bus_rl.rewards.costs import RewardConfig

CONTROL_FLAGS = {
    "M1": (False, False),
    "M2": (True, False),
    "M3": (True, True),
}


def _with_arrivals(scenario: Scenario, arrivals: np.ndarray, seed: int | None = None) -> Scenario:
    values = np.asarray(arrivals)
    if values.size and (int(values.max()) > 127 or int(values.min()) < 0):
        raise ValueError("arrival counts outside the int8 range")
    arrivals = np.ascontiguousarray(values, dtype=np.int8)
    arrivals.setflags(write=False)
    seed = scenario.seed if seed is None else seed
    digest = scenario_digest(
        scenario.config, scenario.network, scenario.fleet, arrivals, scenario.traffic_tape, seed
    )
    return replace(scenario, arrival_tape=arrivals, scenario_hash=digest, seed=seed)


def _physical(overrides: dict | None) -> SimConfig:
    payload = asdict(SimConfig())
    payload.update(overrides or {})
    return SimConfig(**payload)


def build_scenario(spec: dict) -> Scenario:
    """Build a scenario from a compact, JSON-serializable descriptor."""
    kind = spec["kind"]
    seed = int(spec["seed"])
    config = _physical(spec.get("physical"))
    variant = spec.get("variant", "base")
    if kind in {"base", "burst", "traffic"}:
        return generate_scenario(seed, config, variant=variant)
    if kind == "empty":
        scenario = generate_scenario(seed, config)
        return _with_arrivals(scenario, np.zeros_like(scenario.arrival_tape))
    if kind == "peak":
        scenario = generate_scenario(seed, config)
        return _with_arrivals(scenario, scenario.arrival_tape.astype(np.int32) * 3)
    if kind == "flood":
        scenario = generate_scenario(seed, config)
        return _with_arrivals(scenario, scenario.arrival_tape.astype(np.int32) * 20)
    if kind == "capacity":
        scenario = generate_scenario(seed, config)
        arrivals = np.array(scenario.arrival_tape, dtype=np.int32, copy=True)
        stops = config.stops_per_route
        # One oversized queue at the first stop of route 0: 45 > capacity 40.
        arrivals[0, 0, 0, 0, stops - 1] = 45
        arrivals[1, 0, 0, 0, stops - 1] = 45
        return _with_arrivals(scenario, arrivals)
    raise ValueError(f"unknown scenario kind: {kind}")


# Named fixtures: each descriptor is part of the oracle contract and is hashed
# into the manifest. ``control`` selects the M1/M2/M3 feature flags.
CATALOG: dict[str, dict] = {
    "zero_m3": {"kind": "empty", "seed": 2001, "control": "M3", "controller": "coverage"},
    "normal_m1": {"kind": "base", "seed": 2001, "control": "M1", "controller": "coverage"},
    "normal_m2": {"kind": "base", "seed": 2001, "control": "M2", "controller": "coverage"},
    "normal_m3": {"kind": "base", "seed": 2001, "control": "M3", "controller": "coverage"},
    "normal_m3_random": {
        "kind": "base",
        "seed": 2001,
        "control": "M3",
        "controller": "random",
        "controller_seed": 7,
    },
    "peak_m3": {"kind": "peak", "seed": 2001, "control": "M3", "controller": "coverage"},
    "burst_m3": {
        "kind": "burst",
        "seed": 4001,
        "control": "M3",
        "controller": "coverage",
        "variant": "burst",
    },
    "traffic_m3": {
        "kind": "traffic",
        "seed": 5001,
        "control": "M3",
        "controller": "coverage",
        "variant": "traffic",
    },
    "capacity_m3": {"kind": "capacity", "seed": 2001, "control": "M3", "controller": "coverage"},
    "backlog_m3": {
        "kind": "flood",
        "seed": 2001,
        "control": "M3",
        "controller": "coverage",
        "physical": {"patience_s": 20000},
    },
    "abandon_m3": {
        "kind": "base",
        "seed": 2001,
        "control": "M3",
        "controller": "coverage",
        "physical": {"patience_s": 300},
    },
}


def run_config(
    spec: dict,
    *,
    reward: RewardConfig | None = None,
    algorithm: AlgorithmConfig | None = None,
) -> RunConfig:
    enable_reassign, enable_short_turn = CONTROL_FLAGS[spec.get("control", "M3")]
    return RunConfig(
        physical=_physical(spec.get("physical")),
        control=ControlConfig(enable_reassign=enable_reassign, enable_short_turn=enable_short_turn),
        reward=reward or RewardConfig(),
        algorithm=algorithm or AlgorithmConfig(),
        forecast=ForecastConfig(enabled=False),
        source=f"oracle:{spec.get('control', 'M3')}",
    )


def catalog_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / "tests" / "backend_parity" / "fixtures"
