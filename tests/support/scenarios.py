"""Scenario/environment helpers shared by the parity tests."""

from __future__ import annotations

import json
from functools import lru_cache

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_sim.parity.scenarios import build_scenario, run_config


@lru_cache(maxsize=64)
def _scenario_for_key(spec_json: str):
    return build_scenario(json.loads(spec_json))


def cached_scenario(spec: dict):
    return _scenario_for_key(json.dumps(spec, sort_keys=True, default=str))


def new_env(spec: dict) -> BusDispatchEnv:
    """Build a reset Python oracle env from a catalog spec."""
    scenario = build_scenario(spec)
    run = run_config(spec)
    env = BusDispatchEnv([scenario], run.physical, reward=run.reward, control=run.control)
    env.reset(seed=0, options={"scenario_index": 0})
    return env
