"""Scenario/environment helpers shared by the parity tests."""

from __future__ import annotations

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_sim.parity.scenarios import build_scenario, run_config


def new_env(spec: dict) -> BusDispatchEnv:
    """Build a reset Python oracle env from a catalog spec."""
    scenario = build_scenario(spec)
    run = run_config(spec)
    env = BusDispatchEnv([scenario], run.physical, reward=run.reward, control=run.control)
    env.reset(seed=0, options={"scenario_index": 0})
    return env
