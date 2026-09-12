"""Python adapter that drives the native `bus_sim` kernel and shapes its output
into the same flat records used by the Python golden fixtures.

This is test/report tooling, not the R3 Gym wrapper.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache

import numpy as np

from bus_rl.backend.native import build_kernel as _build_kernel
from bus_rl.backend.native import scenario_payload
from bus_rl.parity.fixtures import flatten
from bus_rl.parity.scenarios import CONTROL_FLAGS, build_scenario
from bus_rl.parity.snapshot import COST_FIELDS


def scenario_json(scenario, control) -> str:
    enable_reassign, enable_short_turn = control
    return scenario_payload(scenario, enable_reassign, enable_short_turn)


@lru_cache(maxsize=64)
def _scenario_for_key(spec_json: str):
    return build_scenario(json.loads(spec_json))


def cached_scenario(spec: dict):
    """Build each catalog scenario once per test session (deterministic)."""
    return _scenario_for_key(json.dumps(spec, sort_keys=True, default=str))


def make_kernel(spec: dict, *, conservation: bool = True):
    scenario = cached_scenario(spec)
    enable_reassign, enable_short_turn = CONTROL_FLAGS[spec.get("control", "M3")]
    scenario = replace(
        scenario,
        enable_reassign=enable_reassign,
        enable_short_turn=enable_short_turn,
    )
    kernel = _build_kernel(scenario, conservation=conservation)
    kernel.reset()
    return kernel


def run_actions(spec: dict, actions: list[int], *, conservation: bool = True) -> dict:
    """Drive the kernel and return a `flatten`-compatible record."""
    kernel = make_kernel(spec, conservation=conservation)
    boundaries = [kernel.debug_snapshot()]
    ticks: list[dict] = []
    departures: list[dict] = []
    accepted: list[dict] = []
    steps = [_step_record(0, boundaries[0], 0.0, False, [0.0] * len(COST_FIELDS))]
    for index, action in enumerate(actions, start=1):
        result = kernel.debug_step(action)
        for tick in result["ticks"]:
            ticks.append(
                {
                    "state": {
                        "time_s": np.int64(tick["time_s"]),
                        "counters": np.asarray(tick["counters"], dtype=np.int64),
                        "vehicles": np.asarray(tick["vehicles"], dtype=np.int64),
                    },
                    "costs": np.asarray(tick["costs"], dtype=np.float64),
                    "event": np.asarray(tick["event"], dtype=np.int64),
                }
            )
        boundaries.append(kernel.debug_snapshot())
        departures.extend({"step": index, **row} for row in result["departures"])
        accepted.extend({"step": index, **row} for row in result["accepted_actions"])
        steps.append(
            _step_record(
                index,
                boundaries[-1],
                float(result["reward"]),
                bool(result["terminated"]),
                result["costs"],
            )
        )
    obs_keys: list[str] = []
    return {
        "obs_keys": obs_keys,
        "steps": steps,
        "ticks": ticks,
        "departures": departures,
        "accepted_actions": accepted,
    }


def _step_record(index, snapshot, reward, terminated, costs) -> dict:
    state = {key: snapshot[key] for key in snapshot if key != "mean_wait"}
    return {
        "step": index,
        "obs": {},
        "mask": np.zeros(221, dtype=bool),
        "reward": reward,
        "terminated": terminated,
        "truncated": False,
        "costs": np.asarray(costs, dtype=np.float64),
        "state": {
            "time_s": np.int64(state["time_s"]),
            "counters": np.asarray(state["counters"], dtype=np.int64),
            "vehicles": np.asarray(state["vehicles"], dtype=np.int64),
            "cohorts": np.asarray(state["cohorts"], dtype=np.int64),
            "cohort_kind": np.asarray(state["cohort_kind"], dtype=np.int64),
            "headway_targets": np.asarray(state["headway_targets"], dtype=np.int64),
            "headway_changed_at": np.asarray(state["headway_changed_at"], dtype=np.int64),
            "departure_keys": np.asarray(state["departure_keys"], dtype=np.int64),
            "last_departure": np.asarray(state["last_departure"], dtype=np.int64),
            "last_full_departure": np.asarray(state["last_full_departure"], dtype=np.int64),
            "terminal_settled": np.int64(state["terminal_settled"]),
        },
        "mean_wait": snapshot["mean_wait"],
        "events": {},
    }


def kernel_flat(spec: dict, actions: list[int], *, conservation: bool = True) -> dict:
    return flatten(run_actions(spec, actions, conservation=conservation))


# Observation keys in the frozen oracle order; mirror
# `bus_rl.env.observation.observe`.
OBS_KEYS = (
    "stops",
    "arrival_history",
    "forecast",
    "vehicles",
    "routes",
    "stop_valid",
    "vehicle_valid",
    "route_valid",
    "context",
)


def replay_observations(
    spec: dict, actions: list[int], *, conservation: bool = True
) -> tuple[list[dict], list[np.ndarray]]:
    """Replay ``actions`` and return observations/masks at every boundary."""
    kernel = make_kernel(spec, conservation=conservation)
    observations = [{key: np.asarray(value) for key, value in kernel.observe().items()}]
    masks = [np.asarray(kernel.action_mask(), dtype=bool)]
    for action in actions:
        kernel.debug_step(int(action))
        observations.append({key: np.asarray(value) for key, value in kernel.observe().items()})
        masks.append(np.asarray(kernel.action_mask(), dtype=bool))
    return observations, masks


# Keys the R1 kernel owns and must match the oracle exactly. Observation and
# mask parity belong to R2.
KERNEL_COMPARE_KEYS = (
    (
        "time_s",
        "counters",
        "vehicles",
        "cohort_unique",
        "cohort_order",
        "cohort_ptr",
        "headway_targets",
        "headway_changed_at",
        "departure_keys",
        "last_departure",
        "last_full_departure",
        "terminal_settled",
        "reward",
        "terminated",
        "truncated",
        "mean_wait",
        "tick__vehicles",
        "tick__counters",
        "tick__time_s",
        "tick__events",
    )
    + tuple(f"costs__{field}" for field in COST_FIELDS)
    + tuple(f"tick__costs__{field}" for field in COST_FIELDS)
)
