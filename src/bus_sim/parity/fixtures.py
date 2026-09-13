"""Golden fixture export/replay for the Rust migration (R0.2).

A fixture is a deterministic scenario + action trace + the Python oracle's
per-boundary and per-tick outputs. The same replay function will later be
pointed at the native backend, so the on-disk contract is backend-agnostic.

Step records and per-tick records come from the same public ``BusDispatchEnv``
pass: ``step`` accepts an ``on_tick`` trace hook, so the fixture is recorded in
a single run (no mirror pass to drift).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.provenance import action_schema_hash, physical_config_hash
from bus_sim.oracle.costs import mean_waiting_minutes
from bus_sim.oracle.domain import StepCosts
from bus_sim.parity.controllers import make_controller
from bus_sim.parity.scenarios import build_scenario, run_config
from bus_sim.parity.snapshot import (
    COHORT_FIELDS,
    COST_FIELDS,
    VEHICLE_FIELDS,
    counter_array,
    mask_snapshot,
    numpy_state_snapshot,
    vehicle_array,
)

FIXTURE_SCHEMA_VERSION = 1
MAX_STEPS = 200


def _costs_array(costs: StepCosts) -> np.ndarray:
    return np.array([getattr(costs, name) for name in COST_FIELDS], dtype=np.float64)


def _event_row(entry: dict) -> list[int]:
    return [
        int(entry["time_s"]),
        int(entry["boarded"]),
        int(entry["denied"]),
        int(entry["abandoned"]),
    ]


def _tick_state(state) -> dict:
    """Tick rows only need vehicles/counters/clock; `flatten` drops the rest."""
    return {
        "time_s": np.array(state.current_time_s, dtype=np.int64),
        "counters": counter_array(state),
        "vehicles": vehicle_array(state),
    }


def _boundary(env, step_index, obs, mask, reward, terminated, truncated, costs, events) -> dict:
    return {
        "step": step_index,
        "obs": {key: np.asarray(value) for key, value in obs.items()},
        "mask": mask_snapshot(mask),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "costs": _costs_array(costs),
        "state": numpy_state_snapshot(env.state),
        "mean_wait": mean_waiting_minutes(env.state, env.scenario.config.tick_s),
        "events": events,
    }


def _new_env(spec: dict) -> BusDispatchEnv:
    scenario = build_scenario(spec)
    run = run_config(spec)
    env = BusDispatchEnv([scenario], run.physical, reward=run.reward, control=run.control)
    env.reset(seed=0, options={"scenario_index": 0})
    return env


def new_env(spec: dict) -> BusDispatchEnv:
    """Public constructor used by parity tests."""
    return _new_env(spec)


def _tick_recorder():
    """Return ``(rows, capture)``; ``capture`` snapshots the world after a tick."""
    rows: list[dict] = []

    def capture(state, tick_costs, event) -> None:
        rows.append(
            {
                "state": _tick_state(state),
                "costs": _costs_array(tick_costs),
                "event": _event_row(event),
            }
        )

    return rows, capture


def record(spec: dict, *, max_steps: int = MAX_STEPS) -> dict:
    """Run the oracle public API once, capturing per-tick state via ``on_tick``."""
    env = _new_env(spec)
    controller = make_controller(spec.get("controller", "coverage"), spec.get("controller_seed", 0))
    observation = env._observation()
    mask = env.action_masks()
    ticks, on_tick = _tick_recorder()

    steps: list[dict] = [
        _boundary(env, 0, observation, mask, 0.0, False, False, StepCosts(), {"ticks": []})
    ]
    actions: list[int] = []
    departure_events: list[dict] = []
    accepted_events: list[dict] = []

    step_index = 1
    terminated = truncated = False
    while step_index <= max_steps:
        action = controller.act(observation, mask)
        before_events = len(env.state.event_log)
        before_departures = len(env.state.departures)
        before_accepted = len(env.state.accepted_actions)
        observation, reward, terminated, truncated, info = env.step(action, on_tick=on_tick)
        actions.append(int(action))
        events = {
            "ticks": [_event_row(row) for row in env.state.event_log[before_events:]],
        }
        steps.append(
            _boundary(
                env,
                step_index,
                observation,
                env.action_masks(),
                reward,
                terminated,
                truncated,
                info["costs"],
                events,
            )
        )
        departure_events.extend(
            {"step": step_index, **row} for row in env.state.departures[before_departures:]
        )
        accepted_events.extend(
            {"step": step_index, **row} for row in env.state.accepted_actions[before_accepted:]
        )
        mask = env.action_masks()
        step_index += 1
        if terminated or truncated:
            break

    return {
        "spec": spec,
        "scenario_hash": env.scenario.scenario_hash,
        "physical_config_hash": physical_config_hash(env.scenario.config),
        "control": asdict(run_config(spec).control),
        "reward": asdict(run_config(spec).reward),
        "action_schema_hash": action_schema_hash(),
        "actions": actions,
        "steps": steps,
        "ticks": ticks,
        "departures": departure_events,
        "accepted_actions": accepted_events,
        "obs_keys": list(steps[0]["obs"]),
    }


def _cohort_frames(steps: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact per-step cohort state: unique rows + order index per step.

    Cohort rows repeat across steps, so only first-appearance rows are stored;
    each boundary keeps an index list that preserves iteration order.
    """
    unique: dict[tuple, int] = {}
    rows: list[tuple] = []
    order_all: list[int] = []
    ptr: list[int] = [0]
    for step in steps:
        cohorts = step["state"]["cohorts"]
        kinds = step["state"]["cohort_kind"]
        for index in range(len(cohorts)):
            key = (int(kinds[index]),) + tuple(int(value) for value in cohorts[index])
            slot = unique.get(key)
            if slot is None:
                slot = len(rows)
                unique[key] = slot
                rows.append(key)
            order_all.append(slot)
        ptr.append(len(order_all))
    unique_rows = (
        np.array(rows, dtype=np.int64).reshape(-1, len(COHORT_FIELDS) + 2)
        if rows
        else np.zeros((0, len(COHORT_FIELDS) + 2), np.int64)
    )
    return unique_rows, np.array(order_all, dtype=np.int64), np.array(ptr, dtype=np.int64)


def flatten(record: dict) -> dict[str, np.ndarray]:
    """Turn a nested record into flat arrays for ``np.savez``/comparison."""
    steps = record["steps"]
    ticks = record["ticks"]
    flat: dict[str, np.ndarray] = {}
    for key in record["obs_keys"]:
        flat[f"obs__{key}"] = np.stack([step["obs"][key] for step in steps])
    flat["mask"] = np.stack([step["mask"] for step in steps])
    flat["reward"] = np.array([step["reward"] for step in steps], dtype=np.float64)
    flat["terminated"] = np.array([step["terminated"] for step in steps], dtype=bool)
    flat["truncated"] = np.array([step["truncated"] for step in steps], dtype=bool)
    flat["mean_wait"] = np.array(
        [np.nan if step["mean_wait"] is None else step["mean_wait"] for step in steps],
        dtype=np.float64,
    )
    for index, field in enumerate(COST_FIELDS):
        flat[f"costs__{field}"] = np.array(
            [step["costs"][index] for step in steps], dtype=np.float64
        )
    for key in steps[0]["state"]:
        if key in {"cohorts", "cohort_kind"}:
            continue
        flat[key] = np.stack([step["state"][key] for step in steps])
    flat["cohort_unique"], flat["cohort_order"], flat["cohort_ptr"] = _cohort_frames(steps)
    flat["tick__vehicles"] = np.stack([row["state"]["vehicles"] for row in ticks])
    flat["tick__counters"] = np.stack([row["state"]["counters"] for row in ticks])
    flat["tick__time_s"] = np.array([row["state"]["time_s"] for row in ticks], dtype=np.int64)
    flat["tick__events"] = np.array([row["event"] for row in ticks], dtype=np.int64)
    for index, field in enumerate(COST_FIELDS):
        flat[f"tick__costs__{field}"] = np.array(
            [row["costs"][index] for row in ticks], dtype=np.float64
        )
    return flat


def save_fixture(record: dict, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    input_payload = {
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture": directory.name,
        "spec": record["spec"],
        "scenario_hash": record["scenario_hash"],
        "physical_config_hash": record["physical_config_hash"],
        "control": record["control"],
        "reward": record["reward"],
        "action_schema_hash": record["action_schema_hash"],
        "actions": record["actions"],
        "num_steps": len(record["actions"]),
        "obs_keys": record["obs_keys"],
        "cost_fields": list(COST_FIELDS),
        "vehicle_fields": list(VEHICLE_FIELDS),
        "cohort_fields": list(COHORT_FIELDS),
        "tolerances": {
            "obs_rtol": 1e-6,
            "obs_atol": 1e-6,
            "cost_rtol": 1e-9,
            "cost_atol": 1e-9,
        },
    }
    (directory / "input.json").write_text(json.dumps(input_payload, indent=2, sort_keys=True))
    np.savez_compressed(directory / "expected.npz", **flatten(record))
    events = {"departures": record["departures"], "accepted_actions": record["accepted_actions"]}
    (directory / "expected_events.json").write_text(json.dumps(events, indent=2, sort_keys=True))


def load_fixture(directory: Path) -> tuple[dict, dict[str, np.ndarray], dict]:
    directory = Path(directory)
    input_payload = json.loads((directory / "input.json").read_text())
    with np.load(directory / "expected.npz", allow_pickle=False) as arrays:
        expected = {key: arrays[key] for key in arrays.files}
    events = json.loads((directory / "expected_events.json").read_text())
    return input_payload, expected, events


def replay_fixture(input_payload: dict) -> dict[str, np.ndarray]:
    """Rebuild the trajectory from ``input.json`` without reading expected output."""
    return flatten(record(input_payload["spec"], max_steps=len(input_payload["actions"]) + 1))


def replay_explicit(spec: dict, actions: list[int]) -> dict[str, np.ndarray]:
    """Replay an explicit action list through the public env API."""
    env = _new_env(spec)
    ticks, on_tick = _tick_recorder()
    steps = [
        _boundary(
            env,
            0,
            env._observation(),
            env.action_masks(),
            0.0,
            False,
            False,
            StepCosts(),
            {"ticks": []},
        )
    ]
    for index, action in enumerate(actions, start=1):
        before = len(env.state.event_log)
        observation, reward, terminated, truncated, info = env.step(action, on_tick=on_tick)
        events = {"ticks": [_event_row(row) for row in env.state.event_log[before:]]}
        steps.append(
            _boundary(
                env,
                index,
                observation,
                env.action_masks(),
                reward,
                terminated,
                truncated,
                info["costs"],
                events,
            )
        )
        if terminated or truncated:
            break
    record = {
        "obs_keys": list(steps[0]["obs"]),
        "steps": steps,
        "ticks": ticks,
        "departures": [],
        "accepted_actions": [],
    }
    return flatten(record)


def fixtures_summary(root: Path) -> dict:
    root = Path(root)
    entries = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        payload = json.loads((directory / "input.json").read_text())
        entries[directory.name] = {
            "scenario_hash": payload["scenario_hash"],
            "control": payload["control"],
            "num_steps": payload["num_steps"],
            "expected_npz_bytes": (directory / "expected.npz").stat().st_size,
        }
    return entries
