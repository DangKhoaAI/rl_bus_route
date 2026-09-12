"""Deterministic episode evaluation and CSV metrics."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from bus_rl.baselines import (
    FixedController,
    ProportionalController,
    RandomValidController,
    ThresholdController,
)
from bus_rl.config import ControlConfig, RunConfig
from bus_rl.domain import PassengerStatus, StepCosts
from bus_rl.env.bus_dispatch import BusDispatchEnv
from bus_rl.rewards.costs import DEFAULT_REWARD, RewardConfig, add_costs, interval_cost
from bus_rl.timing import TIMERS


def make_controller(method: str, model=None, seed: int = 0):
    if method == "fixed":
        return FixedController()
    if method == "threshold":
        return ThresholdController()
    if method == "proportional":
        return ProportionalController()
    if method == "random":
        return RandomValidController(seed)
    if method in {"ppo", "maskable_ppo"}:
        if model is None:
            raise ValueError("ppo controller requires a loaded model")
        return PPOController(model)
    raise ValueError(f"unknown method: {method}")


class PPOController:
    def __init__(self, model):
        self.model = model

    def act(self, observation, mask):
        action, _ = self.model.predict(observation, action_masks=mask, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])


def _wait_minutes(cohort, state, tick_s: int) -> tuple[float, bool]:
    arrival_s = cohort.arrival_tick * tick_s
    if cohort.status is PassengerStatus.COMPLETED and cohort.boarding_tick is not None:
        return (cohort.boarding_tick * tick_s - arrival_s) / 60.0, False
    if cohort.status is PassengerStatus.ABANDONED and cohort.abandonment_tick is not None:
        return (cohort.abandonment_tick * tick_s - arrival_s) / 60.0, False
    if cohort.status is PassengerStatus.ONBOARD and cohort.boarding_tick is not None:
        return (cohort.boarding_tick * tick_s - arrival_s) / 60.0, False
    return (state.current_time_s - arrival_s) / 60.0, True


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize_episode(
    state,
    scenario,
    costs: StepCosts,
    reward: float,
    reward_config: RewardConfig | None = None,
) -> dict:
    tick_s = scenario.config.tick_s
    waits: list[float] = []
    censored_flags: list[bool] = []
    route_waits: dict[int, list[float]] = defaultdict(list)
    excessive = 0
    unique_denied = 0
    for cohort in state.iter_cohorts():
        wait, censored = _wait_minutes(cohort, state, tick_s)
        waits.extend([wait] * cohort.count)
        censored_flags.extend([censored] * cohort.count)
        route_waits[cohort.route_id].extend([wait] * cohort.count)
        if wait >= 15:
            excessive += cohort.count
        if cohort.first_denied:
            unique_denied += cohort.count
    generated = state.generated_count
    none = generated == 0
    mean_wait = None if none else float(np.mean(waits))
    p95_wait = None if none else _percentile(waits, 95)
    worst_route = None
    worst_mean = None
    for route_id, values in route_waits.items():
        route_mean = float(np.mean(values))
        if worst_mean is None or route_mean > worst_mean:
            worst_mean, worst_route = route_mean, route_id
    gaps: list[float] = []
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for event in state.departures:
        if event.get("pattern") != "FULL":
            continue
        grouped[int(event["route_id"]), int(event["direction"])].append(int(event["time_s"]))
    for times in grouped.values():
        ordered = sorted(times)
        gaps.extend(later - earlier for earlier, later in pairwise(ordered))
    kinds = [event["kind"] for event in state.accepted_actions]
    core_cost = interval_cost(costs, DEFAULT_REWARD)
    trained_cost = interval_cost(costs, reward_config or DEFAULT_REWARD)
    return {
        "scenario_seed": scenario.seed,
        "scenario_hash": scenario.scenario_hash,
        "generated": generated,
        "completed": state.completed_count,
        "abandoned": state.abandoned_count,
        "unfinished": state.waiting_count + state.onboard_count,
        "unique_denied": unique_denied,
        "waiting_pm": costs.waiting_pm,
        "onboard_pm": costs.onboard_pm,
        "crowding_pm": costs.crowding_pm,
        "active_bus_min": costs.active_bus_min,
        "deadhead_bus_min": costs.deadhead_bus_min,
        "excessive_wait_pm": costs.excessive_wait_pm,
        "first_denied_count": costs.first_denied_count,
        "abandoned_count": costs.abandoned_count,
        "mission_changes": costs.mission_changes,
        "terminal_unfinished_count": costs.terminal_unfinished_count,
        "total_cost": trained_cost,
        "total_cost_core": core_cost,
        "reward": reward,
        "mean_wait": mean_wait,
        "p95_wait": p95_wait,
        "completed_share": None if none else state.completed_count / generated,
        "abandoned_share": None if none else state.abandoned_count / generated,
        "unfinished_share": None
        if none
        else (state.waiting_count + state.onboard_count) / generated,
        "unique_denied_share": None if none else unique_denied / generated,
        "excessive_wait_share": None if none else excessive / generated,
        "censored_share": None if none else sum(censored_flags) / generated,
        "worst_route_id": worst_route,
        "worst_route_mean_wait": worst_mean,
        "mean_headway_s": float(np.mean(gaps)) if gaps else None,
        "p95_headway_s": _percentile(gaps, 95),
        "gap20_count": sum(gap > 1_200 for gap in gaps),
        "headway_target_violations": sum(gap > 900 for gap in gaps),
        "reserve_dispatches": kinds.count("DISPATCH"),
        "reassigns": kinds.count("REASSIGN"),
        "short_turns": kinds.count("SHORT_TURN"),
        "recalls": kinds.count("RECALL"),
        "decisions": scenario.config.horizon_s // scenario.config.control_interval_s,
    }


def _trace_row(env, action_kind: str) -> dict:
    state = env.state
    queues = []
    for route in range(env.config.route_count):
        queues.append(sum(cohort.count for cohort in state.cohorts if cohort.route_id == route))
    buses = [
        {
            "id": bus.vehicle_id,
            "phase": bus.phase.name,
            "route_id": bus.route_id,
            "pattern": bus.pattern.name,
            "load": bus.load,
        }
        for bus in state.vehicles.values()
    ]
    return {
        "time_s": state.current_time_s,
        "action": action_kind,
        "queues": queues,
        "headway_targets": [
            state.headway_targets_s[route] for route in range(env.config.route_count)
        ],
        "buses": buses,
        "waiting": state.waiting_count,
        "onboard": state.onboard_count,
        "generated": state.generated_count,
        "abandoned": state.abandoned_count,
        "completed": state.completed_count,
    }


def rollout(
    env: BusDispatchEnv,
    controller,
    scenario_index: int,
    *,
    trace: bool = False,
) -> tuple[dict, list[dict]]:
    started = perf_counter()
    observation, _ = env.reset(seed=0, options={"scenario_index": scenario_index})
    reward_sum = 0.0
    costs = StepCosts()
    rows: list[dict] = []
    from bus_rl.control.actions import ACTION_TABLE

    while True:
        mask = env.action_masks()
        action = controller.act(observation, mask)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        costs = add_costs(costs, info["costs"])
        if trace:
            rows.append(_trace_row(env, ACTION_TABLE[action].kind))
        if terminated or truncated:
            break
    with TIMERS.span("eval.summarize"):
        metrics = summarize_episode(env.state, env.scenario, costs, reward_sum, env.reward)
    metrics["wall_s"] = perf_counter() - started
    metrics["scenario_index"] = scenario_index
    return metrics, rows


def evaluate_scenarios(
    scenarios,
    run: RunConfig,
    method: str,
    *,
    model=None,
    model_seed: int | None = None,
    split: str = "",
    forecaster=None,
    trace_index: int | None = None,
) -> tuple[pd.DataFrame, dict[int, list[dict]]]:
    env = BusDispatchEnv(
        scenarios,
        run.physical,
        forecaster=forecaster,
        reward=run.reward,
        control=run.control,
    )
    controller = make_controller(method, model=model, seed=model_seed or 0)
    records = []
    traces: dict[int, list[dict]] = {}
    for index in range(len(scenarios)):
        metrics, rows = rollout(env, controller, index, trace=trace_index == index)
        metrics["method"] = method
        metrics["model_seed"] = model_seed
        metrics["split"] = split
        records.append(metrics)
        if rows:
            traces[index] = rows
    return pd.DataFrame.from_records(records), traces


def write_results(frame: pd.DataFrame, output: Path, traces: dict | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "results.csv", index=False)
    if traces:
        import json

        (output / "events.jsonl").write_text(
            "\n".join(json.dumps(row, default=str) for rows in traces.values() for row in rows)
            + "\n"
        )


def mean_cost(model, scenarios, run: RunConfig, forecaster=None, limit: int | None = None) -> float:
    subset = list(scenarios) if limit is None else list(scenarios)[:limit]
    frame, _ = evaluate_scenarios(
        subset, run, "ppo", model=model, model_seed=run.algorithm.seed, forecaster=forecaster
    )
    return float(frame["total_cost"].mean())


def apply_control(run: RunConfig, control: ControlConfig | None) -> RunConfig:
    return (
        run
        if control is None
        else RunConfig(
            physical=run.physical,
            control=control,
            reward=run.reward,
            algorithm=run.algorithm,
            forecast=run.forecast,
            source=run.source,
        )
    )
