"""Backend-agnostic episode summary inputs and metrics.

`evaluation/runner.py` needs cohort/lineage, departure and counter data to
compute every metric. This module defines a small value interface
(`SummaryInputs`/`CohortView`) that both the Python oracle env and the native
env can produce, and the shared metric math. Statistics and plots stay in
Python; only the raw per-episode inputs are exported.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from bus_rl.domain import PassengerStatus, StepCosts
from bus_rl.rewards.costs import DEFAULT_REWARD, RewardConfig, interval_cost


@dataclass(frozen=True)
class CohortView:
    route_id: int
    arrival_tick: int
    count: int
    status: int
    boarding_tick: int | None
    abandonment_tick: int | None
    first_denied: bool


@dataclass(frozen=True)
class SummaryInputs:
    current_time_s: int
    generated: int
    waiting: int
    onboard: int
    completed: int
    abandoned: int
    cohorts: list[CohortView]
    departures: list[dict]
    accepted_kinds: list[str]


def from_python_state(state) -> SummaryInputs:
    """Adapter for the Python oracle ``WorldState``."""
    cohorts = [
        CohortView(
            route_id=cohort.route_id,
            arrival_tick=cohort.arrival_tick,
            count=cohort.count,
            status=int(cohort.status),
            boarding_tick=cohort.boarding_tick,
            abandonment_tick=cohort.abandonment_tick,
            first_denied=cohort.first_denied,
        )
        for cohort in state.iter_cohorts()
    ]
    return SummaryInputs(
        current_time_s=state.current_time_s,
        generated=state.generated_count,
        waiting=state.waiting_count,
        onboard=state.onboard_count,
        completed=state.completed_count,
        abandoned=state.abandoned_count,
        cohorts=cohorts,
        departures=list(state.departures),
        accepted_kinds=[event["kind"] for event in state.accepted_actions],
    )


# `full_snapshot` cohort rows are:
# [bus_id, cohort_id, lineage_id, route_id, direction, origin, destination,
#  arrival_tick, count, status, first_denied, boarding_tick, completion_tick,
#  abandonment_tick]
def from_native_payload(payload: dict) -> SummaryInputs:
    """Adapter for ``bus_sim.Kernel.episode_summary_inputs()``."""
    counters = np.asarray(payload["counters"])
    rows = np.asarray(payload["cohorts"])
    kinds = np.asarray(payload["cohort_kind"])
    cohorts: list[CohortView] = []
    for row, _kind in zip(rows, kinds, strict=True):
        boarding = None if int(row[11]) < 0 else int(row[11])
        abandonment = None if int(row[13]) < 0 else int(row[13])
        cohorts.append(
            CohortView(
                route_id=int(row[3]),
                arrival_tick=int(row[7]),
                count=int(row[8]),
                status=int(row[9]),
                boarding_tick=boarding,
                abandonment_tick=abandonment,
                first_denied=bool(row[10]),
            )
        )
    return SummaryInputs(
        current_time_s=int(payload["time_s"]),
        generated=int(counters[0]),
        waiting=int(counters[1]),
        onboard=int(counters[2]),
        completed=int(counters[3]),
        abandoned=int(counters[4]),
        cohorts=cohorts,
        departures=list(payload["departures"]),
        accepted_kinds=[str(kind) for kind in payload["accepted_kinds"]],
    )


def _wait_minutes(cohort: CohortView, current_time_s: int, tick_s: int) -> tuple[float, bool]:
    arrival_s = cohort.arrival_tick * tick_s
    if cohort.status == PassengerStatus.COMPLETED and cohort.boarding_tick is not None:
        return (cohort.boarding_tick * tick_s - arrival_s) / 60.0, False
    if cohort.status == PassengerStatus.ABANDONED and cohort.abandonment_tick is not None:
        return (cohort.abandonment_tick * tick_s - arrival_s) / 60.0, False
    if cohort.status == PassengerStatus.ONBOARD and cohort.boarding_tick is not None:
        return (cohort.boarding_tick * tick_s - arrival_s) / 60.0, False
    return (current_time_s - arrival_s) / 60.0, True


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize_inputs(
    inputs: SummaryInputs,
    scenario,
    costs: StepCosts,
    reward: float,
    reward_config: RewardConfig | None = None,
) -> dict:
    """Metric math shared by both backends (verbatim from the R0 oracle)."""
    tick_s = scenario.config.tick_s
    waits: list[float] = []
    censored_flags: list[bool] = []
    route_waits: dict[int, list[float]] = defaultdict(list)
    excessive = 0
    unique_denied = 0
    for cohort in inputs.cohorts:
        wait, censored = _wait_minutes(cohort, inputs.current_time_s, tick_s)
        waits.extend([wait] * cohort.count)
        censored_flags.extend([censored] * cohort.count)
        route_waits[cohort.route_id].extend([wait] * cohort.count)
        if wait >= 15:
            excessive += cohort.count
        if cohort.first_denied:
            unique_denied += cohort.count
    generated = inputs.generated
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
    for event in inputs.departures:
        if event.get("pattern") != "FULL":
            continue
        grouped[int(event["route_id"]), int(event["direction"])].append(int(event["time_s"]))
    for times in grouped.values():
        ordered = sorted(times)
        gaps.extend(later - earlier for earlier, later in pairwise(ordered))
    kinds = inputs.accepted_kinds
    core_cost = interval_cost(costs, DEFAULT_REWARD)
    trained_cost = interval_cost(costs, reward_config or DEFAULT_REWARD)
    return {
        "scenario_seed": scenario.seed,
        "scenario_hash": scenario.scenario_hash,
        "generated": generated,
        "completed": inputs.completed,
        "abandoned": inputs.abandoned,
        "unfinished": inputs.waiting + inputs.onboard,
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
        "completed_share": None if none else inputs.completed / generated,
        "abandoned_share": None if none else inputs.abandoned / generated,
        "unfinished_share": None if none else (inputs.waiting + inputs.onboard) / generated,
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


def empty_inputs() -> SummaryInputs:
    return SummaryInputs(0, 0, 0, 0, 0, 0, [], [], [])
