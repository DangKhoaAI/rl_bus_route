from __future__ import annotations

from bus_rl.domain import PassengerStatus, Phase, StepCosts, WorldState


def integrate_tick_costs(state: WorldState, duration_s: int) -> StepCosts:
    minutes = duration_s / 60
    waiting = state.waiting_count * minutes
    onboard = state.onboard_count * minutes
    crowding = sum(max(0, bus.load - 30) for bus in state.vehicles.values()) * minutes
    active = sum(bus.phase is not Phase.DEPOT_IDLE for bus in state.vehicles.values()) * minutes
    deadhead = sum(bus.phase is Phase.DEADHEAD for bus in state.vehicles.values()) * minutes
    excessive = (
        sum(
            c.count
            for c in state.cohorts
            if c.status.value == "WAITING" and state.current_time_s - c.arrival_tick * 30 >= 900
        )
        * minutes
    )
    return StepCosts(waiting, onboard, crowding, active, deadhead, excessive)


def add_costs(left: StepCosts, right: StepCosts) -> StepCosts:
    return StepCosts(
        **{name: getattr(left, name) + getattr(right, name) for name in left.__dataclass_fields__}
    )


def mean_waiting_minutes(state: WorldState, tick_s: int = 30) -> float | None:
    """Mean observed wait; None when no one was generated (no future-demand fill-in)."""
    if state.generated_count == 0:
        return None
    total = 0.0
    for cohort in state.cohorts:
        if cohort.status is PassengerStatus.COMPLETED and cohort.boarding_tick is not None:
            end_s = cohort.boarding_tick * tick_s
        elif cohort.status is PassengerStatus.ABANDONED and cohort.abandonment_tick is not None:
            end_s = cohort.abandonment_tick * tick_s
        else:
            end_s = state.current_time_s
        total += cohort.count * (end_s - cohort.arrival_tick * tick_s) / 60
    return total / state.generated_count


def interval_cost(costs: StepCosts) -> float:
    return (
        costs.waiting_pm
        + 0.25 * costs.onboard_pm
        + 0.5 * costs.crowding_pm
        + 0.5 * costs.active_bus_min
        + 0.5 * costs.deadhead_bus_min
        + costs.excessive_wait_pm
        + 5 * costs.first_denied_count
        + 60 * costs.abandoned_count
        + 2 * costs.mission_changes
        + 60 * costs.terminal_unfinished_count
    )
