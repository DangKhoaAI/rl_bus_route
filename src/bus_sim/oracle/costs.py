from __future__ import annotations

from dataclasses import dataclass

from bus_sim.oracle.domain import PassengerStatus, Phase, StepCosts, WorldState


@dataclass(frozen=True)
class RewardConfig:
    waiting: float = 1.0
    onboard: float = 0.25
    crowding: float = 0.5
    active: float = 0.5
    deadhead: float = 0.5
    fairness: float = 1.0
    first_denied: float = 5.0
    abandoned: float = 60.0
    mission: float = 2.0
    unfinished: float = 60.0
    n_ref: float = 3_000.0


def integrate_tick_costs(state: WorldState, duration_s: int) -> StepCosts:
    minutes = duration_s / 60
    waiting = state.waiting_count * minutes
    onboard = state.onboard_count * minutes
    crowding = sum(max(0, bus.load - 30) for bus in state.vehicles.values()) * minutes
    active = sum(bus.phase is not Phase.DEPOT_IDLE for bus in state.vehicles.values()) * minutes
    deadhead = sum(bus.phase is Phase.DEADHEAD for bus in state.vehicles.values()) * minutes
    excessive = (
        sum(
            cohort.count
            for cohort in state.cohorts
            if state.current_time_s - cohort.arrival_tick * 30 >= 900
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
    for cohort in state.iter_cohorts():
        if cohort.status is PassengerStatus.COMPLETED and cohort.boarding_tick is not None:
            end_s = cohort.boarding_tick * tick_s
        elif cohort.status is PassengerStatus.ABANDONED and cohort.abandonment_tick is not None:
            end_s = cohort.abandonment_tick * tick_s
        else:
            end_s = state.current_time_s
        total += cohort.count * (end_s - cohort.arrival_tick * tick_s) / 60
    return total / state.generated_count


DEFAULT_REWARD = RewardConfig()


def interval_cost(costs: StepCosts, reward: RewardConfig | None = None) -> float:
    reward = reward or DEFAULT_REWARD
    return (
        reward.waiting * costs.waiting_pm
        + reward.onboard * costs.onboard_pm
        + reward.crowding * costs.crowding_pm
        + reward.active * costs.active_bus_min
        + reward.deadhead * costs.deadhead_bus_min
        + reward.fairness * costs.excessive_wait_pm
        + reward.first_denied * costs.first_denied_count
        + reward.abandoned * costs.abandoned_count
        + reward.mission * costs.mission_changes
        + reward.unfinished * costs.terminal_unfinished_count
    )
