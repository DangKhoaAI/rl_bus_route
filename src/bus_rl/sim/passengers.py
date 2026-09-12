"""FIFO passenger boarding, alighting, and abandonment."""

from __future__ import annotations

from dataclasses import dataclass

from bus_rl.domain import PassengerCohort, PassengerStatus, Pattern, Vehicle, WorldState


@dataclass(frozen=True)
class PassengerEvents:
    boarded_count: int = 0
    alighted_count: int = 0
    first_denied_count: int = 0


def _eligible(cohort: PassengerCohort, bus: Vehicle, direction: int, stop_index: int) -> bool:
    if (
        cohort.status is not PassengerStatus.WAITING
        or cohort.direction != direction
        or cohort.origin_index != stop_index
    ):
        return False
    if bus.pattern is Pattern.FULL:
        return True
    turn = bus.turn_stop if bus.turn_stop is not None else 3
    if direction == 1:
        return cohort.destination_index <= turn
    return stop_index <= turn


def _split_for_boarding(
    state: WorldState, cohort: PassengerCohort, count: int, bus_id: int
) -> None:
    boarded = PassengerCohort(
        cohort_id=state.next_cohort_id,
        lineage_id=cohort.lineage_id,
        route_id=cohort.route_id,
        direction=cohort.direction,
        origin_index=cohort.origin_index,
        destination_index=cohort.destination_index,
        arrival_tick=cohort.arrival_tick,
        count=count,
        status=PassengerStatus.ONBOARD,
        first_denied=cohort.first_denied,
        boarding_tick=state.current_time_s // 30,
    )
    state.next_cohort_id += 1
    cohort.count -= count
    state.cohorts.append(boarded)
    state.vehicles[bus_id].passengers.append(boarded)


def board_visit(
    state: WorldState, bus_id: int, route_id: int, direction: int, stop_index: int
) -> PassengerEvents:
    """Board FIFO eligible cohorts at a real visit; never moves the vehicle itself."""
    bus = state.vehicles[bus_id]
    if bus.route_id != route_id:
        raise ValueError("bus route does not match visit route")
    if bus.load > bus.capacity:
        raise AssertionError("bus is already over capacity")
    candidates = sorted(
        (
            c
            for c in state.cohorts
            if c.route_id == route_id and _eligible(c, bus, direction, stop_index)
        ),
        key=lambda c: (c.arrival_tick, c.cohort_id),
    )
    boarded = denied = 0
    for cohort in candidates:
        available = bus.capacity - bus.load
        take = min(available, cohort.count)
        if take:
            _split_for_boarding(state, cohort, take, bus_id)
            boarded += take
        if cohort.count and not cohort.first_denied:
            cohort.first_denied = True
            denied += cohort.count
    state.cohorts[:] = [c for c in state.cohorts if c.count > 0]
    state.assert_conservation()
    return PassengerEvents(boarded_count=boarded, first_denied_count=denied)


def alight_visit(state: WorldState, bus_id: int, stop_index: int) -> int:
    bus = state.vehicles[bus_id]
    alighted = 0
    for cohort in bus.passengers:
        if cohort.destination_index == stop_index and cohort.status is PassengerStatus.ONBOARD:
            cohort.status = PassengerStatus.COMPLETED
            cohort.completion_tick = state.current_time_s // 30
            alighted += cohort.count
    bus.passengers[:] = [c for c in bus.passengers if c.status is PassengerStatus.ONBOARD]
    state.assert_conservation()
    return alighted


def abandon_expired(state: WorldState, patience_s: int, tick_s: int) -> int:
    abandoned = 0
    for cohort in state.cohorts:
        if (
            cohort.status is PassengerStatus.WAITING
            and state.current_time_s - cohort.arrival_tick * tick_s >= patience_s
        ):
            cohort.status = PassengerStatus.ABANDONED
            cohort.abandonment_tick = state.current_time_s // tick_s
            abandoned += cohort.count
    state.assert_conservation()
    return abandoned
