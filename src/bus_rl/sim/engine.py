"""Deterministic fixed-tick engine. Dispatch actions are added in Task 3."""

from __future__ import annotations

from bus_rl.domain import Action, PassengerCohort, Scenario, StepCosts, WorldState
from bus_rl.sim.passengers import abandon_expired, board_visit
from bus_rl.sim.vehicles import complete_expired_phase


def _add_arrivals(state: WorldState, scenario: Scenario) -> None:
    config = scenario.config
    if state.current_time_s >= config.demand_end_s:
        return
    tick = state.current_time_s // config.tick_s
    tape = scenario.arrival_tape[tick]
    for route_id in range(config.route_count):
        for direction_index, direction in enumerate((1, -1)):
            for origin in range(config.stops_per_route):
                for destination in range(config.stops_per_route):
                    count = int(tape[route_id, direction_index, origin, destination])
                    if not count:
                        continue
                    state.cohorts.append(
                        PassengerCohort(
                            cohort_id=state.next_cohort_id,
                            lineage_id=state.next_cohort_id,
                            route_id=route_id,
                            direction=direction,
                            origin_index=origin,
                            destination_index=destination,
                            arrival_tick=tick,
                            count=count,
                        )
                    )
                    state.next_cohort_id += 1
                    state.generated_total += count


def advance_tick(state: WorldState, scenario: Scenario) -> StepCosts:
    config = scenario.config
    if state.current_time_s >= config.horizon_s:
        return StepCosts()
    boarded = denied = 0
    # Step 1: events already due at this boundary.
    for bus in state.vehicles.values():
        if bus.remaining_s == 0:
            b, d = complete_expired_phase(state, scenario, bus)
            boarded += b
            denied += d
    # Step 2: exogenous arrivals, then abandonment before any boarding at this boundary.
    _add_arrivals(state, scenario)
    abandoned = abandon_expired(state, config.patience_s, config.tick_s)
    # Terminal visits may board, but the autonomous terminal dispatcher starts in Task 3.
    for bus in state.vehicles.values():
        if bus.phase.name == "TERMINAL_IDLE" and bus.route_id is not None:
            stop_index = scenario.network.routes[bus.route_id].stops.index(bus.node)
            event = board_visit(state, bus.vehicle_id, bus.route_id, bus.direction, stop_index)
            boarded += event.boarded_count
            denied += event.first_denied_count
    # Step 5: advance timers over [t, t + tick).
    for bus in state.vehicles.values():
        if bus.remaining_s > 0:
            bus.remaining_s = max(0, bus.remaining_s - config.tick_s)
    state.current_time_s += config.tick_s
    state.assert_conservation()
    state.event_log.append(
        {
            "time_s": state.current_time_s,
            "boarded": boarded,
            "denied": denied,
            "abandoned": abandoned,
        }
    )
    return StepCosts(first_denied_count=denied, abandoned_count=abandoned)


def advance_interval(
    state: WorldState, scenario: Scenario, action: Action | None = None
) -> StepCosts:
    action = action or Action()
    if action.kind != "NOOP":
        raise ValueError("Task 2 only supports NOOP; operational actions arrive in Task 3")
    result = StepCosts()
    for _ in range(scenario.config.control_interval_s // scenario.config.tick_s):
        tick_costs = advance_tick(state, scenario)
        result.first_denied_count += tick_costs.first_denied_count
        result.abandoned_count += tick_costs.abandoned_count
    return result
