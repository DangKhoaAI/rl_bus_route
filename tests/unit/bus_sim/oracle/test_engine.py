from bus_rl.execution.scenarios.generation import generate_scenario
from bus_sim.oracle.costs import interval_cost
from bus_sim.oracle.domain import Action, PassengerCohort, Phase, SimConfig, initial_state
from bus_sim.oracle.engine import advance_interval, advance_tick
from tests.support.factories import empty_scenario


def test_noop_interval_is_four_ticks_and_preserves_fleet():
    scenario = empty_scenario()
    state = initial_state(scenario)
    total = len(state.vehicles)
    advance_interval(state, scenario, Action())
    assert state.current_time_s == 120
    assert len(state.vehicles) == total


def test_empty_demand_noop_runs_service_without_changing_fleet():
    scenario = empty_scenario()
    state = initial_state(scenario)
    fleet = len(state.vehicles)
    saw_service = False
    for _ in range(120):
        advance_interval(state, scenario, Action())
        state.assert_conservation()
        saw_service = saw_service or any(
            bus.phase in (Phase.SERVICE_MOVING, Phase.LAYOVER, Phase.SERVICE_DWELL)
            for bus in state.vehicles.values()
        )
    assert len(state.vehicles) == fleet == scenario.config.fleet_size
    assert saw_service
    assert state.current_time_s == scenario.config.horizon_s


def test_minimum_spacing_allows_only_one_departure_per_opportunity():
    scenario = empty_scenario()
    state = initial_state(scenario)
    advance_tick(state, scenario)
    outbound = [
        bus
        for bus in state.vehicles.values()
        if bus.route_id == 0 and bus.direction == 1 and bus.phase is Phase.SERVICE_MOVING
    ]
    assert len(outbound) <= 1


def test_horizon_settles_unfinished_and_skips_already_abandoned():
    config = SimConfig(horizon_s=120, demand_end_s=0)
    scenario = generate_scenario(5, config)
    state = initial_state(scenario)
    state.add_waiting(PassengerCohort(0, 0, 0, 1, 0, 5, 0, 4))
    state.add_waiting(PassengerCohort(1, 1, 0, 1, 0, 5, -90, 2))
    for bus in state.vehicles.values():
        bus.phase = Phase.DEPOT_IDLE
        bus.route_id = None
    first = advance_interval(state, scenario, Action())
    last = advance_interval(state, scenario, Action())
    assert state.current_time_s == 120
    assert state.abandoned_count == 2
    assert state.waiting_count == 4
    assert first.terminal_unfinished_count == 4
    assert last.terminal_unfinished_count == 0
    assert interval_cost(first) >= 60 * 4
