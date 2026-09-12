from bus_rl.domain import Action, Phase, initial_state
from bus_rl.sim.engine import advance_interval, advance_tick
from bus_rl.sim.vehicles import begin_service_edge
from tests.fixtures import empty_scenario


def test_noop_interval_is_four_ticks_and_preserves_fleet():
    scenario = empty_scenario()
    state = initial_state(scenario)
    total = len(state.vehicles)
    advance_interval(state, scenario, Action())
    assert state.current_time_s == 120
    assert len(state.vehicles) == total


def test_moving_bus_arrives_then_laysover_at_terminal():
    scenario = empty_scenario()
    state = initial_state(scenario)
    bus = state.vehicles[0]
    bus.node = scenario.network.routes[0].stops[-2]
    bus.direction = 1
    begin_service_edge(state, scenario, bus.vehicle_id)
    for _ in range(bus.remaining_s // scenario.config.tick_s + 1):
        advance_tick(state, scenario)
    assert bus.node == scenario.network.routes[0].stops[-1]
    assert bus.phase is Phase.LAYOVER
