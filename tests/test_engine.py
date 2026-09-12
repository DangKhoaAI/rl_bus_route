from bus_rl.domain import Action, Pattern, Phase, initial_state
from bus_rl.sim.engine import advance_interval, advance_tick
from bus_rl.sim.vehicles import begin_service_edge, complete_expired_phase
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


def test_short_turn_laysover_at_turnpoint_then_restores_full():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    bus = state.vehicles[0]
    bus.pattern = Pattern.SHORT
    bus.turn_stop = 3
    route = scenario.network.routes[0]
    bus.node = route.stops[2]
    bus.direction = 1
    bus.phase = Phase.SERVICE_MOVING
    bus.next_node = route.stops[3]
    bus.remaining_s = 0
    complete_expired_phase(state, scenario, bus)
    assert bus.node == route.stops[3]
    assert bus.phase is Phase.LAYOVER
    assert bus.direction == -1
    bus.remaining_s = 0
    complete_expired_phase(state, scenario, bus)
    assert bus.phase is Phase.TERMINAL_IDLE
    assert bus.pattern is Pattern.SHORT
    assert bus.pending_extra
    bus.node = route.stops[1]
    bus.phase = Phase.SERVICE_MOVING
    bus.next_node = route.stops[0]
    bus.remaining_s = 0
    complete_expired_phase(state, scenario, bus)
    assert bus.phase is Phase.LAYOVER
    bus.remaining_s = 0
    complete_expired_phase(state, scenario, bus)
    assert bus.pattern is Pattern.FULL
    assert bus.phase is Phase.TERMINAL_IDLE
    assert not bus.pending_extra
