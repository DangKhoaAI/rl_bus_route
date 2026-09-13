from bus_sim.oracle.actions import ACTION_TABLE, action_id
from bus_sim.oracle.dispatcher import apply_action
from bus_sim.oracle.domain import Action, Pattern, Phase, initial_state
from bus_sim.oracle.engine import advance_interval
from bus_sim.oracle.guards import valid_action_mask
from bus_sim.oracle.vehicles import complete_expired_phase
from tests.support.factories import empty_scenario


def test_dispatch_consumes_time_and_one_reserve():
    scenario = empty_scenario()
    state = initial_state(scenario)
    action = ACTION_TABLE[action_id("DISPATCH", bus_id=9, route_id=0)]
    advance_interval(state, scenario, action)
    assert state.current_time_s == 120
    bus = state.vehicles[9]
    target = scenario.network.routes[0].stops[0]
    assert bus.phase is Phase.DEADHEAD
    assert bus.load == 0
    assert bus.node == scenario.network.depot_node
    assert bus.next_node == target
    assert bus.node != target
    assert state.depot_count == 2


def test_headway_target_does_not_create_a_vehicle_and_recall_keeps_floor():
    scenario = empty_scenario()
    state = initial_state(scenario)
    before = dict(state.last_full_departure_s)
    action = ACTION_TABLE[action_id("SET_HEADWAY", route_id=0, headway_s=360)]
    apply_action(state, scenario, action)
    assert state.headway_targets_s[0] == 360
    assert state.last_full_departure_s == before
    assert len(state.vehicles) == scenario.config.fleet_size
    advance_interval(state, scenario, Action())
    assert not valid_action_mask(state, scenario)[
        action_id("SET_HEADWAY", route_id=0, headway_s=600)
    ]
    for bus_id in (0, 1):
        state.vehicles[bus_id].phase = Phase.DEPOT_IDLE
    assert not valid_action_mask(state, scenario)[action_id("RECALL", bus_id=2)]


def test_short_turn_reserve_deadheads_and_keeps_cooldown():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    action = ACTION_TABLE[action_id("SHORT_TURN", bus_id=9, route_id=0)]
    advance_interval(state, scenario, action)
    bus = state.vehicles[9]
    assert bus.phase is Phase.DEADHEAD
    assert bus.pattern is Pattern.SHORT
    assert bus.load == 0
    assert bus.cooldown_until_s == 1200
    assert state.depot_count == 2
    bus.phase = Phase.LAYOVER
    bus.node = scenario.network.routes[0].stops[0]
    bus.direction = 1
    bus.remaining_s = 0
    complete_expired_phase(state, scenario, bus)
    assert bus.pattern is Pattern.FULL
    assert bus.cooldown_until_s == 1200
    assert not valid_action_mask(state, scenario)[action_id("SHORT_TURN", 9, 0)]
