from bus_rl.control.actions import ACTION_TABLE, action_id
from bus_rl.control.guards import valid_action_mask
from bus_rl.domain import Phase, initial_state
from bus_rl.sim.engine import advance_interval
from tests.fixtures import empty_scenario


def test_static_action_table_has_221_slots_and_noop_is_valid():
    scenario = empty_scenario()
    state = initial_state(scenario)
    mask = valid_action_mask(state, scenario)
    assert len(ACTION_TABLE) == 221
    assert mask[0]
    assert mask[action_id("REASSIGN", bus_id=0, route_id=1)]
    assert not mask[action_id("SHORT_TURN", bus_id=0, route_id=0)]


def test_dispatch_consumes_time_and_one_reserve():
    scenario = empty_scenario()
    state = initial_state(scenario)
    action = ACTION_TABLE[action_id("DISPATCH", bus_id=9, route_id=0)]
    advance_interval(state, scenario, action)
    assert state.current_time_s == 120
    assert state.vehicles[9].phase is Phase.DEADHEAD
    assert state.vehicles[9].load == 0
    assert state.depot_count == 2


def test_headway_target_does_not_create_a_vehicle_and_recall_keeps_floor():
    scenario = empty_scenario()
    state = initial_state(scenario)
    action = ACTION_TABLE[action_id("SET_HEADWAY", route_id=0, headway_s=360)]
    advance_interval(state, scenario, action)
    assert state.headway_targets_s[0] == 360
    assert len(state.vehicles) == scenario.config.fleet_size
    for bus_id in (0, 1):
        state.vehicles[bus_id].phase = Phase.DEPOT_IDLE
    assert not valid_action_mask(state, scenario)[action_id("RECALL", bus_id=2)]
