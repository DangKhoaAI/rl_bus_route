from bus_sim.oracle.actions import ACTION_TABLE, action_id
from bus_sim.oracle.dispatcher import apply_action
from bus_sim.oracle.domain import Action, Pattern, Phase, initial_state
from bus_sim.oracle.engine import advance_interval, advance_tick
from bus_sim.oracle.guards import valid_action_mask
from bus_sim.oracle.vehicles import complete_expired_phase
from tests.support.factories import empty_scenario


def test_static_action_table_has_221_slots_and_noop_is_valid():
    scenario = empty_scenario()
    state = initial_state(scenario)
    mask = valid_action_mask(state, scenario)
    assert len(ACTION_TABLE) == 221
    assert mask[0]
    assert not mask[action_id("REASSIGN", bus_id=0, route_id=1)]
    assert not mask[action_id("SHORT_TURN", bus_id=0, route_id=0)]
    assert not mask[action_id("DISPATCH", bus_id=12, route_id=0)]
    assert not mask[action_id("DISPATCH", bus_id=9, route_id=3)]
    assert empty_scenario("M1").scenario_hash == empty_scenario("M2").scenario_hash
    assert empty_scenario("M1").scenario_hash == empty_scenario("M3").scenario_hash


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


def test_short_turn_from_reserve_or_ready_s0_and_respects_donor_guard():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    mask = valid_action_mask(state, scenario)
    assert mask[action_id("SHORT_TURN", 9, 0)]
    assert mask[action_id("SHORT_TURN", 0, 0)]
    assert not mask[action_id("SHORT_TURN", 2, 0)]
    state.vehicles[1].phase = Phase.SERVICE_MOVING
    mask = valid_action_mask(state, scenario)
    assert not mask[action_id("SHORT_TURN", 0, 0)]
    assert mask[action_id("SHORT_TURN", 9, 0)]
    for stage in ("M1", "M2"):
        other = empty_scenario(stage)
        assert not valid_action_mask(initial_state(other), other)[action_id("SHORT_TURN", 9, 0)]


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
