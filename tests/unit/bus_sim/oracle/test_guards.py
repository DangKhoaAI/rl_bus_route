from bus_sim.oracle.actions import action_id
from bus_sim.oracle.dispatcher import apply_action
from bus_sim.oracle.domain import Action, Phase, initial_state
from bus_sim.oracle.guards import valid_action_mask
from tests.support.factories import empty_scenario


def test_reset_mask_allows_noop_and_blocks_invalid_slots():
    scenario = empty_scenario()
    state = initial_state(scenario)
    mask = valid_action_mask(state, scenario)
    assert mask[0]
    assert not mask[action_id("REASSIGN", bus_id=0, route_id=1)]
    assert not mask[action_id("SHORT_TURN", bus_id=0, route_id=0)]
    assert not mask[action_id("DISPATCH", bus_id=12, route_id=0)]
    assert not mask[action_id("DISPATCH", bus_id=9, route_id=3)]


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


def test_donor_guard_floor_and_replacement():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    recall = action_id("RECALL", bus_id=0)
    assert valid_action_mask(state, scenario)[recall]

    # Removing one committed FULL bus violates the two-bus floor.
    state.vehicles[2].phase = Phase.DEADHEAD
    assert not valid_action_mask(state, scenario)[recall]

    # Without a ready replacement donor at the same terminal the action is invalid.
    state = initial_state(empty_scenario("M3"))
    state.vehicles[1].route_id = None
    state.vehicles[1].phase = Phase.DEPOT_IDLE
    assert not valid_action_mask(state, scenario)[recall]


def test_cooldown_blocks_retargeting_a_bus():
    scenario = empty_scenario("M3")
    state = initial_state(scenario)
    reserve = min(
        vehicle.vehicle_id
        for vehicle in state.vehicles.values()
        if vehicle.phase is Phase.DEPOT_IDLE
    )
    dispatch = action_id("DISPATCH", bus_id=reserve, route_id=0)
    assert valid_action_mask(state, scenario)[dispatch]
    apply_action(state, scenario, Action("DISPATCH", reserve, 0))
    assert state.vehicles[reserve].cooldown_until_s == 1200
    # The bus is now deadheading and on cooldown.
    assert not valid_action_mask(state, scenario)[dispatch]
    assert not valid_action_mask(state, scenario)[action_id("REASSIGN", bus_id=reserve, route_id=1)]
