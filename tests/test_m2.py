from bus_rl.control.actions import ACTION_TABLE, action_id
from bus_rl.control.guards import valid_action_mask
from bus_rl.domain import Phase, initial_state
from bus_rl.sim.engine import advance_interval
from tests.fixtures import empty_scenario


def test_reassign_requires_empty_terminal_donor_and_travels():
    scenario = empty_scenario()
    state = initial_state(scenario)
    bus = state.vehicles[0]
    choice = action_id("REASSIGN", 0, 1)
    assert valid_action_mask(state, scenario)[choice]
    advance_interval(state, scenario, ACTION_TABLE[choice])
    assert (
        bus.phase is Phase.DEADHEAD
        and bus.route_id == 1
        and bus.node != scenario.network.routes[1].stops[0]
    )
    bus.phase = Phase.SERVICE_MOVING
    assert not valid_action_mask(state, scenario)[choice]
