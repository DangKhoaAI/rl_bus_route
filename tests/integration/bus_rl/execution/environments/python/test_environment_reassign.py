from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_sim.oracle.actions import ACTION_TABLE, action_id
from bus_sim.oracle.domain import Pattern, Phase, initial_state
from bus_sim.oracle.engine import advance_interval
from bus_sim.oracle.guards import valid_action_mask
from bus_sim.oracle.passengers import board_visit
from tests.support.factories import empty_scenario, waiting_state


def test_reassign_requires_empty_terminal_donor_and_travels():
    scenario = empty_scenario("M2")
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
    full_ids = [
        other.vehicle_id
        for other in state.vehicles.values()
        if other.pattern is Pattern.FULL and other.phase not in (Phase.DEPOT_IDLE, Phase.DEADHEAD)
    ]
    assert bus.vehicle_id not in full_ids
    bus.phase = Phase.SERVICE_MOVING
    assert not valid_action_mask(state, scenario)[choice]


def test_reassign_masks_loaded_moving_cooldown_and_missing_replacement():
    scenario = empty_scenario("M2")
    state = initial_state(scenario)
    choice = action_id("REASSIGN", 0, 1)
    bus = state.vehicles[0]
    bus.cooldown_until_s = 1
    assert not valid_action_mask(state, scenario)[choice]
    bus.cooldown_until_s = 0
    bus.phase = Phase.SERVICE_MOVING
    assert not valid_action_mask(state, scenario)[choice]
    bus.phase = Phase.TERMINAL_IDLE
    state.vehicles[1].phase = Phase.SERVICE_MOVING
    assert not valid_action_mask(state, scenario)[choice]
    bus.phase = Phase.TERMINAL_IDLE
    state.vehicles[1].phase = Phase.TERMINAL_IDLE
    assert not valid_action_mask(state, scenario)[action_id("REASSIGN", 0, 0)]
    loaded = waiting_state(3, stage="M2")
    board_visit(loaded, 0, 0, 1, 0)
    assert not valid_action_mask(loaded, empty_scenario("M2"))[action_id("REASSIGN", 0, 1)]
    floor = initial_state(scenario)
    floor.vehicles[2].phase = Phase.DEADHEAD
    assert not valid_action_mask(floor, scenario)[action_id("REASSIGN", 0, 1)]


def test_seeded_m2_rollout_keeps_actions_valid():
    scenario = empty_scenario("M2")
    env = BusDispatchEnv([scenario], scenario.config)
    env.reset(seed=11)
    for _ in range(10):
        mask = env.action_masks()
        _, _, terminated, truncated, _ = env.step(int(mask.nonzero()[0][0]))
        assert not truncated
        if terminated:
            break
