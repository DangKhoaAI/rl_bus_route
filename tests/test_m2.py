from bus_rl.control.actions import ACTION_TABLE, action_id
from bus_rl.control.guards import valid_action_mask
from bus_rl.domain import Phase, initial_state
from bus_rl.env.bus_dispatch import BusDispatchEnv
from bus_rl.sim.engine import advance_interval
from tests.fixtures import empty_scenario


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
