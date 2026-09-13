from dataclasses import replace

import numpy as np
import pytest

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.baselines.random import RandomValidController
from bus_sim.oracle.actions import action_id
from bus_sim.oracle.costs import interval_cost, mean_waiting_minutes
from bus_sim.oracle.domain import initial_state
from bus_sim.oracle.engine import advance_tick
from bus_sim.oracle.observation import observe, validate_observation
from bus_sim.parity.scenarios import CATALOG
from bus_sim.parity.snapshot import numpy_state_snapshot
from tests.support.factories import empty_scenario, waiting_state
from tests.support.scenarios import new_env


def test_env_has_fixed_observation_and_finishes_exactly_at_horizon():
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    observation, info = env.reset(seed=7)
    assert not info and observation["vehicles"].shape == (16, 27)
    assert observation["stops"].shape == (4, 2, 8, 7)
    assert observation["stops"].dtype == np.float32
    terminated = truncated = False
    for _ in range(120):
        observation, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated


def test_observation_does_not_expose_scenario_seed_or_future_tape():
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    observation, info = env.reset(seed=3)
    assert "seed" not in observation and "arrival_tape" not in observation and not info
    assert "destination" not in observation
    assert observation["context"][2] == 0
    left = generate_scenario(21)
    future = np.array(left.arrival_tape, copy=True)
    future[40] = future[40] + 1
    right = replace(left, arrival_tape=future)
    env_left, env_right = BusDispatchEnv([left], left.config), BusDispatchEnv([right], right.config)
    obs_left, _ = env_left.reset(seed=0, options={"scenario_index": 0})
    obs_right, _ = env_right.reset(seed=0, options={"scenario_index": 0})
    for _ in range(3):
        obs_left, _, _, _, info_left = env_left.step(0)
        obs_right, _, _, _, info_right = env_right.step(0)
    for key in obs_left:
        np.testing.assert_array_equal(obs_left[key], obs_right[key])
    assert "arrival_tape" not in info_left and "seed" not in info_right


def test_invalid_action_raises_and_nan_observation_is_rejected():
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(action_id("SHORT_TURN", 0, 0))
    observation = observe(waiting_state(3), scenario)
    observation["stops"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_observation(observation)


def test_reset_and_control_do_not_mutate_scenario():
    scenario = generate_scenario(19)
    tape = np.array(scenario.arrival_tape)
    digest = scenario.scenario_hash
    env = BusDispatchEnv([scenario], scenario.config)
    env.reset(seed=3)
    env.step(0)
    np.testing.assert_array_equal(scenario.arrival_tape, tape)
    assert scenario.scenario_hash == digest


def test_invalid_action_is_rejected_without_mutating_state():
    env = new_env(CATALOG["normal_m3"])
    before = numpy_state_snapshot(env.state)
    invalid = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(invalid)
    after = numpy_state_snapshot(env.state)
    for key in before:
        assert np.array_equal(before[key], after[key]), f"state[{key}] mutated on rejection"


def test_conservation_holds_each_tick_and_tapes_ignore_controller_rng():
    scenario = empty_scenario()
    state = waiting_state(8)
    tape = scenario.arrival_tape.copy()
    for _ in range(12):
        advance_tick(state, scenario)
        state.assert_conservation()
        assert state.generated_count == (
            state.waiting_count
            + state.onboard_count
            + state.completed_count
            + state.abandoned_count
        )
    env = BusDispatchEnv([scenario], scenario.config)
    observation, _ = env.reset(seed=4)
    controller = RandomValidController(9)
    for _ in range(6):
        mask = env.action_masks()
        env.step(controller.act(observation, mask))
        observation = observe(env.state, env.scenario)
        env.state.assert_conservation()
    np.testing.assert_array_equal(scenario.arrival_tape, tape)


def test_env_reward_matches_negative_raw_costs_over_nref():
    scenario = empty_scenario()
    env = BusDispatchEnv([scenario], scenario.config)
    env.reset(seed=2)
    reward_sum = 0.0
    cost_sum = 0.0
    for _ in range(8):
        _, reward, _, _, info = env.step(0)
        reward_sum += reward
        cost_sum += interval_cost(info["costs"])
        assert abs(reward - (-interval_cost(info["costs"]) / 3000)) < 1e-12
    assert abs(reward_sum + cost_sum / 3000) < 1e-12
    assert mean_waiting_minutes(initial_state(scenario)) is None
