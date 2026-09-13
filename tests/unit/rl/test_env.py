from dataclasses import replace

import numpy as np
import pytest

from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_sim.oracle.actions import action_id
from bus_sim.oracle.observation import observe, validate_observation
from tests.support.factories import empty_scenario, waiting_state


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
