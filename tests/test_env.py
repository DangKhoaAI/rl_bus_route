import numpy as np

from bus_rl.env.bus_dispatch import BusDispatchEnv
from tests.fixtures import empty_scenario


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
