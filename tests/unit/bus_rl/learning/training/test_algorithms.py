import numpy as np
import pytest
import torch
from gymnasium import spaces

from bus_rl.config import AlgorithmConfig
from bus_rl.learning.training.algorithms import MonteCarloBuffer


def test_monte_carlo_returns_ignore_values_and_do_not_cross_episodes():
    buffer = MonteCarloBuffer(
        6, spaces.Dict({"x": spaces.Box(0, 1, (1,))}), spaces.Discrete(3), gamma=1, n_envs=1
    )
    buffer.rewards[:, 0] = [-1, -2, -3, 4, 5, 6]
    buffer.episode_starts[:, 0] = [1, 0, 0, 1, 0, 0]
    buffer.values[:] = 12345
    buffer.compute_returns_and_advantage(torch.tensor([98765.0]), np.array([True]))
    np.testing.assert_array_equal(buffer.returns[:, 0], [-6, -5, -3, 15, 11, 6])
    np.testing.assert_array_equal(buffer.advantages, buffer.returns)
    assert not buffer.values.any()
    with pytest.raises(ValueError, match="complete episode"):
        buffer.compute_returns_and_advantage(torch.zeros(1), np.array([False]))


def test_unknown_algorithm_is_not_silently_ppo():
    with pytest.raises(ValueError, match="algorithm.name"):
        AlgorithmConfig(name="reinforc")
