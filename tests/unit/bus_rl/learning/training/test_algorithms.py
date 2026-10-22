import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

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


def test_a2c_gae_targets_match_hand_calculation_and_zero_terminal_bootstrap():
    buffer = MaskableDictRolloutBuffer(
        3,
        spaces.Dict({"x": spaces.Box(0, 1, (1,))}),
        spaces.Discrete(3),
        gamma=1,
        gae_lambda=0.95,
        n_envs=1,
    )
    buffer.rewards[:, 0] = [-1, -2, -3]
    buffer.values[:, 0] = [0.5, 1, 1.5]
    buffer.episode_starts[:, 0] = [1, 0, 0]
    # TD residuals [-0.5, -1.5, -4.5], despite a deliberately huge final value.
    buffer.compute_returns_and_advantage(torch.tensor([999.0]), np.array([True]))
    np.testing.assert_allclose(buffer.advantages[:, 0], [-5.98625, -5.775, -4.5], rtol=1e-6)
    np.testing.assert_allclose(buffer.returns[:, 0], [-5.48625, -4.775, -3], rtol=1e-6)
