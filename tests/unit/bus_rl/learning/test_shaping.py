"""Unit contract for training-only potential-based reward shaping."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnv

from bus_rl.config import ShapingConfig
from bus_rl.learning.shaping import PotentialShapingVecEnv, potential_from_observation

STOPS_SHAPE = (4, 2, 8, 7)
VALID_SHAPE = (4, 2, 8)


def _obs(waiting: float, excess: float) -> dict[str, np.ndarray]:
    stops = np.zeros(STOPS_SHAPE, np.float32)
    valid = np.zeros(VALID_SHAPE, np.float32)
    valid[0, 0, 0] = 1.0
    stops[0, 0, 0, 0] = waiting / 40.0
    stops[0, 0, 0, 3] = excess / 40.0
    return {"stops": stops, "stop_valid": valid}


class _ScriptedVecEnv(VecEnv):
    """Minimal VecEnv returning pre-set (obs, reward, done) transitions."""

    def __init__(self, steps, initial_obs=None):
        self._steps = steps
        self._initial_obs = initial_obs if initial_obs is not None else steps[0]["obs"]
        self._index = 0
        obs_space = gym.spaces.Dict(
            {
                "stops": gym.spaces.Box(-np.inf, np.inf, STOPS_SHAPE, np.float32),
                "stop_valid": gym.spaces.Box(0.0, 1.0, VALID_SHAPE, np.float32),
            }
        )
        super().__init__(1, obs_space, gym.spaces.Discrete(2))

    def reset(self):
        self._index = 0
        return self._initial_obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        step = self._steps[self._index]
        self._index += 1
        return (
            step["obs"],
            np.array([step["reward"]], np.float32),
            np.array([step["done"]]),
            [step.get("info", {})],
        )

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


def test_potential_is_causal_and_finite():
    config = ShapingConfig(enabled=True, queue_weight=2.0, excess_weight=5.0)
    observation = _obs(waiting=50.0, excess=4.0)
    expected = -(2.0 * 50.0 + 5.0 * 4.0) / 3000.0
    np.testing.assert_allclose(potential_from_observation(observation, config), [expected])
    # Unbatched input is accepted and returns a length-1 array.
    np.testing.assert_allclose(potential_from_observation(observation, config), [expected])
    # Extra fields (including would-be future data) cannot change the potential.
    perturbed = dict(observation)
    perturbed["arrival_tape"] = np.ones((10, 3, 2, 6))
    perturbed["seed"] = 7
    np.testing.assert_array_equal(
        potential_from_observation(perturbed, config),
        potential_from_observation(observation, config),
    )
    with pytest.raises(ValueError):
        potential_from_observation(
            {
                "stops": np.full(STOPS_SHAPE, np.nan, np.float32),
                "stop_valid": observation["stop_valid"],
            },
            config,
        )


def test_shaping_telescopes_and_zeroes_terminal_potential():
    config = ShapingConfig(enabled=True, queue_weight=2.0, excess_weight=5.0)
    obs0 = _obs(waiting=80.0, excess=0.0)
    obs1 = _obs(waiting=40.0, excess=2.0)
    # Done slot returns the auto-reset observation, whose potential is nonzero;
    # the shaped reward must still use Phi(terminal)=0.
    obs_reset = _obs(waiting=10.0, excess=0.0)
    steps = [
        {"obs": obs1, "reward": -0.05, "done": False},
        {"obs": obs_reset, "reward": -0.04, "done": True},
    ]
    env = PotentialShapingVecEnv(_ScriptedVecEnv(steps, initial_obs=obs0), config, gamma=1.0)
    env.reset()
    _, shaped1, _, _ = env.step(np.array([0]))
    _, shaped2, _, _ = env.step(np.array([0]))

    phi0 = potential_from_observation(obs0, config)[0]
    phi1 = potential_from_observation(obs1, config)[0]
    np.testing.assert_allclose(shaped1[0], -0.05 + phi1 - phi0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(shaped2[0], -0.04 + 0.0 - phi1, rtol=0, atol=1e-6)
    # Telescoping: sum shaped = sum raw - Phi(s_0) because Phi(s_T)=0.
    np.testing.assert_allclose(shaped1[0] + shaped2[0], (-0.05) + (-0.04) - phi0, rtol=0, atol=1e-6)
    assert np.isfinite(shaped1).all() and np.isfinite(shaped2).all()


def test_shaping_requires_enabled_config():
    with pytest.raises(ValueError):
        PotentialShapingVecEnv(
            _ScriptedVecEnv([{"obs": _obs(0, 0), "reward": 0.0, "done": True}]),
            ShapingConfig(),
            gamma=1.0,
        )
