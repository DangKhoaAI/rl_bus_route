"""Native training-path PBRS wiring: shaped reward accounting on the real kernel.

The potential algebra is pinned by ``tests/unit/bus_rl/learning/test_shaping.py``.
This integration check pins the other half of the L2.3 card: a native
``NativeBatchVecEnv`` wrapped for shaping returns exactly
``r + gamma*Phi(s_{t+1}) - Phi(s_t)`` with terminal potential zero, while the
observation/mask API is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from bus_rl.config import load_run_config
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.shaping import PotentialShapingVecEnv, potential_from_observation
from tests.support.paths import CONFIGS

pytestmark = pytest.mark.native


def test_native_shaping_reward_accounting_and_api():
    pytest.importorskip("bus_sim_native")
    from bus_rl.execution.environments.rust.batch import NativeBatchVecEnv

    run = load_run_config(CONFIGS / "experiments" / "rl-improvement" / "pbrs.toml")
    train = [generate_scenario(1001), generate_scenario(1002)]
    raw = NativeBatchVecEnv(train, run, run.algorithm.seed)
    wrapped = PotentialShapingVecEnv(
        NativeBatchVecEnv(train, run, run.algorithm.seed), run.shaping, run.algorithm.gamma
    )
    try:
        raw_obs = raw.reset()
        wrapped_obs = wrapped.reset()
        np.testing.assert_array_equal(raw_obs["stops"], wrapped_obs["stops"])
        assert wrapped.observation_space == raw.observation_space

        previous = potential_from_observation(wrapped_obs, run.shaping)
        for _ in range(8):
            masks = raw.env_method("action_masks")
            actions = np.array([int(np.flatnonzero(mask)[0]) for mask in masks])
            raw_obs, raw_rewards, raw_dones, _ = raw.step(actions)
            wrapped_obs, shaped_rewards, wrapped_dones, _ = wrapped.step(actions)
            np.testing.assert_array_equal(raw_dones, wrapped_dones)

            next_potential = potential_from_observation(wrapped_obs, run.shaping)
            terminal_potential = np.where(wrapped_dones, 0.0, next_potential)
            expected = raw_rewards + run.algorithm.gamma * terminal_potential - previous
            np.testing.assert_allclose(shaped_rewards, expected, rtol=0, atol=1e-6)
            assert np.isfinite(shaped_rewards).all()
            previous = next_potential
    finally:
        raw.close()
        wrapped.close()
