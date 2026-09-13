"""Training-only potential-based reward shaping (L2.3 PBRS).

The agent's raw reward comes from the native kernel. This module changes only
the *learning objective* seen by PPO: it wraps a training ``VecEnv`` and returns

    r_shaped = r + gamma * Phi(s_{t+1}) - Phi(s_t)

with a fixed causal potential built from the current observation's waiting and
excessive-wait counts. ``Phi`` is defined as zero at every terminal state, so the
shaped episode return differs from the raw return by a constant ``-Phi(s_0)``
under a fixed initial-state distribution (Ng et al., 1999). Evaluation is never
wrapped, and raw cost components / ``total_cost_core`` remain the comparison.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

from bus_rl.config import ShapingConfig


def potential_from_observation(
    observation: dict[str, np.ndarray], config: ShapingConfig
) -> np.ndarray:
    """Causal potential per env slot: negative observed queue and excessive wait.

    ``stops[..., 0]`` is waiting passengers / 40 and ``stops[..., 3]`` is the
    subset waiting at least 15 minutes / 40; ``stop_valid`` masks padded stops.
    ``n_ref`` puts the potential in the same units as the normalized training
    reward.
    """
    stops = np.asarray(observation["stops"], dtype=np.float64)
    valid = np.asarray(observation["stop_valid"], dtype=np.float64)
    if stops.ndim == 4:
        stops = stops[None, ...]
        valid = valid[None, ...]
    if stops.shape[:-1] != valid.shape:
        raise ValueError(f"stops {stops.shape} and stop_valid {valid.shape} disagree")
    waiting = (stops[..., 0] * valid).sum(axis=(1, 2, 3)) * 40.0
    excess = (stops[..., 3] * valid).sum(axis=(1, 2, 3)) * 40.0
    potential = -(config.queue_weight * waiting + config.excess_weight * excess) / config.n_ref
    if not np.isfinite(potential).all():
        raise ValueError("shaping potential produced NaN or inf")
    return potential


class PotentialShapingVecEnv(VecEnvWrapper):
    """Apply PBRS on top of a training ``VecEnv`` without touching the kernel."""

    def __init__(self, venv, config: ShapingConfig, gamma: float):
        super().__init__(venv)
        if not config.enabled:
            raise ValueError("PotentialShapingVecEnv requires shaping.enabled=true")
        self.config = config
        self.gamma = float(gamma)
        self._previous_potential: np.ndarray | None = None

    def reset(self):
        observation = self.venv.reset()
        self._previous_potential = potential_from_observation(observation, self.config)
        return observation

    def step_wait(self):
        if self._previous_potential is None:
            raise RuntimeError("PotentialShapingVecEnv.step_wait called before reset")
        observation, rewards, dones, infos = self.venv.step_wait()
        # Phi(s_{t+1}) on the post-step observation; the base VecEnv auto-resets
        # done slots, so this is the reset observation for those slots. The
        # reward uses Phi=0 at the terminal state, while the carried potential
        # must be the reset state for the next decision.
        next_potential = potential_from_observation(observation, self.config)
        terminal_potential = np.where(np.asarray(dones, dtype=bool), 0.0, next_potential)
        shaped = (
            np.asarray(rewards, dtype=np.float64)
            + self.gamma * terminal_potential
            - self._previous_potential
        )
        self._previous_potential = next_potential
        return observation, shaped.astype(np.float32), dones, infos
