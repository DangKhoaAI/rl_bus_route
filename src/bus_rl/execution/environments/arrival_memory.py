"""Causal per-slot arrival history, beyond the native five-bin observation."""

import gymnasium as gym
import numpy as np


class ArrivalMemory:
    def __init__(self, capacity: int, bins: int):
        self.bins = bins
        self.values = np.zeros((capacity, 4, 2, 8, bins), dtype=np.float32)

    def space(self, base: gym.spaces.Dict) -> gym.spaces.Dict:
        if self.bins <= 5:
            return base
        return gym.spaces.Dict(
            {
                **base.spaces,
                "arrival_memory": gym.spaces.Box(0, np.inf, (4, 2, 8, self.bins), np.float32),
            }
        )

    def apply(self, observation: dict, slots, *, reset: bool = False) -> dict:
        if self.bins <= 5:
            return observation
        slots = list(slots)
        if reset:
            self.values[slots] = 0
        else:
            old = self.values[slots].copy()
            self.values[slots, ..., 1:] = old[..., :-1]
            self.values[slots, ..., 0] = np.asarray(observation["arrival_history"])[..., 0]
        # Copies are essential: rollout and terminal observations must remain immutable.
        return {**observation, "arrival_memory": self.values[slots].copy()}
