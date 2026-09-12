"""Shared [256, 128] MLP plus separate actor/critic heads of 128."""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SharedMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self._keys = list(observation_space.spaces.keys())
        n_input = sum(int(np.prod(space.shape)) for space in observation_space.spaces.values())
        self.mlp = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        flat = th.cat([observations[key].flatten(start_dim=1) for key in self._keys], dim=1)
        return self.mlp(flat)


POLICY_KWARGS = {
    "features_extractor_class": SharedMLPExtractor,
    "features_extractor_kwargs": {"features_dim": 128},
    "net_arch": {"pi": [128], "vf": [128]},
}
