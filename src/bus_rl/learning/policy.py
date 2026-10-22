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


def arrival_rate_belief(
    observations: dict[str, th.Tensor],
    episode_steps: int,
    prior_shape: float = 1.0,
    prior_rate: float = 1.0,
    decay: float = 1.0,
) -> th.Tensor:
    """Gamma-Poisson rate posterior over observed bins, optionally discounted.

    Assumes a locally constant Poisson rate, independently for each stop/direction.
    This is a windowed demand belief, not a posterior over the full simulator state.
    At startup only elapsed bins count as exposure; left-padding is not evidence.
    With decay < 1 this is a power-likelihood approximation, not an exact posterior
    under the simulator's time-varying demand process.
    """
    history = observations.get("arrival_memory", observations["arrival_history"])
    lags = history.shape[-1]
    exposure = (observations["context"][:, 0] * episode_steps).round().clamp(0, lags)
    observed = th.arange(lags, device=history.device)[None, :] < exposure[:, None]
    weights = decay ** th.arange(lags, device=history.device, dtype=history.dtype)
    weights = observed * weights[None, :]
    counts = (history * weights[:, None, None, None, :]).sum(dim=-1) * 40.0
    alpha = prior_shape + counts
    beta = prior_rate + weights.sum(dim=-1)[:, None, None, None]
    valid = observations["stop_valid"]
    mean = alpha / beta / 40.0 * valid
    std = alpha.sqrt() / beta / 40.0 * valid
    return th.cat([mean.flatten(1), std.flatten(1)], dim=1)


class BeliefGRUExtractor(BaseFeaturesExtractor):
    """Append causal rate-belief and/or GRU features to the unchanged raw input.

    GRU reads oldest -> newest arrivals and starts from zero on every forward.
    There is no hidden mutable state, so shuffled PPO minibatches, scalar eval,
    native batches and checkpoint loading all use the same feature calculation.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
        use_belief: bool = False,
        use_gru: bool = False,
        episode_steps: int = 120,
        belief_prior_shape: float = 1.0,
        belief_prior_rate: float = 1.0,
        belief_decay: float = 1.0,
    ):
        super().__init__(observation_space, features_dim)
        self._keys = [key for key in observation_space.spaces if key != "arrival_memory"]
        self.history_key = (
            "arrival_memory" if "arrival_memory" in observation_space.spaces else "arrival_history"
        )
        self.belief_prior_shape = belief_prior_shape
        self.belief_prior_rate = belief_prior_rate
        self.belief_decay = belief_decay
        self.use_belief, self.use_gru = use_belief, use_gru
        self.episode_steps = episode_steps
        n_input = sum(int(np.prod(observation_space[key].shape)) for key in self._keys)
        n_stops = int(np.prod(observation_space["arrival_history"].shape[:-1]))
        extra = 2 * n_stops if use_belief else 0
        recurrent_params = 0
        if use_gru:
            self.gru = nn.GRU(n_stops, 32, batch_first=True)
            extra += 32
            recurrent_params = sum(parameter.numel() for parameter in self.gru.parameters())
        # Match the original extractor's parameter budget to within one hidden unit.
        budget = (n_input + 1) * 256 + (256 + 1) * features_dim
        width = round(
            (budget - recurrent_params - features_dim) / (n_input + extra + 1 + features_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_input + extra, width),
            nn.ReLU(),
            nn.Linear(width, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        features = [observations[key].flatten(start_dim=1) for key in self._keys]
        if self.use_belief:
            features.append(
                arrival_rate_belief(
                    observations,
                    self.episode_steps,
                    self.belief_prior_shape,
                    self.belief_prior_rate,
                    self.belief_decay,
                )
            )
        if self.use_gru:
            history = observations[self.history_key] * observations["stop_valid"].unsqueeze(-1)
            sequence = history.flatten(1, 3).transpose(1, 2).flip(1)
            _, hidden = self.gru(sequence)
            features.append(hidden[-1])
        return self.mlp(th.cat(features, dim=1))


def policy_kwargs(
    encoder: str = "gru",
    episode_steps: int = 120,
    *,
    belief_prior_shape: float = 1.0,
    belief_prior_rate: float = 1.0,
    belief_decay: float = 1.0,
) -> dict:
    if encoder == "mlp":
        return POLICY_KWARGS
    if encoder not in {"belief", "gru", "belief_gru"}:
        raise ValueError(f"unknown encoder: {encoder!r}")
    return {
        "features_extractor_class": BeliefGRUExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "use_belief": encoder in {"belief", "belief_gru"},
            "use_gru": encoder in {"gru", "belief_gru"},
            "episode_steps": episode_steps,
            "belief_prior_shape": belief_prior_shape,
            "belief_prior_rate": belief_prior_rate,
            "belief_decay": belief_decay,
        },
        "net_arch": {"pi": [128], "vf": [128]},
    }
