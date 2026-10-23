"""Masked on-policy trainers sharing the established policy and collector."""

from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from torch.nn import functional as F


class MonteCarloBuffer(MaskableDictRolloutBuffer):
    """Return-to-go from complete episodes, without value bootstrapping."""

    def compute_returns_and_advantage(self, last_values, dones):
        if not np.all(dones):
            raise ValueError("REINFORCE rollout must end at complete episode boundaries")
        remaining = np.zeros(self.n_envs, dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            terminal = dones if step == self.buffer_size - 1 else self.episode_starts[step + 1]
            remaining = self.rewards[step] + self.gamma * remaining * (1 - terminal)
            self.returns[step] = remaining
        self.values.fill(0)
        self.advantages = self.returns.copy()


class ConfiguredPPO(MaskablePPO):
    """PPO with explicit masking identity and actual optimizer-step accounting."""

    algorithm_name = "ppo"
    critic_used = True
    trainer_version = 1

    def __init__(self, *args, mask_mode="apply", **kwargs):
        self.mask_mode = mask_mode
        self.optimizer_steps = 0
        super().__init__(*args, **kwargs)

    def learn(self, *args, **kwargs):
        kwargs["use_masking"] = self.mask_mode == "apply"
        return super().learn(*args, **kwargs)

    def predict(self, observation, *args, action_masks=None, **kwargs):
        return super().predict(
            observation,
            *args,
            action_masks=action_masks if self.mask_mode == "apply" else None,
            **kwargs,
        )

    def train(self):
        optimizer = self.policy.optimizer
        original_step = optimizer.step

        def counted_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.optimizer_steps += 1
            return result

        optimizer.step = counted_step
        try:
            super().train()
        finally:
            optimizer.step = original_step


class MaskedA2C(ConfiguredPPO):
    """One synchronous, full-rollout advantage actor-critic update using Adam."""

    algorithm_name = "a2c"

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        data = next(self.rollout_buffer.get(batch_size=None))
        values, log_prob, entropy = self.policy.evaluate_actions(
            data.observations, data.actions.long().flatten(), action_masks=data.action_masks
        )
        weights = data.advantages if self.critic_used else data.returns
        if self.critic_used and self.normalize_advantage:
            weights = (weights - weights.mean()) / (weights.std() + 1e-8)
        policy_loss = -(weights.detach() * log_prob).mean()
        value_loss = (
            F.mse_loss(values.flatten(), data.returns.detach())
            if self.critic_used
            else values.new_zeros(())
        )
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        if not th.isfinite(loss):
            raise ValueError(f"{self.algorithm_name} produced a non-finite loss")
        self.policy.optimizer.zero_grad()
        loss.backward()
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        grad_norm = th.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        if not th.isfinite(grad_norm):
            raise ValueError(f"{self.algorithm_name} produced non-finite gradients")
        self.policy.optimizer.step()
        self.optimizer_steps += 1
        self._n_updates += 1
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates)
        if self.critic_used:
            self.logger.record("train/value_loss", value_loss.item())


class MaskedREINFORCE(MaskedA2C):
    """Monte Carlo policy gradient with entropy and no learned baseline."""

    algorithm_name = "reinforce"
    critic_used = False

    def __init__(self, *args, **kwargs):
        kwargs["rollout_buffer_class"] = MonteCarloBuffer
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        for module in (self.policy.value_net, self.policy.mlp_extractor.value_net):
            module.requires_grad_(False)
        optimizer = self.policy.optimizer
        self.policy.optimizer = type(optimizer)(
            [p for p in self.policy.parameters() if p.requires_grad], **optimizer.defaults
        )


TRAINERS = {"ppo": ConfiguredPPO, "a2c": MaskedA2C, "reinforce": MaskedREINFORCE}


def learned_method(name: str, mask_mode: str = "apply") -> str:
    return name if mask_mode == "apply" else f"{name}_free"
