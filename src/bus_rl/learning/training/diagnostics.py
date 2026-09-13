"""Side-effect-free PPO telemetry and finite-value checks for RL study runs.

The helpers in this module deliberately operate on masks/probabilities supplied by
callers.  They do not alter the environment, policy, RNG state, or PPO loss.  The
training callback below samples the policy at a registered cadence and records
mask-aware exploration telemetry plus PPO logger/finite checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from bus_sim.oracle.actions import ACTION_TABLE

ACTION_FAMILIES = tuple(dict.fromkeys(action.kind for action in ACTION_TABLE))
_FAMILY_INDEX = {
    family: np.asarray([action.kind == family for action in ACTION_TABLE], dtype=bool)
    for family in ACTION_FAMILIES
}


def finite_check(name: str, value) -> dict:
    """Return a serializable finite-value check, including failure details."""
    array = np.asarray(value)
    finite = np.isfinite(array)
    return {
        "name": name,
        "finite": bool(finite.all()),
        "count": int(array.size),
        "nonfinite_count": int((~finite).sum()),
        "shape": list(array.shape),
    }


def require_finite(name: str, value) -> dict:
    result = finite_check(name, value)
    if not result["finite"]:
        raise FloatingPointError(
            f"non-finite {name}: {result['nonfinite_count']}/{result['count']} values"
        )
    return result


def _as_one_dimensional_mask(mask) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 1 or array.size != len(ACTION_TABLE):
        raise ValueError(f"action mask must have shape ({len(ACTION_TABLE)},), got {array.shape}")
    return array


def mask_metrics(mask, action: int | None = None) -> dict:
    """Summarize one valid-action mask and an optional selected action.

    A family with no valid slot gets a null conditional selection rate.  This
    prevents a zero denominator from being reported as evidence that a policy
    ignored that family.
    """
    valid = _as_one_dimensional_mask(mask)
    k = int(valid.sum())
    if k == 0:
        raise ValueError("action mask has K=0 valid actions")
    if action is not None:
        action = int(action)
        if action < 0 or action >= len(ACTION_TABLE) or not valid[action]:
            raise ValueError(f"selected action {action} is not valid under the supplied mask")
    row = {
        "valid_action_count": k,
        "k_one": k == 1,
        "selected_action": action,
        "families": {},
    }
    for family, family_mask in _FAMILY_INDEX.items():
        valid_slots = int(np.count_nonzero(valid & family_mask))
        opportunity = valid_slots > 0
        selected = int(
            action is not None
            and action < len(ACTION_TABLE)
            and ACTION_TABLE[action].kind == family
        )
        row["families"][family] = {
            "valid_slots": valid_slots,
            "opportunity": opportunity,
            "selected": selected,
            "conditional_selection_rate": (float(selected) if opportunity else None),
        }
    return row


def probability_metrics(probabilities, mask, action: int | None = None) -> dict:
    """Compute entropy and family probability mass on the valid support only."""
    valid = _as_one_dimensional_mask(mask)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != valid.shape:
        raise ValueError(f"probabilities must have shape {valid.shape}, got {probs.shape}")
    require_finite("masked probabilities", probs)
    if np.any(probs < -1e-7):
        raise ValueError("masked probabilities contain a negative value")
    if np.any(np.abs(probs[~valid]) > 1e-6):
        raise ValueError("masked probabilities assign mass to an invalid action")
    support = np.where(valid, probs, 0.0)
    total = float(support.sum())
    if not np.isfinite(total) or not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError(f"valid probability mass must sum to one, got {total}")
    k = int(valid.sum())
    positive = support[support > 0.0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
    row = {
        "valid_action_count": k,
        "entropy": 0.0 if k == 1 else entropy,
        "normalized_entropy": None if k == 1 else entropy / np.log(k),
        "families": {},
    }
    for family, family_mask in _FAMILY_INDEX.items():
        valid_slots = int(np.count_nonzero(valid & family_mask))
        opportunity = valid_slots > 0
        selected = int(
            action is not None
            and 0 <= int(action) < len(ACTION_TABLE)
            and ACTION_TABLE[int(action)].kind == family
        )
        row["families"][family] = {
            "valid_slots": valid_slots,
            "opportunity": opportunity,
            "probability_mass": float(support[family_mask].sum()),
            "selected": selected,
            "conditional_selection_rate": (float(selected) if opportunity else None),
        }
    return row


def distribution_summary(name: str, value, *, sample_limit: int = 100_000) -> dict:
    """Summarize a numeric tensor deterministically without storing raw tensors."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = finite_check(name, array)
    if not finite["finite"]:
        return finite
    if array.size > sample_limit:
        stride = max(1, array.size // sample_limit)
        array = array[::stride][:sample_limit]
    if array.size == 0:
        finite.update({"mean": None, "std": None, "min": None, "max": None})
        return finite
    finite.update(
        {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "q05": float(np.quantile(array, 0.05)),
            "q50": float(np.quantile(array, 0.50)),
            "q95": float(np.quantile(array, 0.95)),
        }
    )
    return finite


def _metric_value(logger_values: dict, name: str):
    value = logger_values.get(name)
    if value is None:
        return None
    return float(value)


class TrainingDiagnosticsCallback(BaseCallback):
    """Persist L0 telemetry without changing PPO's sampled actions or loss."""

    def __init__(self, output: Path, *, sample_interval: int = 256):
        super().__init__()
        self.output = Path(output)
        self.sample_interval = max(1, int(sample_interval))
        self.started = 0.0
        self.next_sample = self.sample_interval
        self._mask_totals = {
            family: {"valid_slots": 0, "opportunities": 0, "selected": 0}
            for family in ACTION_FAMILIES
        }
        self._mask_steps = 0
        self._last_update_timesteps = 0

    def _on_training_start(self) -> None:
        self.started = monotonic()
        self._write(
            {
                "kind": "settings",
                "timesteps": 0,
                "elapsed_wall_s": 0.0,
                "sample_interval": self.sample_interval,
                "finite_check": "every_rollout_and_after_each_ppo_update",
                "probability_sampling": "policy_distribution_at_sample_interval",
            }
        )

    def _on_step(self) -> bool:
        masks = self.locals.get("action_masks")
        actions = self.locals.get("actions")
        if masks is None:
            self._write(
                {
                    "kind": "availability",
                    "timesteps": int(self.num_timesteps),
                    "elapsed_wall_s": monotonic() - self.started,
                    "available": False,
                    "reason": "callback locals did not expose action_masks",
                }
            )
            return True
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 1:
            masks = masks[None, :]
        actions_array = None if actions is None else np.asarray(actions).reshape(-1)
        for index, mask in enumerate(masks):
            selected = None if actions_array is None else int(actions_array[index])
            metrics = mask_metrics(mask, selected)
            for family, family_row in metrics["families"].items():
                self._mask_totals[family]["valid_slots"] += family_row["valid_slots"]
                self._mask_totals[family]["opportunities"] += int(family_row["opportunity"])
                self._mask_totals[family]["selected"] += family_row["selected"]
            self._mask_steps += 1
        if self.num_timesteps >= self.next_sample:
            self._sample_policy(masks, actions_array)
            self.next_sample += self.sample_interval
        return True

    def _sample_policy(self, masks: np.ndarray, actions: np.ndarray | None) -> None:
        # MaskablePPO exposes the pre-step observation as `_last_obs`; this
        # is the same observation used to produce the sampled action.
        observation = self.locals.get(
            "obs_tensor", self.locals.get("_last_obs", self.locals.get("obs"))
        )
        row = {
            "kind": "policy_diagnostics",
            "timesteps": int(self.num_timesteps),
            "elapsed_wall_s": monotonic() - self.started,
            "mask_steps": self._mask_steps,
            "valid_action_count_mean": float(masks.sum(axis=1).mean()),
            "valid_action_count_min": int(masks.sum(axis=1).min()),
            "valid_action_count_max": int(masks.sum(axis=1).max()),
            "k_one_count": int(np.count_nonzero(masks.sum(axis=1) == 1)),
            "families": {},
        }
        for family, totals in self._mask_totals.items():
            opportunity = totals["opportunities"]
            row["families"][family] = {
                **totals,
                "conditional_selection_rate": (
                    totals["selected"] / opportunity if opportunity else None
                ),
            }
        if observation is None:
            row["available"] = False
            row["reason"] = "callback locals did not expose observations"
            self._write(row)
            return
        try:
            with torch.no_grad():
                if isinstance(observation, dict) and all(
                    isinstance(value, torch.Tensor) for value in observation.values()
                ):
                    tensor_obs = observation
                else:
                    tensor_obs = obs_as_tensor(observation, self.model.device)
                distribution = self.model.policy.get_distribution(tensor_obs, action_masks=masks)
                categorical = distribution.distribution
                probabilities = categorical.probs.detach().cpu().numpy()
                logits = categorical.logits.detach().cpu().numpy()
            probability_rows = [
                probability_metrics(
                    probabilities[index],
                    masks[index],
                    None if actions is None else int(actions[index]),
                )
                for index in range(len(masks))
            ]
            row["available"] = True
            row["entropy_mean"] = float(np.mean([item["entropy"] for item in probability_rows]))
            normalized = [item["normalized_entropy"] for item in probability_rows]
            normalized = [value for value in normalized if value is not None]
            row["normalized_entropy_mean"] = float(np.mean(normalized)) if normalized else None
            row["probability_family_mass"] = {
                family: float(
                    np.mean(
                        [item["families"][family]["probability_mass"] for item in probability_rows]
                    )
                )
                for family in ACTION_FAMILIES
            }
            row["logits_finite"] = finite_check("logits", logits)
            require_finite("valid logits", logits[masks])
            row["observation_distributions"] = {
                key: distribution_summary(
                    f"observation.{key}",
                    value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value,
                )
                for key, value in observation.items()
            }
        except Exception as error:
            row["available"] = False
            row["reason"] = f"policy diagnostic failed: {type(error).__name__}: {error}"
            self._write(row)
            raise
        self._write(row)

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer
        finite_checks = {}
        for name in ("rewards", "values", "log_probs", "advantages", "returns"):
            value = getattr(buffer, name, None)
            if value is not None:
                finite_checks[name] = require_finite(f"rollout {name}", value)
        distributions = {
            name: distribution_summary(f"rollout {name}", value)
            for name in ("rewards", "values", "advantages", "returns")
            if (value := getattr(buffer, name, None)) is not None
        }
        self._write(
            {
                "kind": "rollout_diagnostics",
                "timesteps": int(self.num_timesteps),
                "elapsed_wall_s": monotonic() - self.started,
                "finite_checks": finite_checks,
                "distributions": distributions,
            }
        )

    def _record_update(self) -> None:
        values = dict(getattr(self.model.logger, "name_to_value", {}))
        metric_names = (
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
            "train/value_loss",
            "train/explained_variance",
            "train/policy_gradient_loss",
            "train/loss",
            "train/n_updates",
        )
        metrics = {name: _metric_value(values, name) for name in metric_names}
        finite_checks = {
            name: require_finite(name, value)
            for name, value in metrics.items()
            if value is not None
        }
        gradients = [
            parameter.grad.detach().cpu().numpy()
            for parameter in self.model.policy.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            gradient_values = np.concatenate([value.reshape(-1) for value in gradients])
            finite_checks["gradients"] = require_finite("gradients", gradient_values)
        self._write(
            {
                "kind": "ppo_update_diagnostics",
                "timesteps": int(self.num_timesteps),
                "elapsed_wall_s": monotonic() - self.started,
                "metrics": metrics,
                "finite_checks": finite_checks,
            }
        )
        self._last_update_timesteps = int(self.num_timesteps)

    def _on_rollout_start(self) -> None:
        # This hook runs after the preceding PPO update, so logger values and
        # gradients describe the update that just completed.
        if self.num_timesteps > self._last_update_timesteps:
            self._record_update()

    def _on_training_end(self) -> None:
        # Exact-budget runs end immediately after an update, before another
        # rollout_start hook.  Capture that final update as well.
        if self.num_timesteps > self._last_update_timesteps:
            self._record_update()
        self._write(
            {
                "kind": "summary",
                "timesteps": int(self.num_timesteps),
                "elapsed_wall_s": monotonic() - self.started,
                "mask_steps": self._mask_steps,
                "mask_totals": self._mask_totals,
            }
        )

    def _write(self, row: dict) -> None:
        path = self.output / "training_diagnostics.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
