"""Map policy requests to executable actions without changing native guards."""

from __future__ import annotations

import numpy as np


def execute_requests(actions, masks, *, invalid_action="error", mask_mode="apply"):
    """Validate the whole batch before returning a separate execution array."""
    requests = np.asarray(actions)
    masks = np.asarray(masks, dtype=bool)
    if requests.ndim != 1 or requests.dtype.kind not in "iu":
        raise ValueError("actions must be a one-dimensional integer array")
    if masks.shape != (len(requests), 221):
        raise ValueError("action/mask batch shape mismatch")
    if np.any(requests < 0) or np.any(requests >= 221):
        raise ValueError("invalid action index outside [0, 220]")
    valid = masks[np.arange(len(requests)), requests]
    if not valid.all() and (invalid_action == "error" or mask_mode == "apply"):
        raise ValueError("invalid action index: infeasible request")
    executed = requests.astype(np.int64, copy=True)
    executed[~valid] = 0
    info = [
        {
            "requested_action_index": int(request),
            "executed_action_index": int(action),
            "requested_action_valid": bool(allowed),
            "valid_action_count": int(mask.sum()),
            "invalid_action_fallback": "none" if allowed else "noop",
        }
        for request, action, allowed, mask in zip(requests, executed, valid, masks, strict=True)
    ]
    return executed, info


class ActionCounters:
    def __init__(self):
        self.steps = self.invalid = self.requested_noop = self.executed_noop = 0

    def add(self, info):
        self.steps += 1
        self.invalid += not info["requested_action_valid"]
        self.requested_noop += info["requested_action_index"] == 0
        self.executed_noop += info["executed_action_index"] == 0

    def metrics(self):
        return {
            "invalid_request_count": self.invalid,
            "invalid_request_rate": self.invalid / self.steps if self.steps else 0.0,
            "requested_noop_count": self.requested_noop,
            "executed_noop_count": self.executed_noop,
            "executed_noop_rate": self.executed_noop / self.steps if self.steps else 0.0,
            "decision_steps": self.steps,
        }
