"""Deterministic controllers used only to drive golden fixtures.

These are not research baselines; they exist to exercise every action family
and a broad range of states in a reproducible way.
"""

from __future__ import annotations

import numpy as np

from bus_rl.control.actions import ACTION_TABLE, action_id

FAMILY_SLOTS: dict[str, list[int]] = {}
for _index, _action in enumerate(ACTION_TABLE):
    FAMILY_SLOTS.setdefault(_action.kind, []).append(_index)

# Rotation order deliberately starts with the rarely-valid families so that a
# 120-decision fixture gives each family a chance to appear.
ROTATION = ("SHORT_TURN", "REASSIGN", "RECALL", "SET_HEADWAY", "DISPATCH", "NOOP")


class CoverageController:
    """Rotates action families and takes the first valid slot in each."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, observation, mask) -> int:
        start = self.calls % len(ROTATION)
        self.calls += 1
        for offset in range(len(ROTATION)):
            family = ROTATION[(start + offset) % len(ROTATION)]
            for slot in FAMILY_SLOTS[family]:
                if mask[slot]:
                    return slot
        return 0


class RandomValidController:
    """Uniform over the currently legal actions; seeded, replayable."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation, mask) -> int:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return action_id("NOOP")
        return int(self.rng.choice(valid))


def make_controller(kind: str, seed: int = 0):
    if kind == "coverage":
        return CoverageController()
    if kind == "random":
        return RandomValidController(seed)
    raise ValueError(f"unknown fixture controller: {kind}")
