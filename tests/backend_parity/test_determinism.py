"""R0.1: identical resets must reproduce identical state/observation/mask."""

from __future__ import annotations

import numpy as np
import pytest

from bus_rl.parity.fixtures import new_env
from bus_rl.parity.scenarios import CATALOG
from bus_rl.parity.snapshot import numpy_state_snapshot

CASES = ["zero_m3", "normal_m3", "burst_m3", "traffic_m3"]


@pytest.mark.parametrize("name", CASES)
def test_reset_is_deterministic(name):
    spec = CATALOG[name]
    env_a = new_env(spec)
    env_b = new_env(spec)
    snap_a = numpy_state_snapshot(env_a.state)
    snap_b = numpy_state_snapshot(env_b.state)
    for key in snap_a:
        assert np.array_equal(snap_a[key], snap_b[key]), f"state[{key}] differs on reset"
    for key, value in env_a._observation().items():
        assert np.array_equal(value, env_b._observation()[key]), f"obs[{key}] differs on reset"
    assert np.array_equal(env_a.action_masks(), env_b.action_masks())


@pytest.mark.parametrize("name", CASES)
def test_zero_action_interval_is_reproducible(name):
    spec = CATALOG[name]
    env_a, env_b = new_env(spec), new_env(spec)
    for _ in range(3):
        env_a.step(0)
        env_b.step(0)
    assert np.array_equal(
        numpy_state_snapshot(env_a.state)["counters"],
        numpy_state_snapshot(env_b.state)["counters"],
    )
    assert env_a.state.current_time_s == env_b.state.current_time_s
