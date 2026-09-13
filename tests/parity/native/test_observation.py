"""R2 differential tests: native observations and the mask cache vs the oracle.

Coverage is the frozen R0 fixture set: every observation channel and all 221
mask bits are compared at reset and after every step. The native kernel builds
observations incrementally (no scan of the full finished archive) but must match
the oracle within the frozen tolerances.

Skipped when the `bus_sim_native` extension has not been built
(`python scripts/build_native.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

from bus_sim.parity.fixtures import load_fixture
from bus_sim.parity.scenarios import CATALOG
from tests.support.contract import OBS_KEYS
from tests.support.paths import GOLDEN
from tests.support.rust_bridge import make_kernel, replay_observations

pytest.importorskip("bus_sim_native")

pytestmark = pytest.mark.native

OBS_RTOL = 1e-6
OBS_ATOL = 1e-6


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_observation_and_mask_match_oracle(name):
    payload, expected, _ = load_fixture(GOLDEN / name)
    observations, masks = replay_observations(payload["spec"], payload["actions"])

    assert len(observations) == expected["mask"].shape[0]
    actual_masks = np.stack(masks)
    assert np.array_equal(actual_masks, expected["mask"].astype(bool)), name

    for key in OBS_KEYS:
        oracle = expected[f"obs__{key}"]
        actual = np.stack([observation[key] for observation in observations])
        assert actual.shape == oracle.shape, (key, actual.shape, oracle.shape)
        assert actual.dtype == oracle.dtype == np.float32
        close = np.isclose(oracle, actual, rtol=OBS_RTOL, atol=OBS_ATOL)
        if not bool(close.all()):
            index = np.unravel_index(np.argmax(np.abs(oracle - actual)), oracle.shape)
            raise AssertionError(
                f"{name}:{key} diverges at {index}: oracle={oracle[index]} native={actual[index]}"
            )


def test_mask_cache_validates_and_reads_do_not_recompute():
    payload, _, _ = load_fixture(GOLDEN / "traffic_m3")
    kernel = make_kernel(payload["spec"])
    assert kernel.debug_validate_mask()
    baseline = kernel.mask_computations

    first = np.asarray(kernel.action_mask())
    for _ in range(5):
        assert np.array_equal(np.asarray(kernel.action_mask()), first)
    assert kernel.mask_computations == baseline

    for action in payload["actions"]:
        kernel.debug_step(int(action))
        assert kernel.debug_validate_mask()
    assert kernel.mask_computations == baseline + len(payload["actions"])


def test_returned_mask_is_an_independent_copy():
    kernel = make_kernel(CATALOG["normal_m1"])
    expected = np.asarray(kernel.action_mask()).copy()
    scratch = np.asarray(kernel.action_mask())
    scratch[:] = False
    again = np.asarray(kernel.action_mask())
    assert np.array_equal(again, expected)
    assert bool(again[0])  # NOOP stays valid


def test_reset_recomputes_the_cache():
    kernel = make_kernel(CATALOG["peak_m3"])
    first_valid = int(np.flatnonzero(kernel.action_mask())[0])
    kernel.debug_step(first_valid)
    before = kernel.mask_computations
    kernel.reset()
    assert kernel.mask_computations == before + 1
    assert kernel.debug_validate_mask()
    assert bool(kernel.action_mask()[0])


def test_out_of_range_and_masked_actions_are_rejected():
    kernel = make_kernel(CATALOG["normal_m3"])
    masked = int(np.flatnonzero(~np.asarray(kernel.action_mask()))[0])
    for action in (masked, 221, 10_000):
        before = kernel.current_time_s
        with pytest.raises(ValueError):
            kernel.debug_step(action)
        assert kernel.current_time_s == before
        assert kernel.debug_validate_mask()
