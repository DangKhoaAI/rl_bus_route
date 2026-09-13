"""`BatchKernel` contract: batched payloads, slot validation and atomicity."""

from __future__ import annotations

import numpy as np
import pytest

from .contract import BATCH_OBS_SHAPES, COST_COUNT, MASK_SIZE

pytestmark = pytest.mark.native

CAPACITY = 3


@pytest.fixture
def batch(native, store):
    return native.BatchKernel.from_store(store, CAPACITY)


def test_reset_batch_payload(batch):
    result = batch.reset_batch([0, 1], [0, 1])
    assert set(result) == {"slot_ids", "obs", "mask"}
    assert result["slot_ids"].tolist() == [0, 1]
    assert result["mask"].shape == (2, MASK_SIZE)
    assert result["mask"].dtype == np.bool_
    for key, shape in BATCH_OBS_SHAPES.items():
        assert result["obs"][key].shape == (2, *shape), key


def test_step_batch_payload(batch):
    batch.reset_batch([0, 1], [0, 1])
    result = batch.step_batch([0, 1], [0, 0])
    assert set(result) == {
        "slot_ids",
        "obs",
        "mask",
        "reward",
        "terminated",
        "truncated",
        "costs",
    }
    assert result["slot_ids"].tolist() == [0, 1]
    assert result["mask"].shape == (2, MASK_SIZE)
    assert result["reward"].shape == (2,)
    assert result["terminated"].shape == (2,) and result["terminated"].dtype == np.bool_
    assert result["truncated"].shape == (2,) and not result["truncated"].any()
    assert result["costs"].shape == (2, COST_COUNT)
    for key, shape in BATCH_OBS_SHAPES.items():
        assert result["obs"][key].shape == (2, *shape), key


def test_reset_batch_validates_slots_and_scenarios(batch, packed):
    with pytest.raises(ValueError, match="duplicate slot"):
        batch.reset_batch([0, 0], [0, 1])
    with pytest.raises(ValueError, match="scenario index .* out of range"):
        batch.reset_batch([0], [len(packed) + 5])
    with pytest.raises(ValueError, match="at least one slot"):
        batch.reset_batch([], [])
    with pytest.raises(ValueError, match="out of range"):
        batch.reset_batch([CAPACITY], [0])
    with pytest.raises(ValueError, match="slot count"):
        batch.reset_batch([0], [0, 1])


def test_step_batch_validates_slots_and_actions(batch):
    with pytest.raises(ValueError, match="has not been reset"):
        batch.step_batch([0], [0])
    batch.reset_batch([0, 1], [0, 1])
    with pytest.raises(ValueError, match="duplicate slot"):
        batch.step_batch([0, 0], [0, 0])
    with pytest.raises(ValueError, match="out of range"):
        batch.step_batch([CAPACITY], [0])
    with pytest.raises(ValueError, match="has not been reset"):
        batch.step_batch([2], [0])
    with pytest.raises(ValueError, match="slot count"):
        batch.step_batch([0], [0, 0])
    with pytest.raises(ValueError, match="invalid action index"):
        batch.step_batch([0], [MASK_SIZE])
    masked = int(np.flatnonzero(~np.asarray(batch.mask_batch([0]))[0])[0])
    with pytest.raises(ValueError, match="invalid action index"):
        batch.step_batch([0], [masked])


def test_rejected_batch_leaves_every_slot_unchanged(batch):
    batch.reset_batch([0, 1], [0, 1])
    batch.step_batch([0, 1], [0, 0])
    before = batch.current_time_s_batch([0, 1]).tolist()
    for slots, actions in (
        ([0, 0], [0, 0]),
        ([CAPACITY], [0]),
        ([2], [0]),
        ([0], [0, 0]),
        ([0], [MASK_SIZE]),
    ):
        with pytest.raises(ValueError):
            batch.step_batch(slots, actions)
    assert batch.current_time_s_batch([0, 1]).tolist() == before
    assert not batch.poisoned


def test_batch_helpers_shape_and_liveness(batch):
    batch.reset_batch([0, 1], [0, 1])
    assert np.asarray(batch.mask_batch([0, 1])).shape == (2, MASK_SIZE)
    assert batch.current_time_s_batch([0, 1]).shape == (2,)
    assert batch.capacity() == CAPACITY

    summaries = batch.summary_inputs_batch([0, 1])
    assert len(summaries) == 2
    assert set(summaries[0]) == set(summaries[1])

    trace = batch.trace_snapshot_slot(0)
    assert len(trace["queues"]) == len(trace["headway_targets"])

    with pytest.raises(ValueError, match="has not been reset"):
        batch.trace_snapshot_slot(2)


def test_set_reward_changes_returned_reward(batch):
    """Regression: the kernel must honor run.reward, not only its defaults.

    ``n_ref`` only rescales the returned reward, so halving it must exactly
    double the reward for an identical reset/step. Before the wiring fix the
    kernel silently ignored every reward field except the defaults.
    """
    batch.reset_batch([0], [0])
    baseline = batch.step_batch([0], [0])["reward"][0]
    batch.reset_batch([0], [0])
    batch.set_reward(
        waiting=1.0,
        onboard=0.25,
        crowding=0.5,
        active=0.5,
        deadhead=0.5,
        fairness=1.0,
        first_denied=5.0,
        abandoned=60.0,
        mission=2.0,
        unfinished=60.0,
        n_ref=1500.0,
    )
    scaled = batch.step_batch([0], [0])["reward"][0]
    assert scaled == pytest.approx(2.0 * baseline)
