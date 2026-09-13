"""Single-episode `Kernel` contract: payloads, observation schema, ownership."""

from __future__ import annotations

import numpy as np
import pytest

from .contract import COST_COUNT, MASK_SIZE, OBS_KEYS

pytestmark = pytest.mark.native


def test_cost_fields_match_shared_vocabulary(native):
    from bus_sim.parity.snapshot import COST_FIELDS

    assert native.cost_fields() == list(COST_FIELDS)


def test_reset_contract_payload(kernel):
    result = kernel.reset_contract()
    assert set(result) == {"obs", "mask"}
    assert result["mask"].shape == (MASK_SIZE,)
    assert result["mask"].dtype == np.bool_
    assert bool(result["mask"][0])  # NOOP is always legal


def test_step_contract_payload(kernel):
    kernel.reset_contract()
    result = kernel.step_contract(0)
    assert set(result) == {"obs", "mask", "reward", "terminated", "truncated", "costs"}
    assert isinstance(result["reward"], float)
    assert isinstance(result["terminated"], bool)
    assert result["truncated"] is False
    assert len(result["costs"]) == COST_COUNT
    assert all(isinstance(value, float) for value in result["costs"])
    assert result["mask"].shape == (MASK_SIZE,)


def test_observation_interface_matches_oracle(scenarios, kernel):
    """Same keys/shapes/dtypes as `bus_sim.oracle.observation.observe`.

    This is an interface check, not a physics comparison: the Rust wrapper
    feeds these arrays into the shared Gym observation space.
    """
    from bus_sim.oracle.domain import initial_state
    from bus_sim.oracle.observation import observe

    expected = observe(initial_state(scenarios[0]), scenarios[0])
    observed = kernel.reset_contract()["obs"]
    assert set(observed) == set(expected) == set(OBS_KEYS)
    for key, reference in expected.items():
        value = observed[key]
        assert value.shape == reference.shape, key
        assert value.dtype == reference.dtype, key
        assert value.flags["C_CONTIGUOUS"], key


def test_step_rejects_masked_action_without_mutation(kernel):
    result = kernel.reset_contract()
    masked = np.flatnonzero(~result["mask"])
    if masked.size == 0:
        pytest.skip("scenario has no masked action at reset")
    before = kernel.current_time_s
    with pytest.raises(ValueError, match="invalid action index"):
        kernel.step_contract(int(masked[0]))
    assert kernel.current_time_s == before


def test_step_rejects_out_of_range_action(kernel):
    kernel.reset_contract()
    with pytest.raises(ValueError, match="invalid action index"):
        kernel.step_contract(MASK_SIZE)


def test_step_before_reset_is_rejected(native, packed):
    kernel = native.Kernel(*packed[0])
    with pytest.raises(ValueError, match="reset"):
        kernel.step_contract(0)


def test_action_mask_is_an_owned_copy(kernel):
    kernel.reset_contract()
    first = kernel.action_mask()
    assert first.flags["OWNDATA"]
    first[:] = True
    assert not kernel.action_mask().all()


def test_observation_is_not_overwritten_by_the_next_step(kernel):
    kernel.reset_contract()
    observed = kernel.step_contract(0)["obs"]["stops"]
    snapshot = observed.copy()
    kernel.step_contract(0)
    assert np.array_equal(observed, snapshot)


def test_missing_extension_failure_is_actionable(monkeypatch):
    from bus_rl.execution.environments.rust import bridge

    monkeypatch.setattr(bridge, "native_available", lambda: False)
    with pytest.raises(RuntimeError, match="build_native.py"):
        bridge.require_native()
