"""Scenario payload and tape buffers handed across the FFI boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.native

# `ScenarioPayload` in crates/bus-sim-core/src/lib.rs. `enable_*` default to
# false via serde, but the bridge always emits them.
PAYLOAD_KEYS = {"config", "network", "fleet", "enable_reassign", "enable_short_turn"}


def test_payload_key_set_matches_rust_schema(packed):
    payload, _, _ = packed[0]
    assert set(json.loads(payload)) == PAYLOAD_KEYS


def test_payload_missing_required_field_is_rejected(native, packed):
    payload, arrivals, traffic = packed[0]
    body = json.loads(payload)
    del body["config"]
    with pytest.raises(ValueError):
        native.Kernel(json.dumps(body), arrivals, traffic)


def test_malformed_payload_is_rejected(native, packed):
    _, arrivals, traffic = packed[0]
    with pytest.raises(ValueError):
        native.Kernel("{", arrivals, traffic)


def test_tape_dtypes_and_layout(packed):
    _, arrivals, traffic = packed[1]
    assert arrivals.dtype == np.int8 and arrivals.ndim == 5
    assert traffic.dtype == np.float32 and traffic.ndim == 2
    assert arrivals.flags["C_CONTIGUOUS"] and traffic.flags["C_CONTIGUOUS"]


def test_non_contiguous_tape_is_rejected(native, packed):
    payload, arrivals, traffic = packed[0]
    with pytest.raises(ValueError):
        native.Kernel(payload, arrivals[:, ::-1], traffic)


def test_wrong_dtype_tape_is_rejected(native, packed):
    payload, arrivals, traffic = packed[0]
    with pytest.raises((TypeError, ValueError)):
        native.Kernel(payload, arrivals.astype(np.float32), traffic)


def test_store_add_returns_sequential_index(native, packed):
    store = native.ScenarioStore()
    assert len(store) == 0
    for index, (payload, arrivals, traffic) in enumerate(packed):
        assert store.add(payload, arrivals, traffic) == index
        assert len(store) == index + 1


def test_from_store_rejects_out_of_range_index(native, store, packed):
    with pytest.raises(ValueError, match="out of range"):
        native.Kernel.from_store(store, len(packed))
