"""R1 differential tests: the native kernel must reproduce the R0 oracle.

These tests are skipped when the `bus_sim` extension has not been built
(`python scripts/build_native.py`). Observation/mask parity is R2 and is not
asserted here beyond the guard-derived action mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bus_rl.parity.fixtures import load_fixture
from bus_rl.parity.scenarios import CATALOG
from bus_rl.parity.snapshot import compare_records
from tests.backend_parity.rust_bridge import (
    KERNEL_COMPARE_KEYS,
    kernel_flat,
    make_kernel,
    run_actions,
)

pytest.importorskip("bus_sim")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_kernel_matches_oracle_state_and_costs(name):
    payload, expected, _ = load_fixture(FIXTURES / name)
    flat = kernel_flat(payload["spec"], payload["actions"])
    expected_subset = {key: expected[key] for key in KERNEL_COMPARE_KEYS if key in expected}
    actual_subset = {key: flat[key] for key in KERNEL_COMPARE_KEYS if key in flat}
    divergences = compare_records(expected_subset, actual_subset, limit=5)
    assert divergences == [], divergences


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_kernel_matches_oracle_events(name):
    payload, _, events = load_fixture(FIXTURES / name)
    record = run_actions(payload["spec"], payload["actions"])
    assert record["departures"] == events["departures"]
    assert record["accepted_actions"] == events["accepted_actions"]


def test_kernel_reset_is_deterministic():
    spec = CATALOG["burst_m3"]
    kernel = make_kernel(spec)
    first = kernel.debug_snapshot()
    kernel.reset()
    second = kernel.debug_snapshot()
    assert first["time_s"] == second["time_s"] == 0
    for key in ("counters", "vehicles", "cohorts", "headway_targets"):
        assert np.array_equal(np.asarray(first[key]), np.asarray(second[key]))


def test_kernel_rejects_invalid_action_before_mutation():
    payload, _expected, _ = load_fixture(FIXTURES / "normal_m3")
    kernel = make_kernel(payload["spec"])
    invalid = int(np.flatnonzero(~kernel.action_mask())[0])
    before = kernel.debug_snapshot()
    with pytest.raises(ValueError):
        kernel.debug_step(invalid)
    after = kernel.debug_snapshot()
    assert before["time_s"] == after["time_s"]
    assert np.array_equal(before["counters"], after["counters"])
    assert np.array_equal(before["vehicles"], after["vehicles"])


def test_kernel_terminal_settlement_is_charged_once():
    _, expected, _ = load_fixture(FIXTURES / "backlog_m3")
    payload, _, _ = load_fixture(FIXTURES / "backlog_m3")
    flat = kernel_flat(payload["spec"], payload["actions"])
    terminal = flat["costs__terminal_unfinished_count"]
    assert terminal[:-1].sum() == 0.0
    assert terminal[-1] > 0.0
    assert flat["terminal_settled"][:-1].sum() == 0
    assert np.array_equal(flat["counters"], expected["counters"])


def test_kernel_consumes_config_ticks_per_interval():
    payload, _, _ = load_fixture(FIXTURES / "zero_m3")
    flat = kernel_flat(payload["spec"], payload["actions"])
    ticks = flat["tick__time_s"].shape[0]
    assert ticks == len(payload["actions"]) * 4
