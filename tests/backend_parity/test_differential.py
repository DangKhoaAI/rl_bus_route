"""R0.2: replay golden fixtures and report first divergence usefully."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bus_rl.control.actions import ACTION_TABLE
from bus_rl.parity.fixtures import load_fixture, replay_explicit, replay_fixture
from bus_rl.parity.scenarios import CATALOG
from bus_rl.parity.snapshot import compare_records, first_divergence

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_fixture_replays_exactly(name):
    input_payload, expected, _events = load_fixture(FIXTURES / name)
    replayed = replay_fixture(input_payload)
    divergences = compare_records(expected, replayed, limit=5)
    assert divergences == [], divergences

    # The same action list replayed through the public env API must match too.
    explicit = replay_explicit(input_payload["spec"], input_payload["actions"])
    assert compare_records(expected, explicit, limit=5) == []


def test_corrupted_fixture_reports_first_divergence():
    _input_payload, expected, _events = load_fixture(FIXTURES / "normal_m3")
    corrupted = {key: np.array(value, copy=True) for key, value in expected.items()}
    corrupted["obs__stops"][7, 0, 0, 0, 0] += 1e-3
    divergence = first_divergence(expected, corrupted)
    assert divergence is not None
    assert "obs[stops]" in divergence["path"]
    assert divergence["expected"] != divergence["actual"]

    # An exact-category divergence (mask bit) must also be caught.
    mask_corrupted = {key: np.array(value, copy=True) for key, value in expected.items()}
    mask_corrupted["mask"][3, 10] = ~mask_corrupted["mask"][3, 10]
    mask_divergence = first_divergence(expected, mask_corrupted)
    assert mask_divergence is not None
    assert mask_divergence["path"].startswith("mask")


def test_events_are_action_driven_and_complete():
    families = {action.kind for action in ACTION_TABLE}
    for name in sorted(CATALOG):
        input_payload, _, events = load_fixture(FIXTURES / name)
        assert input_payload["fixture_schema_version"] == 1
        assert len(input_payload["actions"]) == input_payload["num_steps"]
        assert all(0 <= action < len(ACTION_TABLE) for action in input_payload["actions"])
        mission_events = [row for row in events["accepted_actions"] if row["kind"] != "NOOP"]
        assert all(row["kind"] in families for row in events["accepted_actions"])
        if events["accepted_actions"]:
            assert mission_events
            assert all("step" in row and "time_s" in row for row in mission_events)
        assert all("time_s" in row and "bus_id" in row for row in events["departures"])
