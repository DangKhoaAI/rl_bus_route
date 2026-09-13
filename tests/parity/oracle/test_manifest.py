"""R0.1: the committed oracle manifest must verify and stay self-consistent."""

from __future__ import annotations

import json
import os

import pytest

from bus_rl.provenance import canonical_hash
from bus_sim.parity.manifest import verify_manifest
from tests.support.paths import ROOT

MANIFEST = ROOT / "reports" / "rust-migration" / "oracle-manifest.json"
DEEP = os.environ.get("BUS_RL_DEEP") == "1"


def _payload() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_references_and_hashes_verify():
    assert verify_manifest(_payload(), ROOT) == []


@pytest.mark.skipif(not DEEP, reason="set BUS_RL_DEEP=1 for the 1,200-scenario regeneration check")
def test_scenario_order_regenerates():
    assert verify_manifest(_payload(), ROOT, regenerate_scenarios=True) == []


def test_manifest_self_hash_is_consistent():
    payload = _payload()
    stored = payload.pop("self_hash")
    assert canonical_hash(payload) == stored


def test_contract_is_frozen_and_complete():
    payload = _payload()
    action = payload["contract"]["action"]
    assert action["discrete_size"] == 221
    assert sum(action["family_counts"].values()) == 221
    assert action["noop_index"] == 0
    observation = payload["contract"]["observation"]
    assert observation["obs_version"] == 2
    assert "stops" in observation["keys"] and "context" in observation["keys"]
    assert payload["contract"]["cost_components"]
    assert payload["contract"]["terminal"]["settlement"]
    assert payload["discrepancies"]
    for discrepancy in payload["discrepancies"]:
        assert discrepancy["id"] and discrepancy["reproducer"] and discrepancy["port_decision"]
