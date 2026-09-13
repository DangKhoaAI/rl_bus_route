"""Evaluator/provenance payloads crossing back from Rust into Python."""

from __future__ import annotations

import pytest

from bus_sim.parity.controllers import CoverageController

pytestmark = pytest.mark.native

DEPARTURE_KEYS = {"time_s", "route_id", "direction", "bus_id", "pattern"}
BUS_KEYS = {"id", "phase", "route_id", "pattern", "load"}


@pytest.fixture
def finished_kernel(native, packed):
    """A kernel driven through a full coverage episode (all action families)."""
    kernel = native.Kernel(*packed[0])
    kernel.reset_contract()
    controller = CoverageController()
    for _ in range(500):
        if kernel.terminal():
            break
        kernel.step_contract(controller.act(None, kernel.action_mask()))
    return kernel


def test_summary_inputs_feed_the_python_adapter(finished_kernel):
    from bus_rl.evaluation.summary import from_native_payload

    payload = finished_kernel.episode_summary_inputs()
    assert {"time_s", "departures", "accepted_kinds", "vehicles", "cohorts"} <= set(payload)
    inputs = from_native_payload(payload)
    assert inputs is not None


def test_departure_rows_use_the_union_pattern_type(finished_kernel):
    """Scheduled departures carry `int`, extra (short-turn) ones carry `str` (D3)."""
    departures = finished_kernel.episode_summary_inputs()["departures"]
    assert departures, "a coverage episode must emit departures"
    for row in departures:
        assert set(row) == DEPARTURE_KEYS
        assert isinstance(row["time_s"], int)
        assert isinstance(row["pattern"], (int, str))


def test_trace_snapshot_schema(finished_kernel):
    trace = finished_kernel.trace_snapshot()
    assert {
        "time_s",
        "queues",
        "headway_targets",
        "buses",
        "waiting",
        "onboard",
        "generated",
        "abandoned",
        "completed",
    } <= set(trace)
    for bus in trace["buses"]:
        assert set(bus) == BUS_KEYS
        assert isinstance(bus["phase"], str)
        assert isinstance(bus["pattern"], str)
        assert isinstance(bus["load"], int)


def test_native_build_info_reports_the_installed_library():
    from bus_rl.execution.environments.rust.bridge import PROJECT_ROOT, native_build_info
    from bus_rl.provenance import file_hash

    info = native_build_info()
    assert info is not None
    assert info["crate"] == "bus-sim-python"
    library = PROJECT_ROOT / "src" / "bus_sim_native.so"
    assert info["library_sha256_runtime"] == file_hash(library)
