"""Fixtures for the Python↔Rust boundary suite.

The suite pins the *interface* at the `bus_sim_native` gate — payload schema,
array shapes/dtypes, error mapping, ownership and lifecycle — rather than
numerical parity against the retired Python oracle engine. Tests skip when the
extension has not been built (`python scripts/build_native.py`).
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

SKIP_REASON = "bus_sim_native not built; run `python scripts/build_native.py`"


def _load_native():
    if importlib.util.find_spec("bus_sim_native") is None:
        return None
    return importlib.import_module("bus_sim_native")


_NATIVE = _load_native()


def pytest_collection_modifyitems(config, items):
    if _NATIVE is not None:
        return
    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def native():
    if _NATIVE is None:
        pytest.skip(SKIP_REASON)
    return _NATIVE


@pytest.fixture(scope="session")
def scenarios():
    from bus_sim.parity.scenarios import CATALOG, build_scenario

    return [build_scenario(CATALOG["normal_m3"]), build_scenario(CATALOG["peak_m3"])]


@pytest.fixture(scope="session")
def packed(scenarios):
    """`(payload_json, arrivals int8, traffic float32)` per scenario.

    Uses the same serialization helpers as the live Rust backend so the test
    exercises the real data-exchange path.
    """
    from bus_rl.execution.environments.rust.bridge import _scenario_tapes, scenario_payload

    out = []
    for scenario in scenarios:
        arrivals, traffic = _scenario_tapes(scenario)
        out.append((scenario_payload(scenario), arrivals, traffic))
    return out


@pytest.fixture(scope="session")
def store(native, packed):
    store = native.ScenarioStore()
    for payload, arrivals, traffic in packed:
        store.add(payload, arrivals, traffic)
    return store


@pytest.fixture
def kernel(native, packed):
    return native.Kernel(*packed[0])
