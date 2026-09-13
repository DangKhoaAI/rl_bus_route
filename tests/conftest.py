"""Shared pytest configuration.

The native Rust kernel is the default run path. Tests that exercise the
deprecated Python oracle (either the engine directly or `BusDispatchEnv`) are
marked `legacy_python` and skipped unless `--legacy-python` is passed, so a
plain `pytest` run covers the maintained backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bus_sim.oracle import domain

# Tests coupled to the deprecated Python oracle engine or its Gym wrapper.
LEGACY_TEST_GLOBS = (
    "unit/bus_sim/oracle/test_*.py",
    "integration/bus_rl/execution/environments/python/test_*.py",
    "integration/bus_rl/learning/baselines/test_*.py",
    "integration/bus_rl/learning/training/test_*.py",
)
TESTS_DIR = Path(__file__).resolve().parent


def pytest_addoption(parser):
    parser.addoption(
        "--legacy-python",
        action="store_true",
        default=False,
        help="run tests for the deprecated Python oracle backend",
    )


def _is_legacy(item) -> bool:
    path = getattr(item, "path", None)
    if path is None:
        return False
    relative = Path(path).relative_to(TESTS_DIR)
    return any(relative.match(glob) for glob in LEGACY_TEST_GLOBS)


def pytest_collection_modifyitems(config, items):
    run_legacy = config.getoption("--legacy-python")
    skip = pytest.mark.skip(reason="deprecated Python backend; pass --legacy-python to run")
    for item in items:
        if _is_legacy(item):
            item.add_marker(pytest.mark.legacy_python)
            if not run_legacy:
                item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _small_torch_threads():
    """Tests are not benchmarks: 2 threads is much faster here (see R4 tuning)."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _enable_conservation_checks():
    domain.CONSERVATION_CHECKS = True
    yield
    domain.CONSERVATION_CHECKS = False
