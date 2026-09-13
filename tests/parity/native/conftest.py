"""Shared fixtures for the native (Rust) parity tests."""

from __future__ import annotations

import pytest

from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.scenarios.generation import generate_manifest
from bus_rl.learning.training.checkpoint import load_model
from tests.support.paths import REFERENCE
from tests.support.reference import reference_run


@pytest.fixture(scope="module")
def reference_bundle():
    if not REFERENCE.exists():
        pytest.skip("reference checkpoint absent; produce runs/diagnose-after first")
    run = reference_run(backend="rust")
    scenarios = generate_manifest("validation", 10)
    load_env = make_env_for_run(scenarios[:1], run)
    model, metadata = load_model(REFERENCE, load_env, run.physical)
    scalar_frame, scalar_traces = evaluate_scenarios(
        scenarios,
        run,
        "ppo",
        model=model,
        model_seed=run.algorithm.seed,
        trace_all=True,
    )
    try:
        yield {
            "run": run,
            "scenarios": scenarios,
            "model": model,
            "metadata": metadata,
            "scalar_frame": scalar_frame,
            "scalar_traces": scalar_traces,
        }
    finally:
        closer = getattr(load_env, "close", None)
        if closer is not None:
            closer()
