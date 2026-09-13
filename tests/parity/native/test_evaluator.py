"""R3.2/R3.3 evaluator parity, backend selection and checkpoint compatibility.

Skipped when the `bus_sim_native` extension has not been built
(`python scripts/build_native.py`).
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bus_rl.cli import main as cli_main
from bus_rl.config import ControlConfig, RuntimeConfig, load_run_config
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.training.checkpoint import assert_backend_compatible, run_metadata
from bus_sim.parity.scenarios import CATALOG
from tests.support.compare import assert_frames, assert_obs
from tests.support.paths import ROOT
from tests.support.scenarios import cached_scenario

pytest.importorskip("bus_sim_native")

from bus_rl.execution.environments.rust.environment import NativeBusDispatchEnv

pytestmark = pytest.mark.native


@pytest.mark.parametrize("method", ["fixed", "threshold", "proportional", "random"])
def test_evaluator_matches_across_backends(method):
    run = load_run_config(ROOT / "configs" / "eval.toml")
    scenarios = [generate_scenario(2001), generate_scenario(4001, variant="burst")]
    python_run = replace(run, runtime=RuntimeConfig(backend="python"))
    rust_run = replace(run, runtime=RuntimeConfig(backend="rust"))
    frame_py, traces_py = evaluate_scenarios(scenarios, python_run, method, trace_index=0)
    frame_rs, traces_rs = evaluate_scenarios(scenarios, rust_run, method, trace_index=0)
    assert_frames(frame_py, frame_rs)
    assert len(traces_py[0]) == len(traces_rs[0])
    for left, right in zip(traces_py[0], traces_rs[0], strict=True):
        assert left["action"] == right["action"]
        assert left["time_s"] == right["time_s"]
        assert list(left["queues"]) == list(right["queues"])
        assert list(left["headway_targets"]) == list(right["headway_targets"])
        assert left["buses"] == right["buses"]
        for key in ("waiting", "onboard", "generated", "abandoned", "completed"):
            assert left[key] == right[key]


def test_forecast_context_matches_across_backends():
    from bus_rl.learning.forecasting import HistoricalForecaster

    scenario = generate_scenario(2001)
    forecaster = HistoricalForecaster(tick_s=scenario.config.tick_s)
    forecaster.fit([scenario.arrival_tape], scenario.config)
    oracle = BusDispatchEnv(
        [scenario], scenario.config, forecaster=forecaster, control=ControlConfig()
    )
    native = NativeBusDispatchEnv(
        [scenario], scenario.config, forecaster=forecaster, control=ControlConfig()
    )
    observation_py, _ = oracle.reset(seed=0, options={"scenario_index": 0})
    observation_rs, _ = native.reset(seed=0, options={"scenario_index": 0})
    assert observation_py["context"][2] == 1.0
    assert observation_rs["context"][2] == 1.0
    assert_obs(observation_py, observation_rs)
    for _ in range(4):
        step_py = oracle.step(0)
        step_rs = native.step(0)
        assert_obs(step_py[0], step_rs[0])
        np.testing.assert_allclose(step_py[1], step_rs[1], rtol=1e-9, atol=1e-9)


def test_zero_demand_metrics_match_and_stay_none():
    run = load_run_config(ROOT / "configs" / "eval.toml")
    scenario = cached_scenario(CATALOG["zero_m3"])
    python_run = replace(run, runtime=RuntimeConfig(backend="python"))
    rust_run = replace(run, runtime=RuntimeConfig(backend="rust"))
    frame_py, _ = evaluate_scenarios([scenario], python_run, "fixed")
    frame_rs, _ = evaluate_scenarios([scenario], rust_run, "fixed")
    assert_frames(frame_py, frame_rs)
    assert frame_py["mean_wait"].isna().all()
    assert frame_py["generated"].iloc[0] == 0


def test_evaluation_outputs_render_through_report_pipeline(tmp_path):
    from bus_rl.evaluation.plots import render_plots
    from bus_rl.evaluation.runner import write_results

    run = load_run_config(ROOT / "configs" / "eval.toml")
    rust_run = replace(run, runtime=RuntimeConfig(backend="rust"))
    frame, traces = evaluate_scenarios(
        [generate_scenario(2001)], rust_run, "threshold", trace_index=0
    )
    write_results(frame, tmp_path, traces)
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "events.jsonl").exists()
    render_plots(frame, tmp_path / "plots", traces=next(iter(traces.values())))
    assert (tmp_path / "plots" / "cost_bars.png").exists()


def test_backend_selection_defaults_to_python_and_requires_extension(monkeypatch):
    from bus_rl.execution.environments import factory

    run = load_run_config(ROOT / "configs" / "eval.toml")
    assert run.runtime.backend == "python"
    with pytest.raises(ValueError):
        RuntimeConfig(backend="nope")
    rust_run = replace(run, runtime=RuntimeConfig(backend="rust"))
    monkeypatch.setattr("bus_rl.execution.environments.rust.bridge.native_available", lambda: False)
    with pytest.raises(RuntimeError):
        factory.make_env_for_run([generate_scenario(1)], rust_run)


def test_metadata_records_backend_and_native_build():
    run = load_run_config(ROOT / "configs" / "eval.toml")
    python_meta = run_metadata(run)
    assert python_meta["backend"] == "python"
    assert python_meta["native_build"] is None
    rust_meta = run_metadata(replace(run, runtime=RuntimeConfig(backend="rust")))
    assert rust_meta["backend"] == "rust"
    native = rust_meta["native_build"]
    assert native["library_sha256"]
    # The recorded hash must be the binary actually installed, and at least one
    # build record must describe it (the frozen R4 record may not).
    import hashlib

    actual = hashlib.sha256((ROOT / "src" / "bus_sim_native.so").read_bytes()).hexdigest()
    assert native["library_sha256"] == actual
    assert native["library_sha256_runtime"] == actual
    assert native["build_record_matches_runtime"]


def test_cross_backend_checkpoint_rules():
    assert_backend_compatible({"backend": "python", "backend_parity_verified": True}, "rust")
    assert_backend_compatible({}, "rust")
    assert_backend_compatible({"backend": "rust"}, "rust")
    with pytest.raises(ValueError):
        assert_backend_compatible({"backend": "python", "backend_parity_verified": False}, "rust")


def test_cli_backend_entrypoints(tmp_path):
    data = tmp_path / "data"
    cli_main(
        [
            "generate",
            "--config",
            str(ROOT / "configs/base.toml"),
            "--output",
            str(data),
            "--counts",
            "train=1,validation=1",
        ]
    )
    train_out = tmp_path / "train-rust"
    cli_main(
        [
            "train",
            "--config",
            str(ROOT / "configs/pilot.toml"),
            "--manifest",
            str(data / "manifest.json"),
            "--seed",
            "11",
            "--output",
            str(train_out),
            "--timesteps",
            "32",
            "--n-envs",
            "1",
            "--eval-limit",
            "1",
            "--backend",
            "rust",
        ]
    )
    metadata = json.loads((train_out / "metadata.json").read_text())
    assert metadata["backend"] == "rust"
    assert metadata["native_build"]["library_sha256"]
    assert (train_out / "best.zip").exists()

    outputs = {}
    for backend in ("python", "rust"):
        out = tmp_path / f"eval-{backend}"
        cli_main(
            [
                "evaluate",
                "--config",
                str(ROOT / "configs/eval.toml"),
                "--manifest",
                str(data / "manifest.json"),
                "--split",
                "validation",
                "--method",
                "threshold",
                "--output",
                str(out),
                "--backend",
                backend,
            ]
        )
        outputs[backend] = pd.read_csv(out / "results.csv")
    np.testing.assert_allclose(
        outputs["python"]["total_cost"], outputs["rust"]["total_cost"], rtol=1e-9, atol=1e-9
    )
