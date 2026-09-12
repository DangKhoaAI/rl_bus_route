"""R3 integration tests: native Gym wrapper, evaluator parity, backend choice.

Skipped when the `bus_sim` extension has not been built
(`python scripts/build_native.py`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bus_rl.config import ControlConfig, RuntimeConfig, load_run_config
from bus_rl.data.scenario import generate_scenario
from bus_rl.domain import SimConfig, StepCosts
from bus_rl.env.bus_dispatch import BusDispatchEnv
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.parity.fixtures import load_fixture
from bus_rl.parity.scenarios import CATALOG, CONTROL_FLAGS, build_scenario
from bus_rl.training.checkpoint import assert_backend_compatible, run_metadata

pytest.importorskip("bus_sim")

from bus_rl.env.native_bus_dispatch import NativeBusDispatchEnv

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


def _control(spec: dict) -> ControlConfig:
    enable_reassign, enable_short_turn = CONTROL_FLAGS[spec.get("control", "M3")]
    return ControlConfig(enable_reassign=enable_reassign, enable_short_turn=enable_short_turn)


def _assert_obs(left: dict, right: dict, *, rtol: float = 1e-6, atol: float = 1e-6) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        np.testing.assert_allclose(value, right[key], rtol=rtol, atol=atol, err_msg=key)


def _assert_value(x, y, label: str) -> None:
    if x is None or y is None:
        assert x is None and y is None, (label, x, y)
    elif isinstance(x, float) or isinstance(y, float):
        if np.isnan(x) or np.isnan(y):
            assert np.isnan(x) and np.isnan(y), (label, x, y)
        else:
            np.testing.assert_allclose(float(x), float(y), rtol=1e-9, atol=1e-9, err_msg=label)
    else:
        assert x == y, (label, x, y)


def _assert_frames(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right)
    for column in left.columns:
        if column == "wall_s":
            continue
        for index, (x, y) in enumerate(zip(left[column], right[column], strict=True)):
            _assert_value(x, y, f"{column}[{index}]")


# --------------------------------------------------------------------------- #
# R3.1 native Gym contract
# --------------------------------------------------------------------------- #


def test_native_env_contract_and_masked_rejection():
    import gymnasium as gym

    scenario = generate_scenario(2001)
    env = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 221
    observation, info = env.reset(seed=5)
    assert info == {}
    assert env.observation_space.contains(observation)
    masked = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(masked)
    terminated = False
    decisions = scenario.config.horizon_s // scenario.config.control_interval_s
    for _ in range(decisions):
        observation, reward, terminated, truncated, info = env.step(0)
        assert observation["stops"].shape == (4, 2, 8, 7)
        assert isinstance(info["costs"], StepCosts)
        assert np.isfinite(reward)
        assert not truncated
    assert terminated


@pytest.mark.parametrize(
    "name",
    ["zero_m3", "normal_m3", "capacity_m3", "abandon_m3", "burst_m3", "traffic_m3"],
)
def test_native_env_matches_python_env_at_every_boundary(name):
    payload, _, _ = load_fixture(FIXTURES / name)
    scenario = build_scenario(payload["spec"])
    control = _control(payload["spec"])
    oracle = BusDispatchEnv([scenario], scenario.config, control=control)
    native = NativeBusDispatchEnv([scenario], scenario.config, control=control)

    observation_py, _ = oracle.reset(seed=0, options={"scenario_index": 0})
    observation_rs, _ = native.reset(seed=0, options={"scenario_index": 0})
    _assert_obs(observation_py, observation_rs)
    assert np.array_equal(oracle.action_masks(), native.action_masks())

    for action in payload["actions"]:
        step_py = oracle.step(int(action))
        step_rs = native.step(int(action))
        _assert_obs(step_py[0], step_rs[0])
        np.testing.assert_allclose(step_py[1], step_rs[1], rtol=1e-9, atol=1e-9)
        assert step_py[2] == step_rs[2]
        assert step_py[3] == step_rs[3]
        for field in StepCosts.__dataclass_fields__:
            _assert_value(
                getattr(step_py[4]["costs"], field),
                getattr(step_rs[4]["costs"], field),
                field,
            )
        assert np.array_equal(oracle.action_masks(), native.action_masks())
        if step_py[2]:
            break


def test_returned_arrays_are_owned_and_survive_later_steps():
    scenario = generate_scenario(2001)
    env = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    observation, _ = env.reset(seed=0)
    saved_obs = {key: np.array(value, copy=True) for key, value in observation.items()}
    for _ in range(3):
        env.step(0)
    env.reset(seed=1)
    # Saved rollout inputs are untouched by later native operations.
    for key, value in saved_obs.items():
        np.testing.assert_array_equal(value, observation[key])
    # Mutating a returned mask cannot corrupt the native cache.
    scratch = env.action_masks()
    scratch[:] = False
    assert env.action_masks().any()


def test_interleaved_native_envs_do_not_share_state():
    scenario = generate_scenario(2001)
    first = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    second = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    solo = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    first.reset(seed=0)
    second.reset(seed=0)
    solo.reset(seed=0)
    for _ in range(5):
        left = first.step(0)
        right = second.step(0)
        reference = solo.step(0)
        _assert_obs(left[0], reference[0])
        _assert_obs(right[0], reference[0])
        assert left[1] == right[1] == reference[1]


def test_native_envs_share_one_scenario_store():
    scenarios = [generate_scenario(2001), generate_scenario(2002), generate_scenario(2003)]
    envs = [
        NativeBusDispatchEnv(scenarios, scenarios[0].config, control=ControlConfig())
        for _ in range(3)
    ]
    assert envs[0]._scenario_store is envs[1]._scenario_store is envs[2]._scenario_store
    assert len(envs[0]._scenario_store) == 3
    # Kernels are per env (independent episode state) but share the packed tapes.
    assert envs[0]._kernels[0] is not envs[1]._kernels[0]
    envs[0].reset(seed=0, options={"scenario_index": 0})
    envs[1].reset(seed=0, options={"scenario_index": 0})
    envs[0].step(0)
    assert envs[0].kernel.current_time_s != envs[1].kernel.current_time_s


def test_action_masks_reuse_the_step_mask_without_recompute():
    scenario = generate_scenario(2001)
    env = NativeBusDispatchEnv([scenario], scenario.config, control=ControlConfig())
    env.reset(seed=0)
    before = env.kernel.mask_computations
    for _ in range(5):
        env.action_masks()
    assert env.kernel.mask_computations == before
    env.step(0)
    assert env.kernel.mask_computations == before + 1
    for _ in range(5):
        env.action_masks()
    assert env.kernel.mask_computations == before + 1


def test_dummy_vec_env_auto_reset():
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = SimConfig(horizon_s=480, demand_end_s=240)
    scenario = generate_scenario(2001, config)

    def make(seed: int):
        def _init():
            env = NativeBusDispatchEnv([scenario], config, control=ControlConfig())
            env.reset(seed=seed)
            return env

        return _init

    vec = DummyVecEnv([make(1), make(2)])
    try:
        observation = vec.reset()
        assert observation["stops"].shape == (2, 4, 2, 8, 7)
        for _ in range(6):
            observation, rewards, _, _ = vec.step(np.zeros(2, dtype=np.int64))
            assert np.isfinite(rewards).all()
            assert observation["stops"].shape == (2, 4, 2, 8, 7)
    finally:
        vec.close()


# --------------------------------------------------------------------------- #
# R3.2 evaluator, traces and forecast
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["fixed", "threshold", "proportional", "random"])
def test_evaluator_matches_across_backends(method):
    run = load_run_config(ROOT / "configs" / "eval.toml")
    scenarios = [generate_scenario(2001), generate_scenario(4001, variant="burst")]
    python_run = replace(run, runtime=RuntimeConfig(backend="python"))
    rust_run = replace(run, runtime=RuntimeConfig(backend="rust"))
    frame_py, traces_py = evaluate_scenarios(scenarios, python_run, method, trace_index=0)
    frame_rs, traces_rs = evaluate_scenarios(scenarios, rust_run, method, trace_index=0)
    _assert_frames(frame_py, frame_rs)
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
    from bus_rl.forecasting.historical import HistoricalForecaster

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
    _assert_obs(observation_py, observation_rs)
    for _ in range(4):
        step_py = oracle.step(0)
        step_rs = native.step(0)
        _assert_obs(step_py[0], step_rs[0])
        np.testing.assert_allclose(step_py[1], step_rs[1], rtol=1e-9, atol=1e-9)


def test_zero_demand_metrics_match_and_stay_none():
    run = load_run_config(ROOT / "configs" / "eval.toml")
    scenario = build_scenario(CATALOG["zero_m3"])
    python_run = replace(run, runtime=RuntimeConfig(backend="python"))
    rust_run = replace(run, runtime=RuntimeConfig(backend="rust"))
    frame_py, _ = evaluate_scenarios([scenario], python_run, "fixed")
    frame_rs, _ = evaluate_scenarios([scenario], rust_run, "fixed")
    _assert_frames(frame_py, frame_rs)
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


# --------------------------------------------------------------------------- #
# R3.3 backend selection, provenance and checkpoint compatibility
# --------------------------------------------------------------------------- #


def test_backend_selection_defaults_to_python_and_requires_extension(monkeypatch):
    from bus_rl.env import factory

    run = load_run_config(ROOT / "configs" / "eval.toml")
    assert run.runtime.backend == "python"
    with pytest.raises(ValueError):
        RuntimeConfig(backend="nope")
    rust_run = replace(run, runtime=RuntimeConfig(backend="rust"))
    monkeypatch.setattr("bus_rl.backend.native.native_available", lambda: False)
    with pytest.raises(RuntimeError):
        factory.make_env_for_run([generate_scenario(1)], rust_run)


def test_metadata_records_backend_and_native_build():
    run = load_run_config(ROOT / "configs" / "eval.toml")
    python_meta = run_metadata(run)
    assert python_meta["backend"] == "python"
    assert python_meta["native_build"] is None
    rust_meta = run_metadata(replace(run, runtime=RuntimeConfig(backend="rust")))
    assert rust_meta["backend"] == "rust"
    assert rust_meta["native_build"]["library_sha256"]


def test_cross_backend_checkpoint_rules():
    assert_backend_compatible({"backend": "python", "backend_parity_verified": True}, "rust")
    assert_backend_compatible({}, "rust")
    assert_backend_compatible({"backend": "rust"}, "rust")
    with pytest.raises(ValueError):
        assert_backend_compatible({"backend": "python", "backend_parity_verified": False}, "rust")


def test_cli_backend_entrypoints(tmp_path):
    data = tmp_path / "data"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bus_rl.cli",
            "generate",
            "--config",
            str(ROOT / "configs/base.toml"),
            "--output",
            str(data),
            "--counts",
            "train=1,validation=1",
        ],
        check=True,
        cwd=ROOT,
    )
    train_out = tmp_path / "train-rust"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bus_rl.cli",
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
        ],
        check=True,
        cwd=ROOT,
    )
    metadata = json.loads((train_out / "metadata.json").read_text())
    assert metadata["backend"] == "rust"
    assert metadata["native_build"]["library_sha256"]
    assert (train_out / "best.zip").exists()

    outputs = {}
    for backend in ("python", "rust"):
        out = tmp_path / f"eval-{backend}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "bus_rl.cli",
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
            ],
            check=True,
            cwd=ROOT,
        )
        outputs[backend] = pd.read_csv(out / "results.csv")
    np.testing.assert_allclose(
        outputs["python"]["total_cost"], outputs["rust"]["total_cost"], rtol=1e-9, atol=1e-9
    )
