"""R4.1 correctness/smoke tests for the native backend.

The heavy interleaved speed benchmark lives in
``scripts/benchmark_backends.py``; this file covers the correctness gate: a
2048-transition MaskablePPO smoke with finite obs/reward/loss, masked-action
safety, checkpoint save/load, and fixed-checkpoint per-day parity.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl.config import ControlConfig, RuntimeConfig, load_run_config
from bus_rl.evaluation.runner import make_controller, rollout
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.environments.rust.environment import NativeBusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_manifest, generate_scenario
from bus_rl.learning.training.checkpoint import (
    load_metadata,
    load_model,
    run_metadata,
    write_metadata,
)
from bus_rl.learning.training.train import fit_algorithm, make_env, make_model
from bus_sim.oracle.costs import RewardConfig

pytest.importorskip("bus_sim_native")

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).parent / "reference" / "diagnose-after" / "last.zip"


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_train_smoke_2048_is_finite_and_roundtrips(tmp_path, backend):
    run = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    algorithm = fit_algorithm(run.algorithm, timesteps=2048, n_envs=2)
    run = replace(run, algorithm=algorithm, runtime=RuntimeConfig(backend=backend))

    scenarios = [generate_scenario(1001), generate_scenario(1002)]
    env = DummyVecEnv([make_env(scenarios, run, algorithm.seed + index) for index in range(2)])
    model = make_model(env, algorithm.seed, run.algorithm)
    # MaskablePPO only collects masked-valid actions; the native env raises on
    # any masked/out-of-range action, so a completed learn is a masked-action
    # safety check as well.
    model.learn(total_timesteps=2048)
    assert model.num_timesteps >= 2048
    losses = {
        key: float(value) for key, value in model.logger.name_to_value.items() if "loss" in key
    }
    assert losses, "no loss values were logged"
    assert all(np.isfinite(value) for value in losses.values()), losses

    checkpoint = tmp_path / "smoke"
    model.save(str(checkpoint))
    assert (tmp_path / "smoke.zip").exists()
    write_metadata(tmp_path, run_metadata(run))

    eval_env = NativeBusDispatchEnv(scenarios, run.physical, reward=run.reward, control=run.control)
    loaded, metadata = load_model(tmp_path / "smoke.zip", eval_env, run.physical, backend=backend)
    observation, _ = eval_env.reset(seed=0, options={"scenario_index": 0})
    for _ in range(4):
        mask = eval_env.action_masks()
        action, _ = loaded.predict(observation, action_masks=mask, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        assert mask[action]
        observation, reward, _, _, _ = eval_env.step(action)
        assert np.isfinite(reward)
    assert metadata["backend"] == backend
    env.close()
    eval_env.close()


def test_smoke_metadata_records_backend_and_native_build():
    run = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    metadata = run_metadata(replace(run, runtime=RuntimeConfig(backend="rust")))
    assert metadata["backend"] == "rust"
    assert metadata["native_build"]["library_sha256"]
    assert metadata["backend_parity_verified"] is True
    assert metadata["torch_threads"] == run.algorithm.torch_threads
    assert metadata["validate_observation"] is True


def _reference_run():
    run = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    metadata = load_metadata(REFERENCE)
    reward = RewardConfig(
        **{
            key: metadata["reward"][key]
            for key in RewardConfig.__dataclass_fields__
            if key in metadata["reward"]
        }
    )
    return replace(run, control=ControlConfig(**metadata["control"]), reward=reward)


def test_fixed_checkpoint_per_day_parity():
    run = _reference_run()
    scenarios = generate_manifest("validation", 10)
    load_env = make_env_for_run(scenarios[:1], run)
    model, metadata = load_model(REFERENCE, load_env, run.physical)
    controller = make_controller("ppo", model=model, seed=metadata.get("seed", 11))

    costs: dict[str, list[float]] = {}
    traces: dict[str, list[list[dict]]] = {}
    for backend in ("python", "rust"):
        env = make_env_for_run(scenarios, replace(run, runtime=RuntimeConfig(backend=backend)))
        backend_costs = []
        backend_traces = []
        for index in range(len(scenarios)):
            metrics, rows = rollout(env, controller, index, trace=True)
            backend_costs.append(metrics["total_cost"])
            backend_traces.append(rows)
        costs[backend] = backend_costs
        traces[backend] = backend_traces
    np.testing.assert_allclose(costs["python"], costs["rust"], rtol=1e-9, atol=1e-9)
    for py_day, rs_day in zip(traces["python"], traces["rust"], strict=True):
        assert len(py_day) == len(rs_day)
        for left, right in zip(py_day, rs_day, strict=True):
            assert left["action"] == right["action"]
            assert left["time_s"] == right["time_s"]
            assert list(left["queues"]) == list(right["queues"])
            assert left["buses"] == right["buses"]
            for key in ("waiting", "onboard", "generated", "abandoned", "completed"):
                assert left[key] == right[key]
