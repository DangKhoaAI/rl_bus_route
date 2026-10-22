from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bus_rl.config import load_run_config
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.training.checkpoint import load_model
from bus_rl.learning.training.train import train_run

pytestmark = pytest.mark.native


@pytest.mark.parametrize(
    "encoder,minutes,decay",
    [
        ("mlp", 10, 1.0),
        ("belief", 10, 1.0),
        ("gru", 10, 1.0),
        ("belief_gru", 10, 1.0),
        ("gru", 20, 1.0),
        ("gru", 30, 1.0),
        ("belief", 20, 0.8),
        ("belief_gru", 20, 0.8),
    ],
)
def test_encoders_train_reload_and_match_scalar_native_batch(tmp_path, encoder, minutes, decay):
    pytest.importorskip("bus_sim_native")
    run = load_run_config(Path("configs/experiments/rl-improvement/core.toml"))
    run = replace(
        run,
        algorithm=replace(
            run.algorithm,
            encoder=encoder,
            history_minutes=minutes,
            belief_decay=decay,
            total_timesteps=256,
            n_steps=16,
            n_envs=2,
            batch_size=32,
            n_epochs=1,
            eval_freq=128,
            diagnostics_enabled=False,
        ),
        runtime=replace(run.runtime, eval_batch_size=2),
    )
    scenarios = [generate_scenario(seed, run.physical) for seed in [81001, 81002, 81003]]
    output = tmp_path / encoder
    metadata = train_run(run, scenarios, scenarios, output)
    assert metadata["total_timesteps_actual"] == 256  # includes native per-slot auto-reset
    assert metadata["algorithm"]["encoder"] == encoder
    assert metadata["backend_parity_verified"] == (minutes == 10)
    assert abs(metadata["policy_parameters"] / 448862 - 1) < 0.002
    model, _ = load_model(output / "best.zip", None, run.physical, backend="rust")
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    batched, batch_traces = evaluate_scenarios(scenarios, run, "ppo", model=model, trace_all=True)
    scalar_run = replace(
        run,
        runtime=replace(run.runtime, native_batch=False, eval_batch_size=1, reuse_eval_pool=False),
    )
    scalar, scalar_traces = evaluate_scenarios(
        scenarios, scalar_run, "ppo", model=model, trace_all=True
    )
    np.testing.assert_allclose(
        batched["total_cost_core"], scalar["total_cost_core"], rtol=0, atol=0
    )
    for index in range(3):
        assert [row["action_index"] for row in batch_traces[index]] == [
            row["action_index"] for row in scalar_traces[index]
        ]
    # Loading the earlier best checkpoint must reproduce its callback evaluation.
    history = pd.read_csv(output / "evaluations.csv")
    assert batched["total_cost_core"].mean() == pytest.approx(history["val_cost"].min())


def test_long_history_matches_native_five_bins_and_resets_after_terminal():
    from bus_rl.execution.environments.factory import make_env_for_run
    from bus_rl.execution.environments.rust.batch import NativeBatchVecEnv

    pytest.importorskip("bus_sim_native")
    run = load_run_config(Path("configs/experiments/belief-gru.toml"))
    run = replace(
        run, algorithm=replace(run.algorithm, encoder="gru", history_minutes=30, n_envs=1)
    )
    scenario = generate_scenario(81001, run.physical)
    scalar = make_env_for_run([scenario], run)
    vector = NativeBatchVecEnv([scenario], run, 11)
    scalar.reset(seed=0)
    vector.reset()
    arrivals = []
    for step in range(120):
        obs, _, done, _, _ = scalar.step(0)
        arrivals.insert(0, obs["arrival_history"][..., 0].copy())
        expected = np.zeros((4, 2, 8, 15), np.float32)
        for lag, counts in enumerate(arrivals[:15]):
            expected[..., lag] = counts
        np.testing.assert_array_equal(obs["arrival_memory"], expected)
        np.testing.assert_array_equal(obs["arrival_memory"][..., :5], obs["arrival_history"])
        vector.step_async(np.array([0]))
        batch, _, dones, infos = vector.step_wait()
        if done:
            assert dones[0]
            np.testing.assert_array_equal(
                infos[0]["terminal_observation"]["arrival_memory"], expected
            )
            assert not batch["arrival_memory"].any()
        else:
            np.testing.assert_array_equal(batch["arrival_memory"][0], expected)
    obs, _ = scalar.reset(seed=0)
    assert not obs["arrival_memory"].any()
    scalar.close()
    vector.close()
