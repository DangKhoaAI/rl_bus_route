from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bus_rl.config import ActionExecutionConfig, load_run_config
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.training.checkpoint import load_model
from bus_rl.learning.training.train import train_run

pytestmark = pytest.mark.native


@pytest.mark.parametrize("name", ["reinforce", "a2c"])
@pytest.mark.parametrize("mode", ["apply", "free"])
def test_algorithm_train_reload_final_update_and_batch_parity(tmp_path, name, mode):
    pytest.importorskip("bus_sim_native")
    run = load_run_config(Path("configs/studies/gru10.toml"))
    run = replace(
        run,
        algorithm=replace(
            run.algorithm,
            name=name,
            mask_mode=mode,
            n_envs=1,
            n_steps=120,
            batch_size=120,
            n_epochs=4,
            total_timesteps=240,
            eval_freq=120,
            validation_timing="post_update",
        ),
        runtime=replace(run.runtime, eval_batch_size=2),
        action_execution=ActionExecutionConfig(invalid_action="noop"),
    )
    scenarios = [generate_scenario(seed, run.physical) for seed in [81001, 81002]]
    out = tmp_path / name
    meta = train_run(run, scenarios, scenarios, out)
    assert meta["algorithm"]["name"] == name
    assert meta["optimizer_steps"] == 2
    assert meta["effective_n_epochs"] == 1
    model, _ = load_model(out / "best.zip", None, run.physical, backend="rust")
    method = name if mode == "apply" else f"{name}_free"
    assert model.mask_mode == mode
    batched, traces = evaluate_scenarios(scenarios, run, method, model=model, trace_all=True)
    scalar_run = replace(run, runtime=replace(run.runtime, native_batch=False, eval_batch_size=1))
    scalar, scalar_traces = evaluate_scenarios(
        scenarios, scalar_run, method, model=model, trace_all=True
    )
    np.testing.assert_array_equal(batched.total_cost_core, scalar.total_cost_core)
    assert set(batched.method) == {method}
    for index in range(2):
        assert [row["action_index"] for row in traces[index]] == [
            row["action_index"] for row in scalar_traces[index]
        ]
    history = pd.read_csv(out / "evaluations.csv")
    assert history.optimizer_steps.tolist() == [0, 1, 2]
    assert history.timesteps.tolist() == [0, 120, 240]
    assert batched.total_cost_core.mean() == pytest.approx(meta["best_val_cost"])
    if name == "reinforce":
        assert not model.critic_used
        assert all(not p.requires_grad for p in model.policy.value_net.parameters())
        assert all(p.grad is None for p in model.policy.mlp_extractor.value_net.parameters())


def test_reinforce_rejects_partial_budget_before_output(tmp_path):
    run = load_run_config(Path("configs/studies/gru10.toml"))
    run = replace(run, algorithm=replace(run.algorithm, name="reinforce", total_timesteps=32))
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="whole episode"):
        train_run(run, [], [], out)
    assert not out.exists()
