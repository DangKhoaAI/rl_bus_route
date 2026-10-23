import json
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


@pytest.mark.parametrize("name", ["reinforce", "a2c", "ppo"])
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
            diagnostics_enabled=True,
            diagnostics_interval=120,
        ),
        runtime=replace(run.runtime, eval_batch_size=2),
        action_execution=ActionExecutionConfig(invalid_action="noop"),
    )
    scenarios = [generate_scenario(seed, run.physical) for seed in [81001, 81002]]
    out = tmp_path / name
    meta = train_run(run, scenarios, scenarios, out)
    assert meta["algorithm"]["name"] == name
    updates_per_rollout = 4 if name == "ppo" else 1
    assert meta["optimizer_steps"] == 2 * updates_per_rollout
    assert meta["effective_n_epochs"] == (4 if name == "ppo" else 1)
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
    assert history.optimizer_steps.tolist() == [0, updates_per_rollout, 2 * updates_per_rollout]
    assert history.timesteps.tolist() == [0, 120, 240]
    assert batched.total_cost_core.mean() == pytest.approx(meta["best_val_cost"])
    if name == "reinforce":
        assert not model.critic_used
        assert all(not p.requires_grad for p in model.policy.value_net.parameters())
        assert all(p.grad is None for p in model.policy.mlp_extractor.value_net.parameters())
    diagnostics = [
        json.loads(line) for line in (out / "training_diagnostics.jsonl").read_text().splitlines()
    ]
    policy_rows = [row for row in diagnostics if row["kind"] == "policy_diagnostics"]
    assert policy_rows and all(row["available"] for row in policy_rows)
    if mode == "free":
        assert all(row["invalid_probability_mass_mean"] > 0 for row in policy_rows)
    else:
        assert all(row["invalid_probability_mass_mean"] == 0 for row in policy_rows)


def test_reinforce_rejects_partial_budget_before_output(tmp_path):
    run = load_run_config(Path("configs/studies/gru10.toml"))
    run = replace(run, algorithm=replace(run.algorithm, name="reinforce", total_timesteps=32))
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="whole episode"):
        train_run(run, [], [], out)
    assert not out.exists()


@pytest.mark.parametrize("entrypoint", ["train", "diagnose"])
@pytest.mark.parametrize("horizon", [60, 150])
def test_reinforce_rejects_incomplete_horizon_before_output(tmp_path, entrypoint, horizon):
    from bus_rl.learning.training.diagnose import diagnose_train

    run = load_run_config(Path("configs/studies/gru10.toml"))
    run = replace(
        run,
        algorithm=replace(run.algorithm, name="reinforce"),
        physical=replace(run.physical, horizon_s=horizon, demand_end_s=0),
    )
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="complete episodes"):
        (train_run if entrypoint == "train" else diagnose_train)(run, [], [], out)
    assert not out.exists()
