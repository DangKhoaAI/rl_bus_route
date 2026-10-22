from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from bus_rl.config import ActionExecutionConfig, load_run_config
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.execution.environments.action_execution import execute_requests
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.environments.rust.batch import NativeBatchVecEnv
from bus_rl.execution.scenarios.generation import generate_scenario
from bus_rl.learning.training.train import make_model

pytestmark = pytest.mark.native


def free_run(name="ppo"):
    run = load_run_config(Path("configs/studies/gru10.toml"))
    return replace(
        run,
        algorithm=replace(
            run.algorithm,
            name=name,
            mask_mode="free",
            n_envs=1,
            n_steps=120,
            batch_size=120,
            n_epochs=1,
            total_timesteps=120,
        ),
        action_execution=ActionExecutionConfig(invalid_action="noop"),
        runtime=replace(run.runtime, eval_batch_size=1),
    )


def test_invalid_requests_are_identical_to_noop_including_final_settlement():
    run = free_run()
    scenario = generate_scenario(81001, run.physical)
    adapted = make_env_for_run([scenario], run)
    strict = make_env_for_run(
        [scenario],
        replace(
            run,
            algorithm=replace(run.algorithm, mask_mode="apply"),
            action_execution=ActionExecutionConfig(),
        ),
    )
    adapted.reset(seed=0)
    strict.reset(seed=0)
    for index in range(120):
        mask = adapted.action_masks()
        invalid = int(np.flatnonzero(~mask)[0])
        observation, reward, done, truncated, info = adapted.step(invalid)
        expected, expected_reward, expected_done, expected_truncated, expected_info = strict.step(0)
        assert (reward, done, truncated) == (expected_reward, expected_done, expected_truncated)
        assert info["costs"] == expected_info["costs"]
        for key in expected:
            np.testing.assert_array_equal(observation[key], expected[key])
        assert info["action_execution"]["executed_action_index"] == 0
        assert not info["action_execution"]["requested_action_valid"]
        assert done == (index == 119)
    adapted.close()
    strict.close()


@pytest.mark.parametrize("name", ["reinforce", "a2c", "ppo"])
def test_free_collects_requested_actions_and_evaluates_with_scalar_batch_parity(name):
    run = free_run(name)
    scenario = generate_scenario(81001, run.physical)
    env = NativeBatchVecEnv([scenario], run, seed=11)
    model = make_model(env, 11, run.algorithm)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.fill_(-100)
        model.policy.action_net.bias[1] = 100  # DISPATCH for a non-depot bus: always invalid here.
    model.learn(total_timesteps=120)
    assert np.all(model.rollout_buffer.actions == 1)
    assert np.all(model.rollout_buffer.action_masks)
    native, traces = evaluate_scenarios(
        [scenario], run, f"{name}_free", model=model, trace_all=True
    )
    scalar_run = replace(run, runtime=replace(run.runtime, native_batch=False))
    scalar, scalar_traces = evaluate_scenarios(
        [scenario], scalar_run, f"{name}_free", model=model, trace_all=True
    )
    assert native.iloc[0].invalid_request_rate == 1
    assert native.iloc[0].executed_noop_rate == 1
    assert native.iloc[0].decision_steps == 120
    assert native.iloc[0].total_cost_core == scalar.iloc[0].total_cost_core
    assert all(row["requested_action_index"] == 1 and row["action_index"] == 0 for row in traces[0])
    assert [row["action_index"] for row in scalar_traces[0]] == [0] * 120
    env.close()


def test_execution_rejects_bad_input_and_does_not_mutate_requested_actions():
    masks = np.zeros((2, 221), bool)
    masks[:, 0] = True
    requests = np.array([1, 2])
    actual, infos = execute_requests(requests, masks, invalid_action="noop", mask_mode="free")
    np.testing.assert_array_equal(requests, [1, 2])
    np.testing.assert_array_equal(actual, [0, 0])
    assert not any(info["requested_action_valid"] for info in infos)
    for bad in [[0, 221], [0, -1], [0.0, 1.0], [[0, 1]]]:
        with pytest.raises(ValueError):
            execute_requests(bad, masks, invalid_action="noop", mask_mode="free")
    with pytest.raises(ValueError, match="infeasible"):
        execute_requests(requests, masks, invalid_action="noop", mask_mode="apply")
