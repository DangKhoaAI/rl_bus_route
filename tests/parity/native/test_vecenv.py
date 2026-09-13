"""O2 native SB3 VecEnv adapter and training-parity tests.

The adapters must match ``DummyVecEnv`` bit-for-bit for masked actions,
auto-reset and a full 2048-transition MaskablePPO update.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl.execution.environments.rust.batch import NativeBatchVecEnv
from bus_rl.execution.runtime import configure_torch_distributions
from bus_rl.execution.scenarios.generation import generate_manifest, generate_scenario
from bus_rl.learning.training.train import make_env, make_model
from bus_sim.oracle.domain import SimConfig
from tests.support.contract import OBS_KEYS
from tests.support.reference import reference_run

pytest.importorskip("bus_sim_native")

pytestmark = pytest.mark.native


def test_vecenv_matches_dummyvecenv_for_masked_actions_and_auto_reset():
    run = reference_run(backend="rust")
    scenarios = generate_manifest("train", 16)
    dummy = DummyVecEnv([make_env(scenarios, run, run.algorithm.seed + i) for i in range(4)])
    native = NativeBatchVecEnv(scenarios, run, run.algorithm.seed)
    try:
        left = dummy.reset()
        right = native.reset()
        for key in OBS_KEYS:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        for step in range(300):
            masks = np.stack(dummy.env_method("action_masks"))
            actions = np.array([int(np.flatnonzero(masks[i])[0]) for i in range(4)])
            left, lr, ld, li = dummy.step(actions)
            right, rr, rd, ri = native.step(actions)
            np.testing.assert_array_equal(lr, rr, err_msg=f"reward:{step}")
            np.testing.assert_array_equal(ld, rd, err_msg=f"done:{step}")
            for key in OBS_KEYS:
                np.testing.assert_array_equal(left[key], right[key], err_msg=f"{key}:{step}")
            for info_left, info_right in zip(li, ri, strict=True):
                assert info_left["TimeLimit.truncated"] == info_right["TimeLimit.truncated"]
                assert ("terminal_observation" in info_left) == (
                    "terminal_observation" in info_right
                )
                if "terminal_observation" in info_left:
                    for key in OBS_KEYS:
                        np.testing.assert_array_equal(
                            info_left["terminal_observation"][key],
                            info_right["terminal_observation"][key],
                            err_msg=f"terminal:{key}:{step}",
                        )
    finally:
        dummy.close()
        native.close()


def test_vecenv_auto_reset_seeds_are_honored():
    run = reference_run(backend="rust")
    short = generate_scenario(7, replace(SimConfig(), horizon_s=480, demand_end_s=360))
    short_run = replace(run, algorithm=replace(run.algorithm, n_envs=1))
    dummy = DummyVecEnv([make_env([short], short_run, short_run.algorithm.seed)])
    native = NativeBatchVecEnv([short], short_run, short_run.algorithm.seed)
    try:
        assert dummy.envs[0].scenario_index == native.kernel.scenario_indices()[0]
        dummy.reset()
        native.reset()
        assert dummy.envs[0].scenario_index == native.kernel.scenario_indices()[0]
        mask = np.asarray(native.kernel.masks([0]))
        for _ in range(4):
            action = int(np.flatnonzero(mask[0])[0])
            dummy.step(np.array([action]))
            native.step(np.array([action]))
            mask = np.asarray(native.kernel.masks([0]))
            assert dummy.envs[0].scenario_index == native.kernel.scenario_indices()[0]
    finally:
        dummy.close()
        native.close()


def test_training_parity_2048_transitions_bit_identical():
    run = reference_run(backend="rust")
    scenarios = generate_manifest("train", 16)
    algorithm = run.algorithm
    configure_torch_distributions(True)

    def _train(native: bool):
        if native:
            env = NativeBatchVecEnv(scenarios, run, algorithm.seed)
        else:
            env = DummyVecEnv(
                [make_env(scenarios, run, algorithm.seed + i) for i in range(algorithm.n_envs)]
            )
        model = make_model(env, algorithm.seed, algorithm)
        model.learn(total_timesteps=2048)
        weights = {
            key: value.detach().cpu().numpy().copy()
            for key, value in model.policy.state_dict().items()
        }
        optimizer = model.policy.optimizer.state_dict()
        env.close()
        return weights, optimizer

    baseline_weights, baseline_optimizer = _train(False)
    native_weights, native_optimizer = _train(True)
    assert set(baseline_weights) == set(native_weights)
    for key in baseline_weights:
        np.testing.assert_array_equal(baseline_weights[key], native_weights[key], err_msg=key)
    assert set(baseline_optimizer["state"]) == set(native_optimizer["state"])
    for parameter in baseline_optimizer["state"]:
        for field, value in baseline_optimizer["state"][parameter].items():
            other = native_optimizer["state"][parameter][field]
            left = value.cpu().numpy() if hasattr(value, "cpu") else value
            right = other.cpu().numpy() if hasattr(other, "cpu") else other
            np.testing.assert_array_equal(left, right, err_msg=f"{parameter}:{field}")


def test_disabling_distribution_validation_is_bit_identical():
    run = reference_run(backend="rust")
    scenarios = generate_manifest("train", 16)
    env = DummyVecEnv([make_env(scenarios, run, run.algorithm.seed + i) for i in range(4)])
    model = make_model(env, run.algorithm.seed, run.algorithm)
    model.policy.set_training_mode(False)
    observation = env.reset()
    masks = np.stack(env.env_method("action_masks"))
    from stable_baselines3.common.utils import obs_as_tensor

    obs_tensor = obs_as_tensor(observation, "cpu")
    import torch

    def sample():
        torch.manual_seed(123)
        with torch.no_grad():
            actions, values, logprob = model.policy(obs_tensor, action_masks=masks)
            argmax = model.policy.get_distribution(obs_tensor, action_masks=masks).get_actions(
                deterministic=True
            )
        return actions, values, logprob, argmax

    try:
        configure_torch_distributions(True)
        actions_on, values_on, logprob_on, argmax_on = sample()
        predict_on, _ = model.predict(observation, action_masks=masks, deterministic=True)
        configure_torch_distributions(False)
        actions_off, values_off, logprob_off, argmax_off = sample()
        predict_off, _ = model.predict(observation, action_masks=masks, deterministic=True)
        for left, right, label in (
            (actions_on, actions_off, "actions"),
            (values_on, values_off, "values"),
            (logprob_on, logprob_off, "log_prob"),
            (argmax_on, argmax_off, "argmax"),
        ):
            assert torch.equal(left, right), label
        np.testing.assert_array_equal(np.asarray(predict_on), np.asarray(predict_off))
    finally:
        configure_torch_distributions(True)
        env.close()


def test_validate_distributions_off_matches_on_training_2048():
    run = reference_run(backend="rust")
    scenarios = generate_manifest("train", 16)
    algorithm = run.algorithm

    def _train(enabled: bool):
        configure_torch_distributions(enabled)
        env = NativeBatchVecEnv(scenarios, run, algorithm.seed)
        model = make_model(env, algorithm.seed, algorithm)
        model.learn(total_timesteps=2048)
        weights = {
            key: value.detach().cpu().numpy().copy()
            for key, value in model.policy.state_dict().items()
        }
        optimizer = model.policy.optimizer.state_dict()
        env.close()
        return weights, optimizer

    try:
        weights_on, optimizer_on = _train(True)
        weights_off, optimizer_off = _train(False)
    finally:
        configure_torch_distributions(True)
    for key in weights_on:
        np.testing.assert_array_equal(weights_on[key], weights_off[key], err_msg=key)
    for parameter in optimizer_on["state"]:
        for field, value in optimizer_on["state"][parameter].items():
            other = optimizer_off["state"][parameter][field]
            left = value.cpu().numpy() if hasattr(value, "cpu") else value
            right = other.cpu().numpy() if hasattr(other, "cpu") else other
            np.testing.assert_array_equal(left, right, err_msg=f"{parameter}:{field}")


def test_metadata_records_effective_distribution_validation():
    from bus_rl.learning.training.checkpoint import run_metadata

    base = reference_run(backend="rust")
    run = replace(base, runtime=replace(base.runtime, validate_distributions=False))
    configure_torch_distributions(False)
    try:
        metadata = run_metadata(run)
        assert metadata["validate_distributions"] is False
        assert metadata["torch_distribution_validate_args"] is False
    finally:
        configure_torch_distributions(True)
    metadata = run_metadata(base)
    assert metadata["validate_distributions"] is True
    assert metadata["torch_distribution_validate_args"] is True
