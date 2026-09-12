"""O2 native batch step: BatchKernel, SB3 VecEnv adapter and evaluator path.

Covers the spec acceptance list: scalar-vs-batch replay, interleaved envs,
retained outputs, invalid batch, partial reset, VecEnv auto-reset and full
2048-transition training parity.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl.backend.native import shared_store
from bus_rl.config import (
    ControlConfig,
    ForecastConfig,
    RuntimeConfig,
    load_run_config,
)
from bus_rl.data.scenario import generate_manifest, generate_scenario
from bus_rl.domain import SimConfig
from bus_rl.env.factory import make_env_for_run
from bus_rl.env.native_batch import (
    NativeBatchEvalPool,
    NativeBatchKernel,
    NativeBatchVecEnv,
)
from bus_rl.evaluation.pool import make_eval_pool
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.rewards.costs import RewardConfig
from bus_rl.training.checkpoint import load_metadata, load_model
from bus_rl.training.train import make_env, make_model

pytest.importorskip("bus_sim")

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).parent / "reference" / "diagnose-after" / "last.zip"
OBS_KEYS = (
    "stops",
    "arrival_history",
    "forecast",
    "vehicles",
    "routes",
    "stop_valid",
    "vehicle_valid",
    "route_valid",
    "context",
)


def _reference_run():
    run = load_run_config(ROOT / "configs" / "experiments" / "core-threads2.toml", ROOT)
    metadata = load_metadata(REFERENCE)
    reward = RewardConfig(
        **{
            key: metadata["reward"][key]
            for key in RewardConfig.__dataclass_fields__
            if key in metadata["reward"]
        }
    )
    return replace(
        run,
        control=ControlConfig(**metadata["control"]),
        reward=reward,
        runtime=replace(run.runtime, backend="rust"),
    )


def _assert_obs(left, right, label: str) -> None:
    for key in OBS_KEYS:
        np.testing.assert_array_equal(np.asarray(left[key]), np.asarray(right[key]), err_msg=f"{label}:{key}")


def _assert_single_result(batched_row: dict, scalar: dict, label: str) -> None:
    _assert_obs(
        {key: value[0] for key, value in batched_row["obs"].items()},
        scalar["obs"],
        label,
    )
    np.testing.assert_array_equal(batched_row["mask"][0], scalar["mask"], err_msg=label)
    assert float(batched_row["reward"][0]) == float(scalar["reward"]), label
    assert bool(batched_row["terminated"][0]) == bool(scalar["terminated"]), label
    np.testing.assert_array_equal(batched_row["costs"][0], scalar["costs"], err_msg=label)


def test_native_batch_is_opt_in_and_config_validated():
    run = load_run_config(ROOT / "configs" / "experiments" / "core-threads2.toml", ROOT)
    assert run.runtime.native_batch is False
    with pytest.raises(ValueError, match="native_batch"):
        RuntimeConfig(backend="python", native_batch=True)
    rust = RuntimeConfig(backend="rust", native_batch=True)
    assert rust.native_batch is True


def test_batch_kernel_scalar_replay_is_exact():

    scenarios = generate_manifest("validation", 100)[:4]
    store = shared_store(scenarios)
    kernels = [store.kernel(index) for index in range(len(scenarios))]
    batch = NativeBatchKernel(scenarios, len(scenarios), store=store)
    slots = list(range(len(scenarios)))

    obs_raw, mask = batch.reset(slots, slots)
    resets = [kernel.reset_contract() for kernel in kernels]
    for position, reset in enumerate(resets):
        _assert_obs(
            {key: value[position] for key, value in obs_raw.items()},
            reset["obs"],
            f"reset:{position}",
        )
        np.testing.assert_array_equal(mask[position], reset["mask"])

    for step in range(60):
        actions = [int(np.flatnonzero(mask[position])[0]) for position in slots]
        result = batch.step(slots, actions)
        for position, kernel in enumerate(kernels):
            row = {
                "obs": {
                    key: value[position : position + 1]
                    for key, value in result["obs"].items()
                },
                "mask": result["mask"][position : position + 1],
                "reward": result["reward"][position : position + 1],
                "terminated": result["terminated"][position : position + 1],
                "costs": result["costs"][position : position + 1],
            }
            _assert_single_result(row, kernel.step_contract(actions[position]), f"step:{step}:{position}")
        mask = np.asarray(result["mask"])


def test_mixed_horizons_only_step_active_slots():
    short = generate_scenario(7, replace(SimConfig(), horizon_s=480, demand_end_s=360))
    long = generate_scenario(8)
    batch = NativeBatchKernel([short, long], 2)
    batch.reset([0, 1], [0, 1])
    short_steps = short.config.horizon_s // short.config.control_interval_s
    long_steps = long.config.horizon_s // long.config.control_interval_s
    assert short_steps < long_steps
    steps = {0: 0, 1: 0}
    active = [0, 1]
    while active:
        result = batch.step(active, [0] * len(active))
        done = np.asarray(result["terminated"])
        for position, slot in enumerate(active):
            steps[slot] += 1
            if done[position]:
                assert steps[slot] == (short_steps if slot == 0 else long_steps)
        just_done = {active[i] for i in range(len(active)) if done[i]}
        active = [slot for slot in active if slot not in just_done]
        if 0 in just_done:
            # the short slot is gone while the long slot keeps stepping
            assert active == [1]
            assert np.asarray(result["mask"]).shape[1] == 221
    assert steps == {0: short_steps, 1: long_steps}


def test_retained_outputs_are_not_mutated_by_later_steps_or_resets():
    scenarios = generate_manifest("validation", 100)[:2]
    batch = NativeBatchKernel(scenarios, 2)
    _obs, _mask = batch.reset([0, 1], [0, 1])
    actions = [0, 0]
    first = batch.step([0, 1], actions)
    snapshot = {key: np.array(value, copy=True) for key, value in first["obs"].items()}
    first_mask = np.array(first["mask"], copy=True)
    first_costs = np.array(first["costs"], copy=True)
    for _ in range(5):
        actions = [0, 0]
        batch.step([0, 1], actions)
    for key in OBS_KEYS:
        np.testing.assert_array_equal(first["obs"][key], snapshot[key], err_msg=key)
    np.testing.assert_array_equal(first["mask"], first_mask)
    np.testing.assert_array_equal(first["costs"], first_costs)
    batch.reset([0], [0])
    for key in OBS_KEYS:
        np.testing.assert_array_equal(first["obs"][key], snapshot[key], err_msg=key)


def test_invalid_batch_is_rejected_before_mutation():
    scenarios = generate_manifest("validation", 100)[:2]
    batch = NativeBatchKernel(scenarios, 2)
    _obs, mask = batch.reset([0, 1], [0, 1])
    time_before = batch.current_times([0, 1])

    masked_out = int(np.flatnonzero(~np.asarray(mask)[0])[0])
    with pytest.raises(ValueError):
        batch.step([0, 1], [masked_out, 0])
    with pytest.raises(ValueError):
        batch.step([0, 1], [999, 0])
    with pytest.raises(ValueError):
        batch.step([0, 1], [0])
    with pytest.raises(ValueError):
        batch.step([0, 0], [0, 0])
    np.testing.assert_array_equal(time_before, batch.current_times([0, 1]))

    fresh = NativeBatchKernel(scenarios, 2)
    with pytest.raises(ValueError, match="reset"):
        fresh.step([1], [0])
    # A valid slot in a batch containing an unreset slot must not advance either.
    with pytest.raises(ValueError, match="reset"):
        fresh.step([0, 1], [0, 0])


def test_partial_reset_keeps_other_slots_untouched():
    scenarios = generate_manifest("validation", 100)[:2]
    batch = NativeBatchKernel(scenarios, 2)
    batch.reset([0, 1], [0, 1])
    batch.step([0, 1], [0, 0])
    batch.step([0, 1], [0, 0])
    before = batch.current_times([0, 1])
    assert before[0] == before[1] > 0
    batch.reset([0], [1])
    after = batch.current_times([0, 1])
    assert after[0] == 0
    assert after[1] == before[1]
    snapshot = np.array(batch.step([1], [0])["costs"][0], copy=True)
    assert snapshot.shape == (10,)


def test_vecenv_matches_dummyvecenv_for_masked_actions_and_auto_reset():
    run = _reference_run()
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
    run = _reference_run()
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
    run = _reference_run()
    scenarios = generate_manifest("train", 16)
    algorithm = run.algorithm

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


def test_native_eval_matches_scalar_including_traces():
    run = _reference_run()
    scenarios = generate_manifest("validation", 100)[:10]
    load_env = make_env_for_run(scenarios[:1], run)
    model, _ = load_model(REFERENCE, load_env, run.physical)
    scalar, scalar_traces = evaluate_scenarios(
        scenarios, run, "ppo", model=model, model_seed=run.algorithm.seed, trace_all=True
    )
    native_run = replace(
        run, runtime=replace(run.runtime, native_batch=True, eval_batch_size=8)
    )
    native, native_traces = evaluate_scenarios(
        scenarios, native_run, "ppo", model=model, model_seed=run.algorithm.seed, trace_all=True
    )
    assert list(scalar["scenario_index"]) == list(native["scenario_index"])
    assert list(scalar["scenario_hash"]) == list(native["scenario_hash"])
    for column in scalar.columns:
        if column == "wall_s":
            continue
        for index in range(len(scalar)):
            left, right = scalar[column].iloc[index], native[column].iloc[index]
            if isinstance(left, float) or isinstance(right, float):
                if np.isnan(left) and np.isnan(right):
                    continue
                np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, err_msg=column)
            else:
                assert left == right, (column, index, left, right)
    assert sorted(scalar_traces) == sorted(native_traces)
    for index in scalar_traces:
        left_rows, right_rows = scalar_traces[index], native_traces[index]
        assert len(left_rows) == len(right_rows)
        for left, right in zip(left_rows, right_rows, strict=True):
            assert left["action_index"] == right["action_index"]
    closer = getattr(load_env, "close", None)
    if closer is not None:
        closer()


def test_native_eval_pool_reuse_and_invalidations():
    run = _reference_run()
    scenarios = generate_manifest("validation", 100)[:3]
    native_run = replace(
        run,
        runtime=replace(run.runtime, native_batch=True, eval_batch_size=4, reuse_eval_pool=True),
    )
    pool = make_eval_pool(scenarios, native_run)
    assert isinstance(pool, NativeBatchEvalPool)
    assert pool.batch_size == 3
    try:
        assert pool.matches(scenarios, native_run, None)
        assert not pool.matches(list(reversed(scenarios)), native_run, None)
        flagged = replace(native_run, control=ControlConfig(enable_reassign=False, enable_short_turn=True))
        assert not pool.matches(scenarios, flagged, None)
        assert not pool.matches(
            scenarios,
            replace(native_run, runtime=replace(native_run.runtime, native_batch=False)),
            None,
        )
        assert not pool.matches(
            scenarios,
            replace(native_run, forecast=ForecastConfig(enabled=True)),
            None,
        )
    finally:
        pool.close()
    assert pool.closed


def test_native_batch_with_forecast_matches_scalar():
    run = _reference_run()
    from bus_rl.forecasting.historical import HistoricalForecaster

    scenarios = [generate_scenario(2001), generate_scenario(2002), generate_scenario(2003)]
    load_env = make_env_for_run(scenarios[:1], run)
    model, _ = load_model(REFERENCE, load_env, run.physical)
    forecaster = HistoricalForecaster(tick_s=scenarios[0].config.tick_s)
    forecaster.fit([scenario.arrival_tape for scenario in scenarios], scenarios[0].config)
    forecast_run = replace(run, forecast=ForecastConfig(enabled=True))
    scalar, _ = evaluate_scenarios(
        scenarios, forecast_run, "ppo", model=model, forecaster=forecaster
    )
    native_run = replace(
        forecast_run,
        runtime=replace(forecast_run.runtime, native_batch=True, eval_batch_size=4),
    )
    native, _ = evaluate_scenarios(
        scenarios, native_run, "ppo", model=model, forecaster=forecaster
    )
    assert list(scalar["total_cost"]) == list(native["total_cost"])
    closer = getattr(load_env, "close", None)
    if closer is not None:
        closer()
