"""O2 native batch step: BatchKernel and the evaluator/pool path.

Covers the spec acceptance list: scalar-vs-batch replay, interleaved slots,
retained outputs, invalid batch, partial reset and batch evaluation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bus_rl.config import ControlConfig, ForecastConfig, RuntimeConfig, load_run_config
from bus_rl.evaluation.pool import make_eval_pool
from bus_rl.evaluation.runner import evaluate_scenarios
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.environments.rust.batch import NativeBatchEvalPool, NativeBatchKernel
from bus_rl.execution.environments.rust.bridge import shared_store
from bus_rl.execution.scenarios.generation import generate_manifest, generate_scenario
from bus_rl.learning.training.checkpoint import load_model
from bus_sim.oracle.domain import SimConfig
from tests.support.compare import assert_obs_exact, assert_single_result
from tests.support.contract import OBS_KEYS
from tests.support.paths import REFERENCE, ROOT
from tests.support.reference import reference_run, requires_reference

pytest.importorskip("bus_sim_native")

pytestmark = pytest.mark.native


def test_native_batch_is_opt_in_and_config_validated():
    run = load_run_config(ROOT / "configs" / "experiments" / "core-threads2.toml", ROOT)
    assert run.runtime.native_batch is False
    with pytest.raises(ValueError, match="native_batch"):
        RuntimeConfig(backend="python", native_batch=True)
    rust = RuntimeConfig(backend="rust", native_batch=True)
    assert rust.native_batch is True


def test_batch_kernel_scalar_replay_is_exact():

    scenarios = generate_manifest("validation", 4)
    store = shared_store(scenarios)
    kernels = [store.kernel(index) for index in range(len(scenarios))]
    batch = NativeBatchKernel(scenarios, len(scenarios), store=store)
    slots = list(range(len(scenarios)))

    obs_raw, mask = batch.reset(slots, slots)
    resets = [kernel.reset_contract() for kernel in kernels]
    for position, reset in enumerate(resets):
        assert_obs_exact(
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
                    key: value[position : position + 1] for key, value in result["obs"].items()
                },
                "mask": result["mask"][position : position + 1],
                "reward": result["reward"][position : position + 1],
                "terminated": result["terminated"][position : position + 1],
                "costs": result["costs"][position : position + 1],
            }
            assert_single_result(
                row, kernel.step_contract(actions[position]), f"step:{step}:{position}"
            )
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
    scenarios = generate_manifest("validation", 2)
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
    scenarios = generate_manifest("validation", 2)
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
    scenarios = generate_manifest("validation", 2)
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


@requires_reference
def test_native_eval_matches_scalar_including_traces():
    run = reference_run(backend="rust")
    scenarios = generate_manifest("validation", 10)
    load_env = make_env_for_run(scenarios[:1], run)
    model, _ = load_model(REFERENCE, load_env, run.physical)
    scalar, scalar_traces = evaluate_scenarios(
        scenarios, run, "ppo", model=model, model_seed=run.algorithm.seed, trace_all=True
    )
    native_run = replace(run, runtime=replace(run.runtime, native_batch=True, eval_batch_size=8))
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


@requires_reference
def test_native_eval_pool_reuse_and_invalidations():
    run = reference_run(backend="rust")
    scenarios = generate_manifest("validation", 3)
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
        flagged = replace(
            native_run, control=ControlConfig(enable_reassign=False, enable_short_turn=True)
        )
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


@requires_reference
def test_native_batch_with_forecast_matches_scalar():
    run = reference_run(backend="rust")
    from bus_rl.learning.forecasting import HistoricalForecaster

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
    native, _ = evaluate_scenarios(scenarios, native_run, "ppo", model=model, forecaster=forecaster)
    assert list(scalar["total_cost"]) == list(native["total_cost"])
    closer = getattr(load_env, "close", None)
    if closer is not None:
        closer()
