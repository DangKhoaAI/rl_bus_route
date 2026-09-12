"""O1 batched evaluation and pool-reuse tests against the shipped evaluator."""

from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bus_rl.config import ControlConfig, ForecastConfig, RuntimeConfig, load_run_config
from bus_rl.data.scenario import generate_manifest, generate_scenario
from bus_rl.domain import SimConfig
from bus_rl.env.factory import make_env_for_run
from bus_rl.evaluation.pool import EvalEnvPool
from bus_rl.evaluation.runner import evaluate_scenarios, mean_cost, validate_eval_runtime
from bus_rl.forecasting.historical import HistoricalForecaster
from bus_rl.rewards.costs import RewardConfig
from bus_rl.training.callbacks import BestValidationCallback
from bus_rl.training.checkpoint import load_metadata, load_model

pytest.importorskip("bus_sim")

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).parent / "reference" / "diagnose-after" / "last.zip"


def _assert_value(left, right, label: str) -> None:
    if left is None or right is None:
        assert left is None and right is None, (label, left, right)
    elif isinstance(left, float) or isinstance(right, float):
        if np.isnan(left) or np.isnan(right):
            assert np.isnan(left) and np.isnan(right), (label, left, right)
        else:
            np.testing.assert_allclose(float(left), float(right), rtol=1e-9, atol=1e-9, err_msg=label)
    else:
        assert left == right, (label, left, right)


def _assert_frames(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert list(left["scenario_index"]) == list(right["scenario_index"])
    skip = {"wall_s"}
    columns = [column for column in left.columns if column not in skip]
    for column in columns:
        assert column in right.columns, column
        for index, (x, y) in enumerate(zip(left[column], right[column], strict=True)):
            _assert_value(x, y, f"{column}[{index}]")


def _first_action_divergence(left_traces: dict, right_traces: dict) -> dict | None:
    for index in sorted(set(left_traces) | set(right_traces)):
        left_rows = left_traces.get(index, [])
        right_rows = right_traces.get(index, [])
        if len(left_rows) != len(right_rows):
            return {
                "field": "trace_length",
                "scenario_index": index,
                "expected": len(left_rows),
                "actual": len(right_rows),
            }
        for step, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
            if left.get("action_index") != right.get("action_index"):
                return {
                    "field": "action_index",
                    "scenario_index": index,
                    "step": step,
                    "time_s": left.get("time_s"),
                    "expected_action": left.get("action_index"),
                    "actual_action": right.get("action_index"),
                    "expected_kind": left.get("action"),
                    "actual_kind": right.get("action"),
                    "reproducer": (
                        "evaluate_scenarios(..., trace_all=True) on "
                        f"tests/backend_parity/reference/diagnose-after day {index} step {step}"
                    ),
                }
    return None


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


@pytest.fixture(scope="module")
def reference_bundle():
    run = _reference_run()
    scenarios = generate_manifest("validation", 100)[:10]
    load_env = make_env_for_run(scenarios[:1], run)
    model, metadata = load_model(REFERENCE, load_env, run.physical)
    scalar_frame, scalar_traces = evaluate_scenarios(
        scenarios,
        run,
        "ppo",
        model=model,
        model_seed=run.algorithm.seed,
        trace_all=True,
    )
    try:
        yield {
            "run": run,
            "scenarios": scenarios,
            "model": model,
            "metadata": metadata,
            "scalar_frame": scalar_frame,
            "scalar_traces": scalar_traces,
        }
    finally:
        closer = getattr(load_env, "close", None)
        if closer is not None:
            closer()


def _eval(run, scenarios, model, *, pool=None, trace_all=False):
    return evaluate_scenarios(
        scenarios,
        run,
        "ppo",
        model=model,
        model_seed=run.algorithm.seed,
        pool=pool,
        trace_all=trace_all,
    )


def test_runtime_defaults_keep_scalar_and_training_hyperparameters():
    run = load_run_config(ROOT / "configs" / "experiments" / "core-threads2.toml", ROOT)
    assert run.runtime.eval_batch_size == 1
    assert run.runtime.reuse_eval_pool is False
    assert run.runtime.native_batch is False
    assert run.runtime.backend == "python"
    assert run.algorithm.n_envs == 4
    assert run.algorithm.n_steps == 256
    assert run.algorithm.batch_size == 256
    assert run.algorithm.n_epochs == 4
    assert run.algorithm.torch_threads == 2
    assert run.algorithm.eval_freq == 12_288
    with pytest.raises(ValueError, match="eval_batch_size"):
        RuntimeConfig(eval_batch_size=0)
    with pytest.raises(ValueError, match="native_batch"):
        RuntimeConfig(backend="python", native_batch=True)
    import bus_sim

    # O2 is implemented but opt-in: the native kernel exists, defaults stay scalar.
    assert hasattr(bus_sim, "BatchKernel")


def test_batch_one_via_pool_matches_scalar(reference_bundle):
    run = reference_bundle["run"]
    scenarios = reference_bundle["scenarios"][:5]
    model = reference_bundle["model"]
    scalar, scalar_traces = _eval(run, scenarios, model, trace_all=True)
    pooled_run = replace(run, runtime=replace(run.runtime, eval_batch_size=1, reuse_eval_pool=True))
    pool = EvalEnvPool(scenarios, pooled_run)
    try:
        batched, batched_traces = _eval(pooled_run, scenarios, model, pool=pool, trace_all=True)
    finally:
        pool.close()
    _assert_frames(scalar, batched)
    divergence = _first_action_divergence(scalar_traces, batched_traces)
    assert divergence is None, divergence
    assert list(batched["scenario_index"]) == list(range(len(scenarios)))


@pytest.mark.parametrize("batch_size", [4, 8, 16, 32])
def test_enabled_batch_sizes_match_scalar_including_partial_last(reference_bundle, batch_size):
    run = reference_bundle["run"]
    # 10 committed validation days: last batch is partial for 4/8/16/32.
    scenarios = reference_bundle["scenarios"]
    model = reference_bundle["model"]
    scalar, scalar_traces = reference_bundle["scalar_frame"], reference_bundle["scalar_traces"]
    batched_run = replace(run, runtime=replace(run.runtime, eval_batch_size=batch_size))
    batched, batched_traces = _eval(batched_run, scenarios, model, trace_all=True)
    _assert_frames(scalar, batched)
    divergence = _first_action_divergence(scalar_traces, batched_traces)
    assert divergence is None, divergence
    assert list(batched["scenario_hash"]) == [scenario.scenario_hash for scenario in scenarios]
    assert len(scenarios) % batch_size != 0


def test_done_slots_are_not_stepped_when_horizons_differ(reference_bundle):
    run = reference_bundle["run"]
    model = reference_bundle["model"]
    short = generate_scenario(7, replace(SimConfig(), horizon_s=480, demand_end_s=360))
    long = generate_scenario(8)
    scenarios = [short, long]
    batched_run = replace(run, runtime=replace(run.runtime, eval_batch_size=2))
    pool = EvalEnvPool(scenarios, batched_run)
    counts = [0, 0]
    try:
        for index, env in enumerate(pool.envs):
            original = env.step

            def _step(action, *, _index=index, _original=original):
                counts[_index] += 1
                return _original(action)

            env.step = _step  # type: ignore[method-assign]
        frame, _ = _eval(batched_run, scenarios, model, pool=pool)
    finally:
        pool.close()
    assert counts[0] == short.config.horizon_s // short.config.control_interval_s
    assert counts[1] == long.config.horizon_s // long.config.control_interval_s
    assert list(frame["scenario_index"]) == [0, 1]
    assert frame["scenario_hash"].tolist() == [short.scenario_hash, long.scenario_hash]


def test_twenty_eval_rounds_do_not_leak_or_mix_days(reference_bundle):
    run = reference_bundle["run"]
    scenarios = reference_bundle["scenarios"][:3]
    model = reference_bundle["model"]
    batched_run = replace(
        run, runtime=replace(run.runtime, eval_batch_size=4, reuse_eval_pool=True)
    )
    scalar, _ = _eval(run, scenarios, model)
    pool = EvalEnvPool(scenarios, batched_run)
    rss = []
    try:
        def _kernels() -> int:
            gc.collect()
            return sum(1 for obj in gc.get_objects() if type(obj).__name__ == "Kernel")

        def _rss_kb() -> int:
            with open("/proc/self/statm", encoding="utf-8") as handle:
                pages = int(handle.read().split()[1])
            return pages * (os_page_kb())

        def os_page_kb() -> int:
            import os

            return os.sysconf("SC_PAGE_SIZE") // 1024

        before_kernels = _kernels()
        for round_id in range(20):
            frame, _ = _eval(batched_run, scenarios, model, pool=pool)
            _assert_frames(scalar, frame)
            assert list(frame["scenario_index"]) == [0, 1, 2]
            assert list(frame["scenario_hash"]) == [item.scenario_hash for item in scenarios]
            assert len(pool.envs) == 3
            assert not pool.closed
            rss.append(_rss_kb())
            del frame
        after_kernels = _kernels()
        # Slots stay at the remaining-day cap; kernels do not grow with round count.
        assert after_kernels <= before_kernels + 8
        assert rss[-1] <= max(rss[:5]) + 65_536
    finally:
        pool.close()
    assert pool.closed


def test_pool_invalidates_on_order_flags_config_forecast_and_contract(reference_bundle):
    run = reference_bundle["run"]
    scenarios = reference_bundle["scenarios"][:4]
    pool = EvalEnvPool(scenarios, run)
    try:
        assert pool.matches(scenarios, run, None)
        assert not pool.matches(list(reversed(scenarios)), run, None)
        assert not pool.matches(scenarios[:3], run, None)
        flagged = replace(run, control=ControlConfig(enable_reassign=False, enable_short_turn=True))
        assert not pool.matches(scenarios, flagged, None)
        config_changed = replace(
            run, runtime=replace(run.runtime, validate_observation=False)
        )
        assert not pool.matches(scenarios, config_changed, None)
        forecast_on = replace(run, forecast=ForecastConfig(enabled=True))
        assert not pool.matches(scenarios, forecast_on, None)
        physical = replace(run, physical=replace(run.physical, capacity=39))
        assert not pool.matches(scenarios, physical, None)
        with pytest.raises(ValueError, match="does not match"):
            evaluate_scenarios(
                list(reversed(scenarios)),
                run,
                "ppo",
                model=reference_bundle["model"],
                pool=pool,
            )
        refreshed = pool.refresh(list(reversed(scenarios)), run, None)
        assert refreshed is not pool
        assert pool.closed
        refreshed.close()
    finally:
        pool.close()


def test_pool_does_not_reuse_metrics_across_checkpoint_loads(reference_bundle):
    run = reference_bundle["run"]
    scenarios = reference_bundle["scenarios"][:3]
    model = reference_bundle["model"]
    batched_run = replace(run, runtime=replace(run.runtime, eval_batch_size=4, reuse_eval_pool=True))
    pool = EvalEnvPool(scenarios, batched_run)
    original = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    try:
        first, _ = _eval(batched_run, scenarios, model, pool=pool)
        with torch.no_grad():
            for parameter in model.policy.parameters():
                parameter.add_(0.25)
        second, _ = _eval(batched_run, scenarios, model, pool=pool)
        scalar_after, _ = _eval(run, scenarios, model)
        assert first["total_cost"].tolist() != second["total_cost"].tolist()
        _assert_frames(second, scalar_after)
    finally:
        model.policy.load_state_dict(original)
        pool.close()


def test_batch_forecast_matches_scalar_or_errors_clearly(reference_bundle):
    run = reference_bundle["run"]
    scenarios = [generate_scenario(2001), generate_scenario(2002), generate_scenario(2003)]
    model = reference_bundle["model"]
    forecaster = HistoricalForecaster(tick_s=scenarios[0].config.tick_s)
    forecaster.fit([scenario.arrival_tape for scenario in scenarios], scenarios[0].config)
    forecast_run = replace(run, forecast=ForecastConfig(enabled=True))
    scalar, _ = evaluate_scenarios(
        scenarios, forecast_run, "ppo", model=model, forecaster=forecaster
    )
    batched_run = replace(
        forecast_run, runtime=replace(forecast_run.runtime, eval_batch_size=4)
    )
    batched, _ = evaluate_scenarios(
        scenarios, batched_run, "ppo", model=model, forecaster=forecaster
    )
    _assert_frames(scalar, batched)
    enabled_without_forecaster = replace(
        run,
        forecast=ForecastConfig(enabled=True),
        runtime=replace(run.runtime, eval_batch_size=4),
    )
    with pytest.raises(ValueError, match="forecaster"):
        validate_eval_runtime(enabled_without_forecaster, "ppo", forecaster=None)


def test_random_heuristic_stays_scalar_or_errors(reference_bundle):
    run = reference_bundle["run"]
    scenarios = reference_bundle["scenarios"][:2]
    scalar, _ = evaluate_scenarios(scenarios, run, "random", model_seed=0)
    assert len(scalar) == 2
    batched = replace(run, runtime=replace(run.runtime, eval_batch_size=4))
    with pytest.raises(ValueError, match="deterministic PPO"):
        evaluate_scenarios(scenarios, batched, "random", model_seed=0)


def test_callback_tie_is_strict_less_than(tmp_path, monkeypatch, reference_bundle):
    run = reference_bundle["run"]
    costs = iter([10.0, 10.0, 9.0])
    monkeypatch.setattr("bus_rl.training.callbacks.mean_cost", lambda *args, **kwargs: next(costs))
    saved: list[str] = []

    class Dummy:
        def save(self, path):
            saved.append(Path(path).name)

    callback = BestValidationCallback(reference_bundle["scenarios"][:2], run, tmp_path)
    callback.model = Dummy()
    callback.num_timesteps = 1
    callback._evaluate()
    callback.num_timesteps = 2
    callback._evaluate()
    callback.num_timesteps = 3
    callback._evaluate()
    assert saved == ["best", "best"]
    assert callback.best == 9.0
    assert [row["val_cost"] for row in callback.history] == [10.0, 10.0, 9.0]
    callback.finalize()
    assert callback._pool is None


def test_callback_restores_model_mode_and_rng(tmp_path, reference_bundle):
    run = replace(
        reference_bundle["run"],
        runtime=replace(reference_bundle["run"].runtime, eval_batch_size=2, reuse_eval_pool=True),
    )
    model = reference_bundle["model"]
    model.policy.set_training_mode(True)
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()
    callback = BestValidationCallback(
        reference_bundle["scenarios"][:2], run, tmp_path, eval_limit=2
    )
    callback.model = model
    callback.num_timesteps = 12_288
    cost = callback._evaluate()
    assert np.isfinite(cost)
    assert model.policy.training is True
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert callback._pool is not None and not callback._pool.closed
    stores = [id(env._scenario_store) for env in callback._pool.envs]
    assert len(set(stores)) == 1
    callback.finalize()
    assert callback._pool is None
    assert (tmp_path / "best.zip").exists()
    assert (tmp_path / "last.zip").exists()
    assert (tmp_path / "evaluations.csv").exists()


def test_mean_cost_uses_shipped_evaluate_path(reference_bundle):
    run = reference_bundle["run"]
    scenarios = reference_bundle["scenarios"][:4]
    model = reference_bundle["model"]
    frame, _ = _eval(run, scenarios, model)
    batched_run = replace(run, runtime=replace(run.runtime, eval_batch_size=4))
    np.testing.assert_allclose(
        mean_cost(model, scenarios, run),
        float(frame["total_cost"].mean()),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        mean_cost(model, scenarios, batched_run),
        float(frame["total_cost"].mean()),
        rtol=1e-9,
        atol=1e-9,
    )


def test_pool_shares_store_and_caps_at_remaining_days(reference_bundle):
    run = replace(
        reference_bundle["run"],
        runtime=replace(reference_bundle["run"].runtime, eval_batch_size=16),
    )
    scenarios = reference_bundle["scenarios"][:5]
    pool = EvalEnvPool(scenarios, run)
    try:
        assert pool.requested_batch_size == 16
        assert pool.batch_size == 5
        stores = [env._scenario_store for env in pool.envs]
        assert all(store is stores[0] for store in stores)
    finally:
        pool.close()
        assert pool.closed
