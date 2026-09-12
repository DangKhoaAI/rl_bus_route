"""Deterministic episode evaluation and CSV metrics."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from bus_rl.baselines import (
    FixedController,
    ProportionalController,
    RandomValidController,
    ThresholdController,
)
from bus_rl.config import ControlConfig, RunConfig
from bus_rl.control.actions import ACTION_TABLE
from bus_rl.domain import StepCosts
from bus_rl.env.factory import make_env_for_run
from bus_rl.evaluation.pool import EvalEnvPool, make_eval_pool
from bus_rl.evaluation.summary import from_native_payload, from_python_state, summarize_inputs
from bus_rl.rewards.costs import RewardConfig, add_costs
from bus_rl.runtime import apply_torch_threads
from bus_rl.timing import TIMERS

PPO_METHODS = frozenset({"ppo", "maskable_ppo"})


def make_controller(method: str, model=None, seed: int = 0):
    if method == "fixed":
        return FixedController()
    if method == "threshold":
        return ThresholdController()
    if method == "proportional":
        return ProportionalController()
    if method == "random":
        return RandomValidController(seed)
    if method in PPO_METHODS:
        if model is None:
            raise ValueError("ppo controller requires a loaded model")
        return PPOController(model)
    raise ValueError(f"unknown method: {method}")


class PPOController:
    def __init__(self, model):
        self.model = model

    def act(self, observation, mask):
        action, _ = self.model.predict(observation, action_masks=mask, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    def act_batch(self, observations: list, masks: list) -> list[int]:
        """One deterministic predict over a batch of observations/masks."""
        if not observations:
            return []
        stacked_obs = {
            key: np.stack([np.asarray(observation[key]) for observation in observations])
            for key in observations[0]
        }
        stacked_masks = np.stack([np.asarray(mask) for mask in masks])
        actions, _ = self.model.predict(
            stacked_obs, action_masks=stacked_masks, deterministic=True
        )
        values = [int(value) for value in np.asarray(actions).reshape(-1)]
        if len(values) != len(observations):
            raise RuntimeError(
                f"batched predict returned {len(values)} actions for {len(observations)} slots"
            )
        return values


def summarize_episode(
    state,
    scenario,
    costs: StepCosts,
    reward: float,
    reward_config: RewardConfig | None = None,
) -> dict:
    """Compatibility wrapper around the shared backend-agnostic summary."""
    return summarize_inputs(from_python_state(state), scenario, costs, reward, reward_config)


def _trace_row(env, action_kind: str, action_index: int | None = None) -> dict:
    row = dict(env.trace_snapshot())
    row["action"] = action_kind
    if action_index is not None:
        row["action_index"] = int(action_index)
    return row


def _want_trace(index: int, trace_index: int | None, trace_all: bool) -> bool:
    return bool(trace_all) or trace_index == index


def validate_eval_runtime(run: RunConfig, method: str, *, forecaster=None) -> None:
    """Reject unsupported O1 combinations; never silently fall back."""
    batch_size = int(run.runtime.eval_batch_size)
    if batch_size < 1:
        raise ValueError(f"runtime.eval_batch_size must be >= 1, got {batch_size!r}")
    if method not in PPO_METHODS and batch_size > 1:
        raise ValueError(
            f"runtime.eval_batch_size={batch_size} is only supported for deterministic PPO; "
            f"{method} stays on the scalar evaluator (set eval_batch_size=1)"
        )
    forecast_requested = forecaster is not None or bool(run.forecast.enabled)
    if batch_size > 1 and forecast_requested:
        if forecaster is None:
            raise ValueError(
                "runtime.eval_batch_size>1 with forecast enabled requires a fitted per-slot "
                "forecaster; the scalar path still runs with eval_batch_size=1"
            )
        if getattr(forecaster, "has_episode_state", False) and not getattr(
            forecaster, "per_slot_state", False
        ):
            raise ValueError(
                "runtime.eval_batch_size>1 with a stateful shared forecaster is unsupported; "
                "use eval_batch_size=1 (scalar) or a per-slot forecast state"
            )
        if not callable(getattr(forecaster, "predict", None)):
            raise ValueError(
                "runtime.eval_batch_size>1 with forecast requires a forecaster.predict "
                "implementation; the scalar path still runs with eval_batch_size=1"
            )


def _save_model_rng(model) -> dict:
    import random

    import torch

    policy = getattr(model, "policy", None)
    return {
        "training": None if policy is None else bool(policy.training),
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_model_rng(model, snapshot: dict) -> None:
    import random

    import torch

    policy = getattr(model, "policy", None)
    if policy is not None and snapshot["training"] is not None:
        policy.set_training_mode(snapshot["training"])
    torch.set_rng_state(snapshot["torch"])
    np.random.set_state(snapshot["numpy"])
    random.setstate(snapshot["python"])


def rollout(
    env,
    controller,
    scenario_index: int,
    *,
    trace: bool = False,
) -> tuple[dict, list[dict]]:
    started = perf_counter()
    observation, _ = env.reset(seed=0, options={"scenario_index": scenario_index})
    reward_sum = 0.0
    costs = StepCosts()
    rows: list[dict] = []

    while True:
        with TIMERS.span("eval.action_mask"):
            mask = env.action_masks()
        with TIMERS.span("eval.infer"):
            action = controller.act(observation, mask)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        costs = add_costs(costs, info["costs"])
        if trace:
            rows.append(_trace_row(env, ACTION_TABLE[action].kind, action_index=action))
        if terminated or truncated:
            break
    with TIMERS.span("eval.summarize"):
        metrics = summarize_inputs(
            env.summary_inputs(), env.scenario, costs, reward_sum, env.reward
        )
    metrics["wall_s"] = perf_counter() - started
    metrics["scenario_index"] = scenario_index
    return metrics, rows


def _evaluate_scalar(
    scenarios,
    run: RunConfig,
    method: str,
    *,
    model,
    model_seed: int | None,
    split: str,
    forecaster,
    trace_index: int | None,
    trace_all: bool,
) -> tuple[pd.DataFrame, dict[int, list[dict]]]:
    env = make_env_for_run(scenarios, run, forecaster=forecaster)
    controller = make_controller(method, model=model, seed=model_seed or 0)
    records = []
    traces: dict[int, list[dict]] = {}
    for index in range(len(scenarios)):
        metrics, rows = rollout(
            env,
            controller,
            index,
            trace=_want_trace(index, trace_index, trace_all),
        )
        metrics["method"] = method
        metrics["model_seed"] = model_seed
        metrics["split"] = split
        records.append(metrics)
        if rows:
            traces[index] = rows
    return pd.DataFrame.from_records(records), traces


def _evaluate_batched(
    scenarios,
    run: RunConfig,
    method: str,
    *,
    model,
    model_seed: int | None,
    split: str,
    forecaster,
    trace_index: int | None,
    trace_all: bool,
    pool: EvalEnvPool,
) -> tuple[pd.DataFrame, dict[int, list[dict]]]:
    """Assign scenarios in input order; one batched predict per control step.

    Env slots still step sequentially. Done slots are not stepped. Terminal
    summary/trace is captured before a slot is reset for the next day.
    """
    del forecaster  # bound into the pool's envs
    controller = make_controller(method, model=model, seed=model_seed or 0)
    n_days = len(scenarios)
    records: list[dict | None] = [None] * n_days
    traces: dict[int, list[dict]] = {}
    batch_size = min(pool.batch_size, n_days)
    offset = 0
    while offset < n_days:
        chunk = list(range(offset, min(offset + batch_size, n_days)))
        slots: list[dict] = []
        for slot_id, scenario_index in enumerate(chunk):
            env = pool.envs[slot_id]
            started = perf_counter()
            observation, _ = env.reset(seed=0, options={"scenario_index": scenario_index})
            slots.append(
                {
                    "env": env,
                    "scenario_index": scenario_index,
                    "observation": observation,
                    "reward_sum": 0.0,
                    "costs": StepCosts(),
                    "done": False,
                    "rows": [],
                    "started": started,
                    "trace": _want_trace(scenario_index, trace_index, trace_all),
                }
            )
        while True:
            active = [slot for slot in slots if not slot["done"]]
            if not active:
                break
            with TIMERS.span("eval.action_mask"):
                masks = [slot["env"].action_masks() for slot in active]
            with TIMERS.span("eval.infer"):
                if isinstance(controller, PPOController):
                    actions = controller.act_batch(
                        [slot["observation"] for slot in active], masks
                    )
                else:
                    actions = [
                        int(controller.act(slot["observation"], mask))
                        for slot, mask in zip(active, masks, strict=True)
                    ]
            for slot, action in zip(active, actions, strict=True):
                env = slot["env"]
                observation, reward, terminated, truncated, info = env.step(action)
                slot["observation"] = observation
                slot["reward_sum"] += float(reward)
                slot["costs"] = add_costs(slot["costs"], info["costs"])
                if slot["trace"]:
                    slot["rows"].append(
                        _trace_row(env, ACTION_TABLE[int(action)].kind, action_index=int(action))
                    )
                if not (terminated or truncated):
                    continue
                slot["done"] = True
                with TIMERS.span("eval.summarize"):
                    metrics = summarize_inputs(
                        env.summary_inputs(),
                        env.scenario,
                        slot["costs"],
                        slot["reward_sum"],
                        env.reward,
                    )
                metrics["wall_s"] = perf_counter() - slot["started"]
                metrics["scenario_index"] = slot["scenario_index"]
                metrics["method"] = method
                metrics["model_seed"] = model_seed
                metrics["split"] = split
                records[slot["scenario_index"]] = metrics
                if slot["rows"]:
                    traces[slot["scenario_index"]] = slot["rows"]
        offset += len(chunk)
    missing = [index for index, row in enumerate(records) if row is None]
    if missing:
        raise RuntimeError(f"evaluation missed scenario indices {missing}")
    return pd.DataFrame.from_records(records), traces


def _native_trace_row(pool, slot: int, action_index: int) -> dict:
    row = dict(pool.kernel.trace_snapshot(slot))
    row["action"] = ACTION_TABLE[int(action_index)].kind
    row["action_index"] = int(action_index)
    return row


def _evaluate_batched_native(
    scenarios,
    run: RunConfig,
    method: str,
    *,
    model,
    model_seed: int | None,
    split: str,
    forecaster,
    trace_index: int | None,
    trace_all: bool,
    pool,
) -> tuple[pd.DataFrame, dict[int, list[dict]]]:
    """O2: one native `step_batch` per control step for all active slots."""
    from bus_rl.env.native_bus_dispatch import _step_costs

    del forecaster  # applied inside the pool's observation post-processing
    controller = make_controller(method, model=model, seed=model_seed or 0)
    n_days = len(scenarios)
    records: list[dict | None] = [None] * n_days
    traces: dict[int, list[dict]] = {}
    batch_size = min(pool.batch_size, n_days)
    offset = 0
    while offset < n_days:
        chunk = list(range(offset, min(offset + batch_size, n_days)))
        slot_ids = list(range(len(chunk)))
        observation, masks = pool.reset(slot_ids, chunk)
        slots: list[dict] = []
        for slot, scenario_index in zip(slot_ids, chunk, strict=True):
            slots.append(
                {
                    "slot": slot,
                    "scenario_index": scenario_index,
                    "observation": {key: value[slot] for key, value in observation.items()},
                    "reward_sum": 0.0,
                    "costs": StepCosts(),
                    "done": False,
                    "rows": [],
                    "started": perf_counter(),
                    "trace": _want_trace(scenario_index, trace_index, trace_all),
                }
            )
        while True:
            active = [slot for slot in slots if not slot["done"]]
            if not active:
                break
            active_slots = [slot["slot"] for slot in active]
            with TIMERS.span("eval.action_mask"):
                active_masks = [np.asarray(masks[slot["slot"]]) for slot in active]
            with TIMERS.span("eval.infer"):
                if isinstance(controller, PPOController):
                    actions = controller.act_batch(
                        [slot["observation"] for slot in active], active_masks
                    )
                else:
                    actions = [
                        int(controller.act(slot["observation"], mask))
                        for slot, mask in zip(active, active_masks, strict=True)
                    ]
            result = pool.step(active_slots, actions)
            step_obs = result["obs"]
            step_rewards = np.asarray(result["reward"])
            step_terminated = np.asarray(result["terminated"])
            step_truncated = np.asarray(result["truncated"])
            step_costs = np.asarray(result["costs"])
            step_masks = np.asarray(result["mask"])
            for position, slot_state in enumerate(active):
                slot = slot_state["slot"]
                slot_state["observation"] = {
                    key: value[position] for key, value in step_obs.items()
                }
                slot_state["reward_sum"] += float(step_rewards[position])
                slot_state["costs"] = add_costs(
                    slot_state["costs"], _step_costs(step_costs[position])
                )
                masks[slot] = step_masks[position]
                if slot_state["trace"]:
                    slot_state["rows"].append(
                        _native_trace_row(pool, slot, int(actions[position]))
                    )
                if not (bool(step_terminated[position]) or bool(step_truncated[position])):
                    continue
                slot_state["done"] = True
                scenario_index = slot_state["scenario_index"]
                with TIMERS.span("eval.summarize"):
                    metrics = summarize_inputs(
                        from_native_payload(pool.kernel.summary_inputs(slot)),
                        pool.applied[scenario_index],
                        slot_state["costs"],
                        slot_state["reward_sum"],
                        run.reward,
                    )
                metrics["wall_s"] = perf_counter() - slot_state["started"]
                metrics["scenario_index"] = scenario_index
                metrics["method"] = method
                metrics["model_seed"] = model_seed
                metrics["split"] = split
                records[scenario_index] = metrics
                if slot_state["rows"]:
                    traces[scenario_index] = slot_state["rows"]
        offset += len(chunk)
    missing = [index for index, row in enumerate(records) if row is None]
    if missing:
        raise RuntimeError(f"evaluation missed scenario indices {missing}")
    return pd.DataFrame.from_records(records), traces


def evaluate_scenarios(
    scenarios,
    run: RunConfig,
    method: str,
    *,
    model=None,
    model_seed: int | None = None,
    split: str = "",
    forecaster=None,
    trace_index: int | None = None,
    trace_all: bool = False,
    pool: EvalEnvPool | None = None,
) -> tuple[pd.DataFrame, dict[int, list[dict]]]:
    validate_eval_runtime(run, method, forecaster=forecaster)
    snapshot = None
    if model is not None:
        apply_torch_threads(run.algorithm.torch_threads)
        snapshot = _save_model_rng(model)
    try:
        batch_size = int(run.runtime.eval_batch_size)
        native = bool(run.runtime.native_batch) and method in PPO_METHODS
        use_batched = method in PPO_METHODS and (batch_size > 1 or pool is not None or native)
        if not use_batched:
            return _evaluate_scalar(
                scenarios,
                run,
                method,
                model=model,
                model_seed=model_seed,
                split=split,
                forecaster=forecaster,
                trace_index=trace_index,
                trace_all=trace_all,
            )
        owned = None
        active = pool
        if active is None:
            owned = make_eval_pool(scenarios, run, forecaster)
            active = owned
        elif active.closed:
            raise ValueError("eval pool has been released")
        elif not active.matches(scenarios, run, forecaster):
            raise ValueError(
                "eval pool does not match the requested scenario order/hash, flags, "
                "config, forecast or contract; rebuild the pool (no silent reuse)"
            )
        try:
            if native:
                return _evaluate_batched_native(
                    scenarios,
                    run,
                    method,
                    model=model,
                    model_seed=model_seed,
                    split=split,
                    forecaster=forecaster,
                    trace_index=trace_index,
                    trace_all=trace_all,
                    pool=active,
                )
            return _evaluate_batched(
                scenarios,
                run,
                method,
                model=model,
                model_seed=model_seed,
                split=split,
                forecaster=forecaster,
                trace_index=trace_index,
                trace_all=trace_all,
                pool=active,
            )
        finally:
            if owned is not None:
                owned.close()
    finally:
        if model is not None and snapshot is not None:
            _restore_model_rng(model, snapshot)


def write_results(frame: pd.DataFrame, output: Path, traces: dict | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "results.csv", index=False)
    if traces:
        import json

        (output / "events.jsonl").write_text(
            "\n".join(json.dumps(row, default=str) for rows in traces.values() for row in rows)
            + "\n"
        )


def mean_cost(
    model,
    scenarios,
    run: RunConfig,
    forecaster=None,
    limit: int | None = None,
    pool: EvalEnvPool | None = None,
) -> float:
    subset = list(scenarios) if limit is None else list(scenarios)[:limit]
    frame, _ = evaluate_scenarios(
        subset,
        run,
        "ppo",
        model=model,
        model_seed=run.algorithm.seed,
        forecaster=forecaster,
        pool=pool,
    )
    return float(frame["total_cost"].mean())


def apply_control(run: RunConfig, control: ControlConfig | None) -> RunConfig:
    return (
        run
        if control is None
        else RunConfig(
            physical=run.physical,
            control=control,
            reward=run.reward,
            algorithm=run.algorithm,
            forecast=run.forecast,
            runtime=run.runtime,
            source=run.source,
        )
    )
