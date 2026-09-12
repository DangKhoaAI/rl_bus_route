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
from bus_rl.domain import StepCosts
from bus_rl.env.factory import make_env_for_run
from bus_rl.evaluation.summary import from_python_state, summarize_inputs
from bus_rl.rewards.costs import RewardConfig, add_costs
from bus_rl.timing import TIMERS


def make_controller(method: str, model=None, seed: int = 0):
    if method == "fixed":
        return FixedController()
    if method == "threshold":
        return ThresholdController()
    if method == "proportional":
        return ProportionalController()
    if method == "random":
        return RandomValidController(seed)
    if method in {"ppo", "maskable_ppo"}:
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


def summarize_episode(
    state,
    scenario,
    costs: StepCosts,
    reward: float,
    reward_config: RewardConfig | None = None,
) -> dict:
    """Compatibility wrapper around the shared backend-agnostic summary."""
    return summarize_inputs(from_python_state(state), scenario, costs, reward, reward_config)


def _trace_row(env, action_kind: str) -> dict:
    row = dict(env.trace_snapshot())
    row["action"] = action_kind
    return row


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
    from bus_rl.control.actions import ACTION_TABLE

    while True:
        mask = env.action_masks()
        action = controller.act(observation, mask)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        costs = add_costs(costs, info["costs"])
        if trace:
            rows.append(_trace_row(env, ACTION_TABLE[action].kind))
        if terminated or truncated:
            break
    with TIMERS.span("eval.summarize"):
        metrics = summarize_inputs(
            env.summary_inputs(), env.scenario, costs, reward_sum, env.reward
        )
    metrics["wall_s"] = perf_counter() - started
    metrics["scenario_index"] = scenario_index
    return metrics, rows


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
) -> tuple[pd.DataFrame, dict[int, list[dict]]]:
    env = make_env_for_run(
        scenarios,
        run,
        forecaster=forecaster,
    )
    controller = make_controller(method, model=model, seed=model_seed or 0)
    records = []
    traces: dict[int, list[dict]] = {}
    for index in range(len(scenarios)):
        metrics, rows = rollout(env, controller, index, trace=trace_index == index)
        metrics["method"] = method
        metrics["model_seed"] = model_seed
        metrics["split"] = split
        records.append(metrics)
        if rows:
            traces[index] = rows
    return pd.DataFrame.from_records(records), traces


def write_results(frame: pd.DataFrame, output: Path, traces: dict | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "results.csv", index=False)
    if traces:
        import json

        (output / "events.jsonl").write_text(
            "\n".join(json.dumps(row, default=str) for rows in traces.values() for row in rows)
            + "\n"
        )


def mean_cost(model, scenarios, run: RunConfig, forecaster=None, limit: int | None = None) -> float:
    subset = list(scenarios) if limit is None else list(scenarios)[:limit]
    frame, _ = evaluate_scenarios(
        subset, run, "ppo", model=model, model_seed=run.algorithm.seed, forecaster=forecaster
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
