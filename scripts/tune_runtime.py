#!/usr/bin/env python
"""R4 runtime tuning (light): finalize Torch threads and measure validation cost.

Runs a small, same-condition comparison for Python and Rust:
- isolated learn at a few Torch thread counts (interleaved per repetition),
- fixed-checkpoint eval at those thread counts,
- observation-validation cost at the env level and end-to-end.

Writes ``reports/rust-migration/runtime-tuning.json``. Not an acceptance run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl.config import ControlConfig, RuntimeConfig, load_run_config
from bus_rl.data.scenario import generate_manifest
from bus_rl.env.factory import make_env_for_run
from bus_rl.evaluation.runner import make_controller, rollout
from bus_rl.parity.controllers import CoverageController
from bus_rl.rewards.costs import RewardConfig
from bus_rl.runtime import apply_torch_threads
from bus_rl.training.checkpoint import load_metadata, load_model
from bus_rl.training.train import make_env, make_model

REPORT_DIR = ROOT / "reports" / "rust-migration"
REFERENCE = ROOT / "tests" / "backend_parity" / "reference" / "diagnose-after" / "last.zip"
BACKENDS = ("python", "rust")


def _run_for(backend: str, *, validate: bool = True, threads: int = 2):
    core = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    return replace(
        core,
        runtime=RuntimeConfig(backend=backend, validate_observation=validate),
        algorithm=replace(core.algorithm, torch_threads=threads),
    )


def _stats(times: list[float]) -> dict:
    return {
        "wall_times_s": [round(value, 6) for value in times],
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
    }


def _learn_once(scenarios, run, transitions: int) -> float:
    env = DummyVecEnv(
        [make_env(scenarios, run, run.algorithm.seed + i) for i in range(run.algorithm.n_envs)]
    )
    model = make_model(env, run.algorithm.seed, run.algorithm)
    started = perf_counter()
    model.learn(total_timesteps=transitions)
    elapsed = perf_counter() - started
    env.close()
    return elapsed


def thread_matrix(threads: list[int], repetitions: int, transitions: int, warmup: int) -> dict:
    scenarios = generate_manifest("train", 16)
    runs = {backend: {t: _run_for(backend, threads=t) for t in threads} for backend in BACKENDS}
    results = {backend: {t: [] for t in threads} for backend in BACKENDS}
    for t in threads:  # per thread count: warm up both backends
        apply_torch_threads(t)
        for backend in BACKENDS:
            for _ in range(warmup):
                _learn_once(scenarios, runs[backend][t], transitions)
    for _ in range(repetitions):  # interleave backends within a repetition
        for t in threads:
            apply_torch_threads(t)
            for backend in BACKENDS:
                results[backend][t].append(_learn_once(scenarios, runs[backend][t], transitions))
    summary = {}
    for backend in BACKENDS:
        summary[backend] = {}
        for t in threads:
            stats = _stats(results[backend][t])
            stats["transitions_per_s_median"] = transitions / stats["median_s"]
            summary[backend][t] = stats
            print(
                f"[tune] learn {backend:6s} threads={t:2d} median={stats['median_s']:.3f}s "
                f"({stats['transitions_per_s_median']:.1f} t/s)",
                flush=True,
            )
    best = {}
    for backend in BACKENDS:
        best[backend] = min(threads, key=lambda t: summary[backend][t]["median_s"])
    return {
        "transitions": transitions,
        "repetitions": repetitions,
        "warmup": warmup,
        "threads": threads,
        "results": summary,
        "best_threads": best,
    }


def eval_matrix(threads: list[int], repetitions: int, warmup: int) -> dict:
    run = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    metadata = load_metadata(REFERENCE)
    run = replace(
        run,
        control=ControlConfig(**metadata["control"]),
        reward=RewardConfig(
            **{
                key: metadata["reward"][key]
                for key in RewardConfig.__dataclass_fields__
                if key in metadata["reward"]
            }
        ),
    )
    scenarios = generate_manifest("validation", 100)[:10]
    model, metadata = load_model(REFERENCE, make_env_for_run(scenarios[:1], run), run.physical)
    controller = make_controller("ppo", model=model, seed=metadata.get("seed", 11))
    envs = {
        (backend, t): make_env_for_run(
            scenarios,
            replace(
                run,
                runtime=RuntimeConfig(backend=backend),
                algorithm=replace(run.algorithm, torch_threads=t),
            ),
        )
        for backend in BACKENDS
        for t in threads
    }
    results = {backend: {t: [] for t in threads} for backend in BACKENDS}
    for t in threads:
        apply_torch_threads(t)
        for backend in BACKENDS:
            for _ in range(warmup):
                for index in range(len(scenarios)):
                    rollout(envs[(backend, t)], controller, index)
    for _ in range(repetitions):
        for t in threads:
            apply_torch_threads(t)
            for backend in BACKENDS:
                env = envs[(backend, t)]
                started = perf_counter()
                for index in range(len(scenarios)):
                    rollout(env, controller, index)
                results[backend][t].append(perf_counter() - started)
    summary = {}
    for backend in BACKENDS:
        summary[backend] = {}
        for t in threads:
            stats = _stats(results[backend][t])
            summary[backend][t] = stats
            print(
                f"[tune] eval  {backend:6s} threads={t:2d} median={stats['median_s']:.3f}s",
                flush=True,
            )
    return {"days": len(scenarios), "repetitions": repetitions, "results": summary}


def _trace(scenario, run) -> list[int]:
    env = make_env_for_run([scenario], run)
    observation, _ = env.reset(seed=0, options={"scenario_index": 0})
    controller = CoverageController()
    actions = []
    while True:
        mask = env.action_masks()
        action = controller.act(observation, mask)
        actions.append(int(action))
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return actions


def validation_cost(repetitions: int) -> dict:
    base = _run_for("python")
    scenario = generate_manifest("train", 1)[0]
    actions = _trace(scenario, base)
    summary = {}
    for backend in BACKENDS:
        summary[backend] = {}
        for validate in (True, False):
            env = make_env_for_run(
                [scenario],
                replace(
                    base,
                    runtime=RuntimeConfig(backend=backend, validate_observation=validate),
                ),
            )
            times = []
            for _ in range(repetitions):
                env.reset(seed=0, options={"scenario_index": 0})
                started = perf_counter()
                for action in actions:
                    env.step(action)
                times.append(perf_counter() - started)
            stats = _stats(times)
            stats["decisions"] = len(actions)
            summary[backend]["validate" if validate else "no_validate"] = stats
            print(
                f"[tune] env {backend:6s} validate={validate!s:5s} median={stats['median_s']:.4f}s",
                flush=True,
            )
    return {"decisions": len(actions), "repetitions": repetitions, "results": summary}


def learn_validation(repetitions: int, transitions: int, warmup: int) -> dict:
    scenarios = generate_manifest("train", 16)
    summary = {}
    for backend in BACKENDS:
        summary[backend] = {}
        for validate in (True, False):
            run = _run_for(backend, validate=validate, threads=2)
            apply_torch_threads(2)
            for _ in range(warmup):
                _learn_once(scenarios, run, transitions)
            times = [_learn_once(scenarios, run, transitions) for _ in range(repetitions)]
            stats = _stats(times)
            stats["transitions_per_s_median"] = transitions / stats["median_s"]
            summary[backend]["validate" if validate else "no_validate"] = stats
            print(
                f"[tune] learn {backend:6s} validate={validate!s:5s} "
                f"median={stats['median_s']:.3f}s",
                flush=True,
            )
    return {"transitions": transitions, "repetitions": repetitions, "results": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", default="2,4,16")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=12_288)
    parser.add_argument("--skip-thread-matrix", action="store_true")
    parser.add_argument("--skip-eval-matrix", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "runtime-tuning.json")
    args = parser.parse_args()

    threads = [int(value) for value in args.threads.split(",") if value.strip()]
    started = perf_counter()
    payload: dict = {
        "tuning_schema_version": 1,
        "role": "r4-runtime-tuning",
        "protocol": {
            "transitions": args.transitions,
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "same_condition": "both backends share the process thread count",
        },
        "machine": {
            "platform": platform.platform(),
            "cpu_count": len(os.sched_getaffinity(0)),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }
    if not args.skip_thread_matrix:
        payload["thread_matrix"] = thread_matrix(
            threads, args.repetitions, args.transitions, args.warmup
        )
    if not args.skip_eval_matrix:
        payload["eval_matrix"] = eval_matrix(threads, args.repetitions, args.warmup)
    if not args.skip_validation:
        payload["validation_env"] = validation_cost(repetitions=5)
        payload["validation_learn"] = learn_validation(
            repetitions=args.repetitions, transitions=args.transitions, warmup=args.warmup
        )
    payload["wall_total_s"] = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[tune] wrote {args.output.relative_to(ROOT)} in {payload['wall_total_s']:.1f}s")


if __name__ == "__main__":
    main()
