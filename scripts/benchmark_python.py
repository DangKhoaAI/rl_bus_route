#!/usr/bin/env python
"""R0.3 unprofiled Python benchmark (simulation, isolated learn, eval).

Profiling is deliberately off: cProfile is never imported and the in-tree
``TIMERS`` are disabled. The script uses only deterministically regenerable
scenarios so the reference does not depend on the git-ignored ``data/generated``.

Usage:
    python scripts/benchmark_python.py [--repetitions 5] [--transitions 12288]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl import domain
from bus_rl.config import ControlConfig, load_run_config
from bus_rl.evaluation.runner import make_controller, rollout
from bus_rl.execution.environments.python.environment import BusDispatchEnv
from bus_rl.execution.scenarios.generation import generate_manifest
from bus_rl.execution.timing import TIMERS
from bus_rl.learning.training.checkpoint import load_metadata, load_model
from bus_rl.learning.training.train import make_env, make_model
from bus_sim.oracle.costs import RewardConfig
from bus_sim.parity.controllers import CoverageController
from bus_sim.parity.scenarios import CATALOG, build_scenario, run_config

REPORT_DIR = ROOT / "reports" / "rust-migration"
REFERENCE = ROOT / "runs" / "diagnose-after" / "last.zip"
SIM_WORKLOADS = ["zero_m3", "normal_m3", "peak_m3", "burst_m3", "traffic_m3"]
TICKS_PER_DECISION = 4


class _NoEval(BaseCallback):
    """Keeps learn isolated from periodic evaluation."""

    def _on_step(self) -> bool:
        return True


def _stats(times: list[float]) -> dict:
    return {
        "wall_times_s": [round(value, 6) for value in times],
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.fmean(times),
    }


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def precompute_trace(spec: dict) -> list[int]:
    env = BusDispatchEnv(
        [build_scenario(spec)],
        run_config(spec).physical,
        reward=run_config(spec).reward,
        control=run_config(spec).control,
    )
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


def time_simulation(spec: dict, actions: list[int]) -> float:
    env = BusDispatchEnv(
        [build_scenario(spec)],
        run_config(spec).physical,
        reward=run_config(spec).reward,
        control=run_config(spec).control,
    )
    env.reset(seed=0, options={"scenario_index": 0})
    started = perf_counter()
    for action in actions:
        env.step(action)
    return perf_counter() - started


def benchmark_simulation(repetitions: int, warmup: int) -> dict:
    traces = {name: precompute_trace(CATALOG[name]) for name in SIM_WORKLOADS}
    results = {}
    for name, actions in traces.items():
        for _ in range(warmup):
            time_simulation(CATALOG[name], actions)
        times = [time_simulation(CATALOG[name], actions) for _ in range(repetitions)]
        stats = _stats(times)
        decisions = len(actions)
        ticks = decisions * TICKS_PER_DECISION
        stats.update(
            {
                "scenario": CATALOG[name],
                "decisions": decisions,
                "ticks": ticks,
                "decisions_per_s_median": decisions / stats["median_s"],
                "ticks_per_s_median": ticks / stats["median_s"],
            }
        )
        results[name] = stats
        print(
            f"[bench] simulation {name:12s} median={stats['median_s']:.3f}s "
            f"decisions/s={stats['decisions_per_s_median']:.1f}",
            flush=True,
        )
    aggregate = {
        "median_total_s": sum(row["median_s"] for row in results.values()),
        "decisions_total": sum(row["decisions"] for row in results.values()),
    }
    return {"workloads": results, "aggregate": aggregate}


def benchmark_learn(repetitions: int, warmup: int, transitions: int, core) -> dict:
    scenarios = generate_manifest("train", 16)
    setup_times = []
    learn_times = []
    actual_transitions = []

    def one_run() -> tuple[float, float, int]:
        start = perf_counter()
        env = DummyVecEnv(
            [
                make_env(scenarios, core, core.algorithm.seed + index)
                for index in range(core.algorithm.n_envs)
            ]
        )
        model = make_model(env, core.algorithm.seed, core.algorithm)
        setup = perf_counter() - start
        started = perf_counter()
        model.learn(total_timesteps=transitions, callback=_NoEval())
        elapsed = perf_counter() - started
        actual = int(model.num_timesteps)
        env.close()
        return setup, elapsed, actual

    for _ in range(warmup):
        one_run()
    for _ in range(repetitions):
        setup, elapsed, actual = one_run()
        setup_times.append(setup)
        learn_times.append(elapsed)
        actual_transitions.append(actual)
    stats = _stats(learn_times)
    stats.update(
        {
            "setup_median_s": statistics.median(setup_times),
            "transitions_requested": transitions,
            "transitions_actual": actual_transitions,
            "n_envs": core.algorithm.n_envs,
            "n_steps": core.algorithm.n_steps,
            "batch_size": core.algorithm.batch_size,
            "n_epochs": core.algorithm.n_epochs,
            "seed": core.algorithm.seed,
            "periodic_eval": False,
            "transitions_per_s_median": transitions / stats["median_s"],
        }
    )
    print(
        f"[bench] isolated_learn median={stats['median_s']:.3f}s "
        f"({stats['transitions_per_s_median']:.1f} transitions/s)",
        flush=True,
    )
    return stats


def benchmark_eval(repetitions: int) -> dict:
    run = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    scenarios = generate_manifest("validation", 100)[:10]
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
    env = BusDispatchEnv(scenarios[:1], run.physical, reward=run.reward, control=run.control)
    model, metadata = load_model(REFERENCE, env, run.physical)
    env = BusDispatchEnv(scenarios, run.physical, reward=run.reward, control=run.control)
    controller = make_controller("ppo", model=model, seed=metadata.get("seed", 11))

    times = []
    costs = []
    for _ in range(repetitions):
        started = perf_counter()
        rep_costs = []
        for index in range(len(scenarios)):
            metrics, _ = rollout(env, controller, index)
            rep_costs.append(metrics["total_cost"])
        times.append(perf_counter() - started)
        costs.append(float(np.mean(rep_costs)))
    stats = _stats(times)
    stats.update(
        {
            "checkpoint": str(REFERENCE.relative_to(ROOT)),
            "days": len(scenarios),
            "scenario_source": "generate_manifest('validation', 100)[:10]",
            "mean_total_cost": float(np.mean(costs)),
            "mean_total_cost_per_rep": costs,
            "historical_mean_total_cost": 15471.25,
        }
    )
    print(
        f"[bench] fixed_checkpoint_eval median={stats['median_s']:.3f}s "
        f"mean_cost={stats['mean_total_cost']:.2f}",
        flush=True,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=12_288)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "python-benchmark.json")
    args = parser.parse_args()

    TIMERS.enabled = False
    TIMERS.reset()
    if domain.CONSERVATION_CHECKS:
        raise SystemExit("conservation checks must be off for the speed reference")

    core = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    started = perf_counter()
    simulation = benchmark_simulation(args.repetitions, args.warmup)
    learn = benchmark_learn(args.repetitions, args.warmup, args.transitions, core)
    evaluation = benchmark_eval(args.repetitions)

    payload = {
        "benchmark_schema_version": 1,
        "role": "python-after-reference",
        "protocol": {
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "profiler": "disabled",
            "timers": "disabled",
            "conservation_checks": "disabled",
            "cprofile_used": False,
            "note": "setup is reported separately from steady-state",
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": len(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "simulation": simulation,
        "isolated_learn": learn,
        "fixed_checkpoint_eval": evaluation,
        "peak_rss_kb": _rss_kb(),
        "wall_total_s": perf_counter() - started,
        "commands": {
            "benchmark": (
                f"python scripts/benchmark_python.py --repetitions {args.repetitions} "
                f"--transitions {args.transitions}"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[bench] wrote {args.output.relative_to(ROOT)} in {payload['wall_total_s']:.1f}s")


if __name__ == "__main__":
    main()
