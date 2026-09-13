#!/usr/bin/env python
"""R4.2 interleaved speed benchmark: Python oracle vs native Rust backend.

Light pass by default: simulation-only over the frozen workload traces plus
isolated learn (>=12,288 transitions) and the fixed-checkpoint evaluation.
Python and Rust runs are interleaved within each repetition so thermal/load
drift affects both equally. cProfile and the in-tree TIMERS stay disabled.

Usage:
    python scripts/benchmark_backends.py [--repetitions 5] [--transitions 12288]
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
from bus_rl.config import ControlConfig, RuntimeConfig, load_run_config
from bus_rl.evaluation.runner import make_controller, rollout
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.runtime import apply_torch_threads
from bus_rl.execution.scenarios.generation import generate_manifest
from bus_rl.execution.timing import TIMERS
from bus_rl.learning.training.checkpoint import load_metadata, load_model
from bus_rl.learning.training.train import make_env, make_model
from bus_sim.oracle.costs import RewardConfig
from bus_sim.parity.controllers import CoverageController
from bus_sim.parity.scenarios import CATALOG, build_scenario, run_config

REPORT_DIR = ROOT / "reports" / "rust-migration"
REFERENCE = ROOT / "tests" / "backend_parity" / "reference" / "diagnose-after" / "last.zip"
SIM_WORKLOADS = ["zero_m3", "normal_m3", "peak_m3", "burst_m3", "traffic_m3"]
TICKS_PER_DECISION = 4
BACKENDS = ("python", "rust")


class _NoEval(BaseCallback):
    """Keep learn isolated from periodic evaluation."""

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


def _run_for(backend: str):
    core = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    return replace(core, runtime=RuntimeConfig(backend=backend))


def _env_for(spec: dict, backend: str):
    cfg = run_config(spec)
    run = replace(cfg, runtime=RuntimeConfig(backend=backend))
    return make_env_for_run([build_scenario(spec)], run)


def precompute_trace(spec: dict) -> list[int]:
    env = _env_for(spec, "python")
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


def time_simulation(spec: dict, actions: list[int], backend: str) -> float:
    env = _env_for(spec, backend)
    env.reset(seed=0, options={"scenario_index": 0})
    started = perf_counter()
    for action in actions:
        env.step(action)
    return perf_counter() - started


def benchmark_simulation(repetitions: int, warmup: int) -> dict:
    traces = {name: precompute_trace(CATALOG[name]) for name in SIM_WORKLOADS}
    results: dict[str, dict] = {backend: {"workloads": {}, "aggregate": {}} for backend in BACKENDS}
    for name, actions in traces.items():
        times: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
        for backend in BACKENDS:
            for _ in range(warmup):
                time_simulation(CATALOG[name], actions, backend)
        for _ in range(repetitions):
            for backend in BACKENDS:
                times[backend].append(time_simulation(CATALOG[name], actions, backend))
        decisions = len(actions)
        ticks = decisions * TICKS_PER_DECISION
        for backend in BACKENDS:
            stats = _stats(times[backend])
            stats.update(
                {
                    "scenario": CATALOG[name],
                    "decisions": decisions,
                    "ticks": ticks,
                    "decisions_per_s_median": decisions / stats["median_s"],
                    "ticks_per_s_median": ticks / stats["median_s"],
                }
            )
            results[backend]["workloads"][name] = stats
        ratio = (
            results["python"]["workloads"][name]["median_s"]
            / results["rust"]["workloads"][name]["median_s"]
        )
        print(
            f"[bench] simulation {name:12s} py={results['python']['workloads'][name]['median_s']:.3f}s "
            f"rust={results['rust']['workloads'][name]['median_s']:.3f}s  speedup={ratio:.2f}x",
            flush=True,
        )
    for backend in BACKENDS:
        rows = results[backend]["workloads"]
        repetitions = len(next(iter(rows.values()))["wall_times_s"])
        totals = [
            sum(rows[name]["wall_times_s"][index] for name in rows) for index in range(repetitions)
        ]
        results[backend]["aggregate"] = {
            "per_rep_total_s": [round(value, 6) for value in totals],
            "median_total_s": statistics.median(totals),
            "min_total_s": min(totals),
            "max_total_s": max(totals),
            "decisions_total": sum(row["decisions"] for row in rows.values()),
            "sum_of_workload_medians_s": sum(row["median_s"] for row in rows.values()),
        }
    py_total = results["python"]["aggregate"]["median_total_s"]
    rust_total = results["rust"]["aggregate"]["median_total_s"]
    results["speedup"] = {"median_total": py_total / rust_total}
    results["paired_per_rep"] = {
        name: {
            "python": results["python"]["workloads"][name]["wall_times_s"],
            "rust": results["rust"]["workloads"][name]["wall_times_s"],
        }
        for name in traces
    }
    print(
        f"[bench] simulation total py={py_total:.3f}s rust={rust_total:.3f}s "
        f"speedup={py_total / rust_total:.2f}x",
        flush=True,
    )
    return results


def _learn_once(scenarios, run, transitions: int, callback_factory) -> tuple[float, float, int]:
    start = perf_counter()
    env = DummyVecEnv(
        [
            make_env(scenarios, run, run.algorithm.seed + index)
            for index in range(run.algorithm.n_envs)
        ]
    )
    model = make_model(env, run.algorithm.seed, run.algorithm)
    setup = perf_counter() - start
    callback = callback_factory()
    started = perf_counter()
    model.learn(total_timesteps=transitions, callback=callback)
    elapsed = perf_counter() - started
    actual = int(model.num_timesteps)
    env.close()
    return setup, elapsed, actual


def benchmark_learn(repetitions: int, warmup: int, transitions: int) -> dict:
    scenarios = generate_manifest("train", 16)
    runs = {backend: _run_for(backend) for backend in BACKENDS}
    results = {backend: {"setup_s": [], "wall_s": [], "actual": []} for backend in BACKENDS}
    for backend in BACKENDS:
        for _ in range(warmup):
            _learn_once(scenarios, runs[backend], transitions, _NoEval)
    for _ in range(repetitions):
        for backend in BACKENDS:
            setup, elapsed, actual = _learn_once(scenarios, runs[backend], transitions, _NoEval)
            results[backend]["setup_s"].append(setup)
            results[backend]["wall_s"].append(elapsed)
            results[backend]["actual"].append(actual)
    summary = {}
    for backend in BACKENDS:
        stats = _stats(results[backend]["wall_s"])
        stats.update(
            {
                "setup_median_s": statistics.median(results[backend]["setup_s"]),
                "transitions_requested": transitions,
                "transitions_actual": results[backend]["actual"],
                "transitions_per_s_median": transitions / stats["median_s"],
                "n_envs": runs[backend].algorithm.n_envs,
                "n_steps": runs[backend].algorithm.n_steps,
                "batch_size": runs[backend].algorithm.batch_size,
                "n_epochs": runs[backend].algorithm.n_epochs,
                "seed": runs[backend].algorithm.seed,
                "periodic_eval": False,
            }
        )
        summary[backend] = stats
        print(
            f"[bench] isolated_learn {backend:6s} median={stats['median_s']:.3f}s "
            f"({stats['transitions_per_s_median']:.1f} transitions/s)",
            flush=True,
        )
    summary["speedup"] = {
        "median_learn": summary["python"]["median_s"] / summary["rust"]["median_s"]
    }
    print(
        f"[bench] isolated_learn speedup={summary['speedup']['median_learn']:.2f}x",
        flush=True,
    )
    return summary


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
    load_env = make_env_for_run(scenarios[:1], run)
    model, metadata = load_model(REFERENCE, load_env, run.physical)
    controller = make_controller("ppo", model=model, seed=metadata.get("seed", 11))

    results = {backend: {"wall_s": [], "mean_cost": [], "per_day": []} for backend in BACKENDS}
    for backend in BACKENDS:  # warm-up per backend, reported separately
        env = make_env_for_run(scenarios, replace(run, runtime=RuntimeConfig(backend=backend)))
        for index in range(len(scenarios)):
            rollout(env, controller, index)
    for _ in range(repetitions):
        for backend in BACKENDS:
            env = make_env_for_run(scenarios, replace(run, runtime=RuntimeConfig(backend=backend)))
            started = perf_counter()
            costs = []
            for index in range(len(scenarios)):
                metrics, _ = rollout(env, controller, index)
                costs.append(metrics["total_cost"])
            results[backend]["wall_s"].append(perf_counter() - started)
            results[backend]["mean_cost"].append(float(np.mean(costs)))
            results[backend]["per_day"].append([float(value) for value in costs])
    summary = {}
    for backend in BACKENDS:
        stats = _stats(results[backend]["wall_s"])
        stats.update(
            {
                "checkpoint": str(REFERENCE.relative_to(ROOT)),
                "days": len(scenarios),
                "mean_total_cost": float(np.mean(results[backend]["mean_cost"])),
                "per_day_costs": results[backend]["per_day"][-1],
            }
        )
        summary[backend] = stats
        print(
            f"[bench] fixed_checkpoint_eval {backend:6s} median={stats['median_s']:.3f}s "
            f"mean_cost={stats['mean_total_cost']:.2f}",
            flush=True,
        )
    summary["speedup"] = {
        "median_eval": summary["python"]["median_s"] / summary["rust"]["median_s"]
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=12_288)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-learn", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--torch-threads", type=int, dest="torch_threads")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "speed-acceptance.json")
    args = parser.parse_args()

    core = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    threads = args.torch_threads if args.torch_threads is not None else core.algorithm.torch_threads
    actual_threads = apply_torch_threads(threads)

    TIMERS.enabled = False
    TIMERS.reset()
    if domain.CONSERVATION_CHECKS:
        raise SystemExit("conservation checks must be off for the speed reference")

    started = perf_counter()
    payload: dict = {
        "benchmark_schema_version": 1,
        "role": "r4.2-interleaved-light",
        "protocol": {
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "eval_warmup": 1,
            "interleaved": True,
            "profiler": "disabled",
            "timers": "disabled",
            "conservation_checks": "disabled",
            "cprofile_used": False,
            "torch_threads": actual_threads,
            "full_workflow_245760": "deferred",
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
    }
    if not args.skip_simulation:
        payload["simulation"] = benchmark_simulation(args.repetitions, args.warmup)
    if not args.skip_learn:
        payload["isolated_learn"] = benchmark_learn(args.repetitions, args.warmup, args.transitions)
    if not args.skip_eval:
        payload["fixed_checkpoint_eval"] = benchmark_eval(args.repetitions)
    payload["peak_rss_kb"] = _rss_kb()
    payload["wall_total_s"] = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[bench] wrote {args.output.relative_to(ROOT)} in {payload['wall_total_s']:.1f}s")


if __name__ == "__main__":
    main()
