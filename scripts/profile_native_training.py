#!/usr/bin/env python
"""R4 optimization profiling (light): Rust training segments, Torch threads,
eval segments and shared-store memory.

Everything here is short (default 4096 transitions, 3 repetitions) and is meant
to find where the remaining learn/eval time goes, not to replace the R4.2
acceptance benchmark. Writes ``reports/rust-migration/profile-native.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
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
from bus_rl.rewards.costs import RewardConfig
from bus_rl.timing import TIMERS
from bus_rl.training.checkpoint import load_metadata, load_model
from bus_rl.training.diagnose import DiagnoseCallback
from bus_rl.training.train import fit_algorithm, make_env, make_model

REPORT_DIR = ROOT / "reports" / "rust-migration"
PROFILE_RUNS = ROOT / "runs" / "rust-migration" / "profile"
REFERENCE = ROOT / "tests" / "backend_parity" / "reference" / "diagnose-after" / "last.zip"


def _run_for(backend: str):
    core = load_run_config(ROOT / "configs" / "experiments" / "core.toml", ROOT)
    return replace(core, runtime=RuntimeConfig(backend=backend))


def _segment_profile(transitions: int) -> dict:
    run = _run_for("rust")
    run = replace(run, algorithm=fit_algorithm(run.algorithm, timesteps=transitions, n_envs=4))
    scenarios = generate_manifest("train", 16)
    env = DummyVecEnv([make_env(scenarios, run, run.algorithm.seed + i) for i in range(4)])
    model = make_model(env, run.algorithm.seed, run.algorithm)
    PROFILE_RUNS.mkdir(parents=True, exist_ok=True)
    TIMERS.enabled = True
    TIMERS.reset()
    callback = DiagnoseCallback(PROFILE_RUNS)
    model.learn(total_timesteps=transitions, callback=callback)
    timers = {row["name"]: row for row in TIMERS.snapshot()}
    env.close()

    def seconds(name: str) -> float:
        return float(timers.get(name, {}).get("seconds", 0.0))

    events = {event["event"]: float(event["seconds"]) for event in callback.events}
    rollout_s = events.get("rollout_collect", 0.0)
    update_s = sum(event["seconds"] for event in callback.events if event["event"] == "ppo_update")
    env_step_s = seconds("env.step")
    mask_s = seconds("env.action_masks")
    observe_s = seconds("env.observe")
    return {
        "transitions": int(model.num_timesteps),
        "n_envs": run.algorithm.n_envs,
        "n_steps": run.algorithm.n_steps,
        "rollout_collect_s": rollout_s,
        "ppo_update_s": update_s,
        "env_step_s": env_step_s,
        "env_action_masks_s": mask_s,
        "env_observe_s": observe_s,
        "inference_and_buffer_s": max(0.0, rollout_s - env_step_s - mask_s),
        "env_share_of_rollout": env_step_s / rollout_s if rollout_s else 0.0,
        "update_share_of_learn": update_s / (rollout_s + update_s) if rollout_s else 0.0,
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


def _thread_sweep(threads: list[int], repetitions: int, transitions: int) -> dict:
    run = _run_for("rust")
    run = replace(run, algorithm=fit_algorithm(run.algorithm, timesteps=transitions, n_envs=4))
    scenarios = generate_manifest("train", 16)
    original = torch.get_num_threads()
    results = {}
    try:
        for value in threads:
            torch.set_num_threads(value)
            _learn_once(scenarios, run, transitions)  # warm-up
            times = [_learn_once(scenarios, run, transitions) for _ in range(repetitions)]
            results[str(value)] = {
                "wall_times_s": [round(t, 6) for t in times],
                "median_s": statistics.median(times),
                "min_s": min(times),
                "max_s": max(times),
                "transitions_per_s_median": transitions / statistics.median(times),
            }
            print(
                f"[profile] threads={value:2d} median={statistics.median(times):.3f}s "
                f"({transitions / statistics.median(times):.1f} transitions/s)",
                flush=True,
            )
    finally:
        torch.set_num_threads(original)
    best = min(results, key=lambda key: results[key]["median_s"])
    return {
        "transitions": transitions,
        "repetitions": repetitions,
        "results": results,
        "best": best,
    }


def _eval_profile(repetitions: int) -> dict:
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
    load_env = make_env_for_run(scenarios[:1], run)
    model, metadata = load_model(REFERENCE, load_env, run.physical)
    controller = make_controller("ppo", model=model, seed=metadata.get("seed", 11))
    env = make_env_for_run(scenarios, replace(run, runtime=RuntimeConfig(backend="rust")))
    TIMERS.enabled = True
    for _ in range(repetitions):
        TIMERS.reset()
        started = perf_counter()
        for index in range(len(scenarios)):
            rollout(env, controller, index)
        total = perf_counter() - started
    timers = {row["name"]: row for row in TIMERS.snapshot()}
    env_step = timers.get("env.step", {}).get("seconds", 0.0)
    infer = timers.get("eval.infer", {}).get("seconds", 0.0)
    summarize = timers.get("eval.summarize", {}).get("seconds", 0.0)
    mask = timers.get("env.action_masks", {}).get("seconds", 0.0)
    return {
        "days": len(scenarios),
        "repetitions": repetitions,
        "last_rep_wall_s": total,
        "env_step_s": env_step,
        "eval_infer_s": infer,
        "eval_summarize_s": summarize,
        "env_action_masks_s": mask,
        "overhead_s": max(0.0, total - env_step - infer - summarize - mask),
    }


_MEMORY_SNIPPET = """
import os
import sys
sys.path.insert(0, {src!r})
from bus_rl.config import ControlConfig
from bus_rl.data.scenario import generate_manifest
from bus_rl.backend.native import build_kernel
from bus_rl.env.native_bus_dispatch import NativeBusDispatchEnv


def rss_kb():
    page = os.sysconf("SC_PAGE_SIZE") // 1024
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * page


scenarios = generate_manifest("train", {count})
before = rss_kb()
if {shared}:
    keep = [NativeBusDispatchEnv(scenarios, scenarios[0].config, control=ControlConfig()) for _ in range({envs})]
else:
    keep = [[build_kernel(scenario) for _ in range({envs})] for scenario in scenarios]
after = rss_kb()
print(len(keep), after - before)
"""


def _measure_mode(count: int, envs: int, shared: bool) -> dict:
    snippet = _MEMORY_SNIPPET.format(src=str(ROOT / "src"), count=count, envs=envs, shared=shared)
    completed = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    _, delta = completed.stdout.strip().split()
    return {"mode": "shared" if shared else "per_env", "rss_delta_kb": int(delta)}


def _memory_check(scenario_count: int, env_count: int) -> dict:
    shared = _measure_mode(scenario_count, env_count, True)
    unshared = _measure_mode(scenario_count, env_count, False)
    return {
        "scenarios": scenario_count,
        "envs": env_count,
        "shared": shared,
        "per_env_copy": unshared,
        "saving_kb": unshared["rss_delta_kb"] - shared["rss_delta_kb"],
        "note": "per_env_copy rebuilds every scenario kernel per env (old path)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=4096)
    parser.add_argument("--threads", default="1,2,4,8,16")
    parser.add_argument("--thread-repetitions", type=int, default=3)
    parser.add_argument("--memory-scenarios", type=int, default=100)
    parser.add_argument("--memory-envs", type=int, default=4)
    parser.add_argument("--skip-segments", action="store_true")
    parser.add_argument("--skip-threads", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "profile-native.json")
    args = parser.parse_args()

    threads = [int(value) for value in args.threads.split(",") if value.strip()]
    started = perf_counter()
    payload: dict = {
        "profile_schema_version": 1,
        "role": "r4-optimization-profile",
        "protocol": {
            "transitions": args.transitions,
            "thread_repetitions": args.thread_repetitions,
            "timers_enabled_for_segments": True,
            "timers_disabled_for_thread_sweep": True,
        },
        "machine": {
            "platform": platform.platform(),
            "torch_threads_default": torch.get_num_threads(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }
    if not args.skip_segments:
        payload["segments"] = _segment_profile(args.transitions)
        print(f"[profile] segments {payload['segments']}", flush=True)
    if not args.skip_threads:
        payload["thread_sweep"] = _thread_sweep(threads, args.thread_repetitions, args.transitions)
    if not args.skip_eval:
        payload["eval"] = _eval_profile(repetitions=3)
        print(f"[profile] eval {payload['eval']}", flush=True)
    if not args.skip_memory:
        payload["memory"] = _memory_check(args.memory_scenarios, args.memory_envs)
        print(f"[profile] memory {payload['memory']}", flush=True)
    payload["wall_total_s"] = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[profile] wrote {args.output.relative_to(ROOT)} in {payload['wall_total_s']:.1f}s")


if __name__ == "__main__":
    main()
