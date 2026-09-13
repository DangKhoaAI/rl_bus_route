#!/usr/bin/env python
"""O0 2-thread Rust baseline and O1 eval-batch / pool-reuse ablation.

Speed numbers are taken with cProfile and TIMERS off. A separate profile run
only ranks bottlenecks; nested timer spans are never summed into a published
total. Historical rust-migration walls are not the candidate denominator.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl import domain
from bus_rl.config import ControlConfig, load_run_config
from bus_rl.data.scenario import generate_manifest
from bus_rl.domain import StepCosts
from bus_rl.env.factory import make_env_for_run
from bus_rl.evaluation.pool import EvalEnvPool
from bus_rl.evaluation.runner import PPOController, evaluate_scenarios, make_controller, rollout
from bus_rl.evaluation.summary import from_native_payload, summarize_inputs
from bus_rl.provenance import file_hash, git_status, lock_hash
from bus_rl.rewards.costs import RewardConfig, add_costs
from bus_rl.runtime import apply_runtime_settings, configure_torch_distributions
from bus_rl.timing import TIMERS
from bus_rl.training.checkpoint import load_metadata, load_model
from bus_rl.training.train import make_env, make_model

REPORT_DIR = ROOT / "reports" / "runtime-optimization"
REFERENCE = ROOT / "tests" / "backend_parity" / "reference" / "diagnose-after" / "last.zip"
CORE = ROOT / "configs" / "experiments" / "core-threads2.toml"
BATCH_SIZES = (1, 4, 8, 16, 32)


class _NoEval(BaseCallback):
    def _on_step(self) -> bool:
        return True


class _SegmentCallback(BaseCallback):
    """Rollout vs PPO-update walls without enabling TIMERS."""

    def __init__(self):
        super().__init__()
        self.rollout_s = 0.0
        self.update_s = 0.0
        self._rollout_started = None
        self._after_rollout = None

    def _on_rollout_start(self) -> None:
        now = perf_counter()
        if self._after_rollout is not None:
            self.update_s += now - self._after_rollout
        self._rollout_started = now

    def _on_rollout_end(self) -> None:
        now = perf_counter()
        self.rollout_s += now - (self._rollout_started or now)
        self._after_rollout = now

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        now = perf_counter()
        if self._after_rollout is not None:
            self.update_s += now - self._after_rollout


def _stats(times: list[float]) -> dict:
    return {
        "wall_times_s": [round(value, 6) for value in times],
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.fmean(times),
    }


def _rss_peak_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _rss_current_kb() -> int:
    page = os.sysconf("SC_PAGE_SIZE") // 1024
    with open("/proc/self/statm", encoding="utf-8") as handle:
        return int(handle.read().split()[1]) * page


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def rust_run():
    core = load_run_config(CORE, ROOT)
    # Component benchmarks measure the accepted opt-in config, which disables
    # Torch distribution validation; the on/off A/B is its own artifact.
    return replace(
        core,
        runtime=replace(core.runtime, backend="rust", validate_distributions=False),
    )


def _reference_run():
    run = rust_run()
    metadata = load_metadata(REFERENCE)
    reward = RewardConfig(
        **{
            key: metadata["reward"][key]
            for key in RewardConfig.__dataclass_fields__
            if key in metadata["reward"]
        }
    )
    return replace(run, control=ControlConfig(**metadata["control"]), reward=reward)


def collect_provenance(threads_actual: int) -> dict:
    sha, dirty = git_status(ROOT)
    try:
        porcelain = _run(["git", "status", "--porcelain"])
    except subprocess.CalledProcessError:
        porcelain = ""
    try:
        diff_stat = _run(["git", "diff", "--stat"])
    except subprocess.CalledProcessError:
        diff_stat = ""
    native_path = ROOT / "reports" / "rust-migration" / "native-build.json"
    native = json.loads(native_path.read_text()) if native_path.exists() else None
    library = ROOT / "src" / "bus_sim.so"
    return {
        "git": {
            "sha": sha,
            "dirty": dirty,
            "porcelain": porcelain,
            "diff_stat": diff_stat,
        },
        "native_build": native,
        "library_sha256": file_hash(library) if library.exists() else None,
        "lock_hash": lock_hash(ROOT),
        "uv_lock_sha256": file_hash(ROOT / "uv.lock") if (ROOT / "uv.lock").exists() else None,
        "config": str(CORE.relative_to(ROOT)),
        "checkpoint": str(REFERENCE.relative_to(ROOT)),
        "checkpoint_sha256": file_hash(REFERENCE),
        "seeds": {"algorithm": 11, "validation_split": 2001},
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": len(os.sched_getaffinity(0)),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "threads": {
            "torch_requested": 2,
            "torch_actual": threads_actual,
            "torch_interop": int(torch.get_num_interop_threads()),
            "omp": os.environ.get("OMP_NUM_THREADS"),
            "mkl": os.environ.get("MKL_NUM_THREADS"),
            "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "flags": {
            "timers": bool(TIMERS.enabled),
            "cprofile": False,
            "conservation_checks": bool(domain.CONSERVATION_CHECKS),
            "validate_observation": True,
            "validate_distributions": bool(
                torch.distributions.Distribution._validate_args
            ),
            "trace": False,
            "logging": "quiet",
        },
    }


def _load_model_and_scenarios(days: int):
    run = _reference_run()
    started = perf_counter()
    scenarios = generate_manifest("validation", 100)[:days]
    env = make_env_for_run(scenarios[:1], run)
    pack_s = perf_counter() - started
    model, metadata = load_model(REFERENCE, env, run.physical)
    return run, scenarios, model, metadata, pack_s, env


def measure_unprofiled(repetitions: int, transitions: int, eval_days: int) -> dict:
    assert TIMERS.enabled is False
    run = rust_run()
    apply_runtime_settings(run)
    train_scenarios = generate_manifest("train", 16)

    load_times = []
    for _ in range(repetitions):
        started = perf_counter()
        generate_manifest("validation", 100)
        from bus_rl.backend.native import shared_store

        shared_store(train_scenarios)
        load_times.append(perf_counter() - started)

    reset_times = []
    env = make_env_for_run(train_scenarios, run)
    for _ in range(repetitions):
        started = perf_counter()
        for index in range(min(16, len(train_scenarios))):
            env.reset(seed=0, options={"scenario_index": index})
        reset_times.append(perf_counter() - started)

    def _learn_once() -> tuple[float, float, float, float, int]:
        callback = _SegmentCallback()
        start = perf_counter()
        vec = DummyVecEnv(
            [
                make_env(train_scenarios, run, run.algorithm.seed + index)
                for index in range(run.algorithm.n_envs)
            ]
        )
        model = make_model(vec, run.algorithm.seed, run.algorithm)
        setup = perf_counter() - start
        started = perf_counter()
        model.learn(total_timesteps=transitions, callback=callback)
        learn = perf_counter() - started
        actual = int(model.num_timesteps)
        vec.close()
        return setup, learn, callback.rollout_s, callback.update_s, actual

    _learn_once()
    setups, learns, rollouts, updates, actuals = [], [], [], [], []
    for _ in range(repetitions):
        setup, learn, rollout_s, update_s, actual = _learn_once()
        setups.append(setup)
        learns.append(learn)
        rollouts.append(rollout_s)
        updates.append(update_s)
        actuals.append(actual)

    ckpt_run, eval_scenarios, model, _, load_pack_s, load_env = _load_model_and_scenarios(
        eval_days
    )
    controller = make_controller("ppo", model=model, seed=ckpt_run.algorithm.seed)
    eval_env = make_env_for_run(eval_scenarios, ckpt_run)

    def _eval_segments() -> dict:
        reset_s = mask_s = infer_s = step_s = summary_s = 0.0
        started = perf_counter()
        costs = []
        for index in range(len(eval_scenarios)):
            t0 = perf_counter()
            observation, _ = eval_env.reset(seed=0, options={"scenario_index": index})
            reset_s += perf_counter() - t0
            while True:
                t0 = perf_counter()
                mask = eval_env.action_masks()
                mask_s += perf_counter() - t0
                t0 = perf_counter()
                action = controller.act(observation, mask)
                infer_s += perf_counter() - t0
                t0 = perf_counter()
                observation, _reward, terminated, truncated, info = eval_env.step(action)
                step_s += perf_counter() - t0
                if terminated or truncated:
                    t0 = perf_counter()
                    metrics = eval_env.summary_inputs()
                    del metrics, info
                    summary_s += perf_counter() - t0
                    costs.append(True)
                    break
        return {
            "wall_s": perf_counter() - started,
            "reset_s": reset_s,
            "mask_s": mask_s,
            "inference_s": infer_s,
            "native_step_wrapper_s": step_s,
            "summary_s": summary_s,
            "days": len(eval_scenarios),
        }

    _eval_segments()
    eval_rows = [_eval_segments() for _ in range(repetitions)]

    ckpt_save = []
    ckpt_load = []
    tmp = ROOT / "runs" / "runtime-optimization" / "o0-ckpt"
    tmp.mkdir(parents=True, exist_ok=True)
    from sb3_contrib import MaskablePPO

    for _ in range(repetitions):
        started = perf_counter()
        model.save(str(tmp / "probe"))
        ckpt_save.append(perf_counter() - started)
        started = perf_counter()
        MaskablePPO.load(str(tmp / "probe.zip"), env=load_env, device="cpu")
        ckpt_load.append(perf_counter() - started)

    vec_times = []
    vec = DummyVecEnv(
        [make_env(train_scenarios, run, run.algorithm.seed + i) for i in range(run.algorithm.n_envs)]
    )
    vec.reset()
    for _ in range(repetitions):
        started = perf_counter()
        for _step in range(64):
            _obs, _r, _d, _i = vec.step(np.zeros(run.algorithm.n_envs, dtype=np.int64))
        vec_times.append(perf_counter() - started)
    vec.close()

    return {
        "protocol": {
            "repetitions": repetitions,
            "warmup": 1,
            "timers": "disabled",
            "profiler": "disabled",
            "conservation_checks": bool(domain.CONSERVATION_CHECKS),
            "transitions": transitions,
            "eval_days": eval_days,
            "note": "nested child spans are reported separately and not summed into total_wall_s",
        },
        "load_pack": _stats(load_times),
        "env_reset": _stats(reset_times),
        "isolated_learn": {
            **_stats(learns),
            "setup_median_s": statistics.median(setups),
            "rollout_collect_median_s": statistics.median(rollouts),
            "ppo_update_median_s": statistics.median(updates),
            "transitions_actual": actuals,
            "n_envs": run.algorithm.n_envs,
            "n_steps": run.algorithm.n_steps,
            "batch_size": run.algorithm.batch_size,
            "n_epochs": run.algorithm.n_epochs,
        },
        "validation": {
            "wall": _stats([row["wall_s"] for row in eval_rows]),
            "reset_median_s": statistics.median([row["reset_s"] for row in eval_rows]),
            "inference_median_s": statistics.median([row["inference_s"] for row in eval_rows]),
            "native_step_wrapper_median_s": statistics.median(
                [row["native_step_wrapper_s"] for row in eval_rows]
            ),
            "mask_median_s": statistics.median([row["mask_s"] for row in eval_rows]),
            "summary_median_s": statistics.median([row["summary_s"] for row in eval_rows]),
            "days": eval_days,
            "checkpoint_load_pack_s": load_pack_s,
            "raw": eval_rows,
        },
        "checkpoint_io": {
            "save": _stats(ckpt_save),
            "load": _stats(ckpt_load),
        },
        "wrapper_vecenv": {
            "dummy_vec_64_steps": _stats(vec_times),
            "n_envs": run.algorithm.n_envs,
        },
        "peak_rss_kb": _rss_peak_kb(),
        "current_rss_kb": _rss_current_kb(),
        "total_wall_note": "sum of component medians is not a published total",
    }


def measure_profile(transitions: int, eval_days: int) -> dict:
    """TIMERS-on ranking only; not a speed number."""
    TIMERS.enabled = True
    TIMERS.reset()
    run = rust_run()
    apply_runtime_settings(run)
    train_scenarios = generate_manifest("train", 16)
    vec = DummyVecEnv(
        [make_env(train_scenarios, run, run.algorithm.seed + i) for i in range(4)]
    )
    model = make_model(vec, run.algorithm.seed, run.algorithm)
    callback = _SegmentCallback()
    model.learn(total_timesteps=transitions, callback=callback)
    learn_timers = {row["name"]: row for row in TIMERS.snapshot()}
    vec.close()

    TIMERS.reset()
    ckpt_run, scenarios, eval_model, _, _, _ = _load_model_and_scenarios(eval_days)
    controller = make_controller("ppo", model=eval_model, seed=ckpt_run.algorithm.seed)
    env = make_env_for_run(scenarios, ckpt_run)
    started = perf_counter()
    for index in range(len(scenarios)):
        rollout(env, controller, index)
    eval_wall = perf_counter() - started
    eval_timers = {row["name"]: row for row in TIMERS.snapshot()}
    TIMERS.enabled = False
    TIMERS.reset()

    def seconds(table: dict, name: str) -> float:
        return float(table.get(name, {}).get("seconds", 0.0))

    ranking = []
    for phase, table in (("learn", learn_timers), ("eval", eval_timers)):
        for name, row in table.items():
            ranking.append(
                {
                    "phase": phase,
                    "name": name,
                    "seconds": float(row["seconds"]),
                    "calls": int(row["calls"]),
                }
            )
    ranking.sort(key=lambda item: item["seconds"], reverse=True)
    return {
        "role": "bottleneck-ranking-only",
        "timers_enabled": True,
        "not_a_speed_number": True,
        "learn": {
            "transitions": transitions,
            "rollout_collect_s": callback.rollout_s,
            "ppo_update_s": callback.update_s,
            "env_step_s": seconds(learn_timers, "env.step"),
            "env_observe_s": seconds(learn_timers, "env.observe"),
            "env_action_masks_s": seconds(learn_timers, "env.action_masks"),
            "nested_note": "env.observe is nested under env.step; do not add them",
        },
        "eval": {
            "days": eval_days,
            "wall_s_profiled": eval_wall,
            "eval_infer_s": seconds(eval_timers, "eval.infer"),
            "env_step_s": seconds(eval_timers, "env.step"),
            "eval_summarize_s": seconds(eval_timers, "eval.summarize"),
            "eval_action_mask_s": seconds(eval_timers, "eval.action_mask"),
        },
        "ranking": ranking[:20],
    }


def _eval_once(run, scenarios, model, *, pool=None) -> tuple[float, float, int]:
    started = perf_counter()
    frame, _ = evaluate_scenarios(
        scenarios,
        run,
        "ppo",
        model=model,
        model_seed=run.algorithm.seed,
        pool=pool,
    )
    elapsed = perf_counter() - started
    return elapsed, float(frame["total_cost"].mean()), len(frame)


def measure_ablation(repetitions: int, warmup: int, eval_days: int) -> dict:
    assert TIMERS.enabled is False
    run = _reference_run()
    apply_runtime_settings(run)
    scenarios = generate_manifest("validation", 100)[:eval_days]
    load_env = make_env_for_run(scenarios[:1], run)
    model, _ = load_model(REFERENCE, load_env, run.physical)

    variants: list[dict] = [{"name": "scalar", "batch_size": 1, "reuse_pool": False, "force_pool": False}]
    for size in BATCH_SIZES:
        variants.append(
            {"name": f"batch-{size}", "batch_size": size, "reuse_pool": False, "force_pool": True}
        )
        variants.append(
            {"name": f"pool-{size}", "batch_size": size, "reuse_pool": True, "force_pool": True}
        )

    pools: dict[str, EvalEnvPool] = {}
    results = {item["name"]: {"wall_s": [], "mean_cost": [], "days": [], "rss_kb": []} for item in variants}
    try:
        for item in variants:
            variant_run = replace(
                run,
                runtime=replace(
                    run.runtime,
                    eval_batch_size=item["batch_size"],
                    reuse_eval_pool=item["reuse_pool"],
                ),
            )
            pool = None
            if item["force_pool"]:
                pool = EvalEnvPool(scenarios, variant_run)
                pools[item["name"]] = pool
            for _ in range(warmup):
                _eval_once(variant_run, scenarios, model, pool=pool)
        for _rep in range(repetitions):
            for item in variants:
                variant_run = replace(
                    run,
                    runtime=replace(
                        run.runtime,
                        eval_batch_size=item["batch_size"],
                        reuse_eval_pool=item["reuse_pool"],
                    ),
                )
                pool = pools.get(item["name"])
                if item["force_pool"] and not item["reuse_pool"]:
                    if pool is not None:
                        pool.close()
                    pool = EvalEnvPool(scenarios, variant_run)
                    pools[item["name"]] = pool
                wall, cost, days = _eval_once(variant_run, scenarios, model, pool=pool)
                results[item["name"]]["wall_s"].append(wall)
                results[item["name"]]["mean_cost"].append(cost)
                results[item["name"]]["days"].append(days)
                results[item["name"]]["rss_kb"].append(_rss_current_kb())
                print(
                    f"[o1] {item['name']:10s} wall={wall:.3f}s cost={cost:.4f} days={days}",
                    flush=True,
                )
    finally:
        for pool in pools.values():
            pool.close()

    summary = {}
    for item in variants:
        row = results[item["name"]]
        stats = _stats(row["wall_s"])
        stats.update(
            {
                "batch_size": item["batch_size"],
                "reuse_pool": item["reuse_pool"],
                "force_batched_path": item["force_pool"],
                "mean_total_cost": float(np.mean(row["mean_cost"])),
                "days_actual": row["days"],
                "rss_kb": row["rss_kb"],
                "rss_median_kb": int(statistics.median(row["rss_kb"])),
            }
        )
        summary[item["name"]] = stats
    return {
        "protocol": {
            "repetitions": repetitions,
            "warmup": warmup,
            "eval_days": eval_days,
            "interleaved": True,
            "timers": "disabled",
            "profiler": "disabled",
            "batch_sizes": list(BATCH_SIZES),
        },
        "variants": summary,
        "peak_rss_kb": _rss_peak_kb(),
    }


def _batched_segments(pool: EvalEnvPool, controller, scenarios) -> dict:
    """O1 breakdown: segment walls of the batched evaluator with TIMERS off.

    Mirrors `evaluation/runner.py::_evaluate_batched` (same reset/step/predict
    calls, done slots skipped, summaries captured before slot reuse).
    """
    n_days = len(scenarios)
    batch_size = min(pool.batch_size, n_days)
    seg = {"reset_s": 0.0, "mask_s": 0.0, "infer_s": 0.0, "step_s": 0.0, "summary_s": 0.0}
    records: list[dict | None] = [None] * n_days
    started = perf_counter()
    offset = 0
    while offset < n_days:
        chunk = list(range(offset, min(offset + batch_size, n_days)))
        slots: list[dict] = []
        for slot_id, scenario_index in enumerate(chunk):
            env = pool.envs[slot_id]
            t0 = perf_counter()
            observation, _ = env.reset(seed=0, options={"scenario_index": scenario_index})
            seg["reset_s"] += perf_counter() - t0
            slots.append(
                {
                    "env": env,
                    "scenario_index": scenario_index,
                    "observation": observation,
                    "reward_sum": 0.0,
                    "costs": StepCosts(),
                    "done": False,
                }
            )
        while True:
            active = [slot for slot in slots if not slot["done"]]
            if not active:
                break
            t0 = perf_counter()
            masks = [slot["env"].action_masks() for slot in active]
            seg["mask_s"] += perf_counter() - t0
            t0 = perf_counter()
            actions = controller.act_batch([slot["observation"] for slot in active], masks)
            seg["infer_s"] += perf_counter() - t0
            for slot, action in zip(active, actions, strict=True):
                env = slot["env"]
                t0 = perf_counter()
                observation, reward, terminated, truncated, info = env.step(int(action))
                seg["step_s"] += perf_counter() - t0
                slot["observation"] = observation
                slot["reward_sum"] += float(reward)
                slot["costs"] = add_costs(slot["costs"], info["costs"])
                if not (terminated or truncated):
                    continue
                slot["done"] = True
                t0 = perf_counter()
                metrics = summarize_inputs(
                    env.summary_inputs(),
                    env.scenario,
                    slot["costs"],
                    slot["reward_sum"],
                    env.reward,
                )
                seg["summary_s"] += perf_counter() - t0
                records[slot["scenario_index"]] = metrics
        offset += len(chunk)
    seg["wall_s"] = perf_counter() - started
    if any(row is None for row in records):
        raise RuntimeError("breakdown missed a scenario")
    seg["days"] = n_days
    seg["batch_size"] = batch_size
    return seg


def measure_o1_breakdown(repetitions: int, warmup: int, eval_days: int, batch_sizes) -> dict:
    """Where the wall moved after O1 batching (TIMERS/profiler off)."""
    assert TIMERS.enabled is False
    run = _reference_run()
    apply_runtime_settings(run)
    scenarios = generate_manifest("validation", 100)[:eval_days]
    load_env = make_env_for_run(scenarios[:1], run)
    model, _ = load_model(REFERENCE, load_env, run.physical)
    controller = PPOController(model)

    results: dict[str, dict] = {}
    pools: dict[int, EvalEnvPool] = {}
    try:
        for size in batch_sizes:
            variant_run = replace(run, runtime=replace(run.runtime, eval_batch_size=size))
            pool = EvalEnvPool(scenarios, variant_run)
            pools[size] = pool
            for _ in range(warmup):
                _batched_segments(pool, controller, scenarios)
        for _ in range(repetitions):
            for size in batch_sizes:
                rows = _batched_segments(pools[size], controller, scenarios)
                rows["rss_current_kb"] = _rss_current_kb()
                results.setdefault(str(size), {"rows": []})["rows"].append(rows)
                print(
                    f"[o1-breakdown] batch={size:2d} wall={rows['wall_s']:.3f}s "
                    f"infer={rows['infer_s']:.3f}s step={rows['step_s']:.3f}s",
                    flush=True,
                )
    finally:
        for pool in pools.values():
            pool.close()

    summary = {}
    for size, payload in results.items():
        rows = payload["rows"]
        summary[size] = {
            key: statistics.median([row[key] for row in rows])
            for key in ("wall_s", "reset_s", "mask_s", "infer_s", "step_s", "summary_s")
        }
        summary[size]["effective_batch"] = rows[0]["batch_size"]
        summary[size]["inference_share"] = (
            summary[size]["infer_s"] / summary[size]["wall_s"] if summary[size]["wall_s"] else 0.0
        )
        summary[size]["rss_median_kb"] = int(
            statistics.median([row["rss_current_kb"] for row in rows])
        )
        summary[size]["raw"] = rows
    return {
        "protocol": {
            "repetitions": repetitions,
            "warmup": warmup,
            "eval_days": eval_days,
            "batch_sizes": list(batch_sizes),
            "interleaved": True,
            "timers": "disabled",
            "profiler": "disabled",
            "note": "segments use perf_counter; nested env.step/obs conversion is inside step_s",
        },
        "summary": summary,
        "peak_rss_kb": _rss_peak_kb(),
    }


def _native_segments(pool, controller, scenarios) -> dict:
    """O2 breakdown: native `step_batch` evaluator with TIMERS off.

    Mirrors `evaluation/runner.py::_evaluate_batched_native`.
    """
    from bus_rl.env.native_bus_dispatch import _step_costs

    n_days = len(scenarios)
    batch_size = min(pool.batch_size, n_days)
    seg = {
        "reset_s": 0.0,
        "mask_s": 0.0,
        "infer_s": 0.0,
        "step_s": 0.0,
        "summary_s": 0.0,
        "summary_export_s": 0.0,
        "summary_convert_s": 0.0,
        "summary_metrics_s": 0.0,
    }
    records: list[dict | None] = [None] * n_days
    started = perf_counter()
    offset = 0
    while offset < n_days:
        chunk = list(range(offset, min(offset + batch_size, n_days)))
        slot_ids = list(range(len(chunk)))
        t0 = perf_counter()
        observation, masks = pool.reset(slot_ids, chunk)
        seg["reset_s"] += perf_counter() - t0
        slots = [
            {
                "slot": slot,
                "scenario_index": scenario_index,
                "observation": {key: value[slot] for key, value in observation.items()},
                "reward_sum": 0.0,
                "costs": StepCosts(),
                "done": False,
            }
            for slot, scenario_index in zip(slot_ids, chunk, strict=True)
        ]
        while True:
            active = [slot for slot in slots if not slot["done"]]
            if not active:
                break
            active_slots = [slot["slot"] for slot in active]
            t0 = perf_counter()
            active_masks = [np.asarray(masks[slot["slot"]]) for slot in active]
            seg["mask_s"] += perf_counter() - t0
            t0 = perf_counter()
            actions = controller.act_batch(
                [slot["observation"] for slot in active], active_masks
            )
            seg["infer_s"] += perf_counter() - t0
            t0 = perf_counter()
            result = pool.step(active_slots, actions)
            seg["step_s"] += perf_counter() - t0
            step_rewards = np.asarray(result["reward"])
            step_terminated = np.asarray(result["terminated"])
            step_costs = np.asarray(result["costs"])
            masks = np.asarray(result["mask"])
            step_obs = result["obs"]
            for position, slot_state in enumerate(active):
                slot_state["observation"] = {
                    key: value[position] for key, value in step_obs.items()
                }
                slot_state["reward_sum"] += float(step_rewards[position])
                slot_state["costs"] = add_costs(
                    slot_state["costs"], _step_costs(step_costs[position])
                )
                if not bool(step_terminated[position]):
                    continue
                slot_state["done"] = True
                t0 = perf_counter()
                payload = pool.kernel.summary_inputs(slot_state["slot"])
                t1 = perf_counter()
                inputs = from_native_payload(payload)
                t2 = perf_counter()
                summarize_inputs(
                    inputs,
                    pool.applied[slot_state["scenario_index"]],
                    slot_state["costs"],
                    slot_state["reward_sum"],
                    pool.run.reward,
                )
                t3 = perf_counter()
                seg["summary_export_s"] += t1 - t0
                seg["summary_convert_s"] += t2 - t1
                seg["summary_metrics_s"] += t3 - t2
                seg["summary_s"] += t3 - t0
                records[slot_state["scenario_index"]] = {"done": True}
        offset += len(chunk)
    seg["wall_s"] = perf_counter() - started
    if any(row is None for row in records):
        raise RuntimeError("native breakdown missed a scenario")
    seg["days"] = n_days
    seg["batch_size"] = batch_size
    return seg


def _learn_segments(native_batch: bool, transitions: int, repetitions: int) -> dict:
    run = rust_run()
    apply_runtime_settings(run)
    train_scenarios = generate_manifest("train", 16)

    def _learn_once() -> tuple[dict, int]:
        callback = _SegmentCallback()
        if native_batch:
            from bus_rl.env.native_batch import NativeBatchVecEnv

            vec = NativeBatchVecEnv(train_scenarios, run, run.algorithm.seed)
        else:
            vec = DummyVecEnv(
                [
                    make_env(train_scenarios, run, run.algorithm.seed + index)
                    for index in range(run.algorithm.n_envs)
                ]
            )
        model = make_model(vec, run.algorithm.seed, run.algorithm)
        started = perf_counter()
        model.learn(total_timesteps=transitions, callback=callback)
        wall = perf_counter() - started
        actual = int(model.num_timesteps)
        vec.close()
        return {
            "wall_s": wall,
            "rollout_s": callback.rollout_s,
            "update_s": callback.update_s,
        }, actual

    _learn_once()
    rows = []
    actuals = []
    for _ in range(repetitions):
        row, actual = _learn_once()
        rows.append(row)
        actuals.append(actual)
    return {
        "native_batch": native_batch,
        "wall": _stats([row["wall_s"] for row in rows]),
        "rollout_median_s": statistics.median([row["rollout_s"] for row in rows]),
        "ppo_update_median_s": statistics.median([row["update_s"] for row in rows]),
        "transitions_actual": actuals,
        "raw": rows,
    }


def measure_o2_ablation(repetitions: int, warmup: int, eval_days: int) -> dict:
    """O1 batched (Python step) vs O2 native batch, plus training VecEnv."""
    assert TIMERS.enabled is False
    run = _reference_run()
    apply_runtime_settings(run)
    scenarios = generate_manifest("validation", 100)[:eval_days]
    load_env = make_env_for_run(scenarios[:1], run)
    model, _ = load_model(REFERENCE, load_env, run.physical)
    controller = PPOController(model)

    from bus_rl.env.native_batch import NativeBatchEvalPool

    variants = []
    for size in (8, 16, 32):
        variants.append(("o1-batch", size))
        variants.append(("o2-native", size))

    pools: dict[tuple, object] = {}
    for kind, size in variants:
        variant_run = replace(
            run,
            runtime=replace(
                run.runtime,
                eval_batch_size=size,
                native_batch=(kind == "o2-native"),
            ),
        )
        pools[(kind, size)] = (
            NativeBatchEvalPool(scenarios, variant_run)
            if kind == "o2-native"
            else EvalEnvPool(scenarios, variant_run)
        )
    for key, pool in pools.items():
        for _ in range(warmup):
            if key[0] == "o2-native":
                _native_segments(pool, controller, scenarios)
            else:
                _batched_segments(pool, controller, scenarios)

    results: dict[str, dict] = {}
    try:
        for _ in range(repetitions):
            for key, pool in pools.items():
                name = f"{key[0]}-{key[1]}"
                rows = (
                    _native_segments(pool, controller, scenarios)
                    if key[0] == "o2-native"
                    else _batched_segments(pool, controller, scenarios)
                )
                rows["rss_current_kb"] = _rss_current_kb()
                results.setdefault(name, {"rows": []})["rows"].append(rows)
                print(
                    f"[o2] {name:12s} wall={rows['wall_s']:.3f}s "
                    f"infer={rows['infer_s']:.3f}s step={rows['step_s']:.3f}s "
                    f"summ={rows['summary_s']:.3f}s",
                    flush=True,
                )
    finally:
        for pool in pools.values():
            pool.close()

    summary = {}
    for name, payload in results.items():
        rows = payload["rows"]
        entry = {}
        for field in (
            "wall_s",
            "reset_s",
            "mask_s",
            "infer_s",
            "step_s",
            "summary_s",
            "summary_export_s",
            "summary_convert_s",
            "summary_metrics_s",
        ):
            values = [row[field] for row in rows if field in row]
            if values:
                entry[field] = statistics.median(values)
        entry["effective_batch"] = rows[0]["batch_size"]
        entry["rss_median_kb"] = int(statistics.median([row["rss_current_kb"] for row in rows]))
        entry["raw"] = rows
        summary[name] = entry

    training = {
        "dummy": _learn_segments(False, 12_288, repetitions),
        "native": _learn_segments(True, 12_288, repetitions),
    }
    return {
        "protocol": {
            "repetitions": repetitions,
            "warmup": warmup,
            "eval_days": eval_days,
            "eval_batch_sizes": [8, 16, 32],
            "transitions": 12_288,
            "timers": "disabled",
            "profiler": "disabled",
        },
        "eval": summary,
        "training": training,
        "peak_rss_kb": _rss_peak_kb(),
    }


def measure_ppo_update_profile(repetitions: int) -> dict:
    """Forward / backward / optimizer / clip split of one PPO update (explained).

    Timer overhead is present, so this ranks sub-steps and is not a speed
    number. One update is 1024 transitions (4 envs x n_steps=256, 4 epochs).
    """
    run = rust_run()
    apply_runtime_settings(run)
    train_scenarios = generate_manifest("train", 16)

    def _profile_once(native_batch: bool) -> dict:
        if native_batch:
            from bus_rl.env.native_batch import NativeBatchVecEnv

            vec = NativeBatchVecEnv(train_scenarios, run, run.algorithm.seed)
        else:
            vec = DummyVecEnv(
                [
                    make_env(train_scenarios, run, run.algorithm.seed + index)
                    for index in range(run.algorithm.n_envs)
                ]
            )
        model = make_model(vec, run.algorithm.seed, run.algorithm)
        algorithm = run.algorithm
        model.learn(total_timesteps=algorithm.n_steps * algorithm.n_envs)
        stats = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0, "clip": 0.0}
        counts = {key: 0 for key in stats}
        original_eval = model.policy.evaluate_actions
        original_backward = torch.Tensor.backward
        original_step = model.policy.optimizer.step
        original_clip = torch.nn.utils.clip_grad_norm_

        def timed(name, function):
            def wrapper(*args, **kwargs):
                started = perf_counter()
                result = function(*args, **kwargs)
                stats[name] += perf_counter() - started
                counts[name] += 1
                return result

            return wrapper

        model.policy.evaluate_actions = timed("forward", original_eval)
        torch.Tensor.backward = timed("backward", original_backward)
        model.policy.optimizer.step = timed("optimizer", original_step)
        torch.nn.utils.clip_grad_norm_ = timed("clip", original_clip)
        started = perf_counter()
        model.train()
        total = perf_counter() - started
        model.policy.evaluate_actions = original_eval
        torch.Tensor.backward = original_backward
        model.policy.optimizer.step = original_step
        torch.nn.utils.clip_grad_norm_ = original_clip
        vec.close()
        accounted = sum(stats.values())
        return {
            "total_s": total,
            **stats,
            "other_s": total - accounted,
            "calls": counts,
        }

    _profile_once(False)
    rows = {"dummy": [], "native": []}
    for _ in range(repetitions):
        rows["dummy"].append(_profile_once(False))
        rows["native"].append(_profile_once(True))
    summary = {}
    for kind, payloads in rows.items():
        summary[kind] = {
            field: statistics.median([row[field] for row in payloads])
            for field in ("total_s", "forward", "backward", "optimizer", "clip", "other_s")
        }
        summary[kind]["calls"] = payloads[0]["calls"]
        summary[kind]["raw"] = payloads
    return {
        "role": "explanatory-only",
        "not_a_speed_number": True,
        "transitions_per_update": run.algorithm.n_steps * run.algorithm.n_envs,
        "n_epochs": run.algorithm.n_epochs,
        "batch_size": run.algorithm.batch_size,
        "summary": summary,
    }


def measure_rollout_profile(repetitions: int) -> dict:
    """Sub-step split of one rollout (n_steps x n_envs) for dummy vs native (O2).

    Wraps `collect_rollouts` collaborators: policy forward (features/distribution/
    sampling), mask retrieval, tensor conversion, native/env step, buffer add and
    GAE. Timer overhead is present, so this ranks sub-steps and is not a speed
    number.
    """
    from sb3_contrib.ppo_mask import ppo_mask

    run = rust_run()
    apply_runtime_settings(run)
    train_scenarios = generate_manifest("train", 16)
    names = (
        "obs_to_tensor",
        "action_masks",
        "features",
        "distribution",
        "sample",
        "policy_forward",
        "env_step",
        "buffer_add",
        "gae",
    )

    def _profile_once(native_batch: bool) -> dict:
        if native_batch:
            from bus_rl.env.native_batch import NativeBatchVecEnv

            vec = NativeBatchVecEnv(train_scenarios, run, run.algorithm.seed)
        else:
            vec = DummyVecEnv(
                [
                    make_env(train_scenarios, run, run.algorithm.seed + index)
                    for index in range(run.algorithm.n_envs)
                ]
            )
        model = make_model(vec, run.algorithm.seed, run.algorithm)
        stats = {name: 0.0 for name in names}
        counts = {name: 0 for name in names}
        restores: list = []

        def wrap(target, attr, name):
            original = getattr(target, attr)
            exposed = attr in getattr(target, "__dict__", {})

            def wrapper(*args, **kwargs):
                started = perf_counter()
                result = original(*args, **kwargs)
                stats[name] += perf_counter() - started
                counts[name] += 1
                return result

            setattr(target, attr, wrapper)

            def restore():
                if exposed:
                    setattr(target, attr, original)
                else:
                    delattr(target, attr)

            restores.append(restore)

        wrap(ppo_mask, "obs_as_tensor", "obs_to_tensor")
        wrap(ppo_mask, "get_action_masks", "action_masks")
        wrap(model.policy, "forward", "policy_forward")
        wrap(model.policy, "extract_features", "features")
        wrap(model.policy, "_get_action_dist_from_latent", "distribution")
        wrap(type(model.policy.action_dist), "get_actions", "sample")
        wrap(vec, "step_wait", "env_step")
        wrap(model.rollout_buffer, "add", "buffer_add")
        wrap(model.rollout_buffer, "compute_returns_and_advantage", "gae")

        callback = _SegmentCallback()
        try:
            model.learn(
                total_timesteps=run.algorithm.n_steps * run.algorithm.n_envs,
                callback=callback,
            )
        finally:
            for restore in reversed(restores):
                restore()
            vec.close()
        policy_forward = stats["policy_forward"]
        forward_parts = stats["features"] + stats["distribution"] + stats["sample"]
        rollout = callback.rollout_s
        accounted = (
            policy_forward
            + stats["obs_to_tensor"]
            + stats["action_masks"]
            + stats["env_step"]
            + stats["buffer_add"]
            + stats["gae"]
        )
        return {
            "rollout_s": rollout,
            "update_s": callback.update_s,
            **stats,
            "forward_other_s": policy_forward - forward_parts,
            "rollout_other_s": rollout - accounted,
            "counts": counts,
        }

    _profile_once(False)
    rows = {"dummy": [], "native": []}
    for _ in range(repetitions):
        rows["dummy"].append(_profile_once(False))
        rows["native"].append(_profile_once(True))
    scalar_fields = ("rollout_s", "update_s", *names, "forward_other_s", "rollout_other_s")
    summary = {}
    for kind, payloads in rows.items():
        summary[kind] = {
            field: statistics.median([row[field] for row in payloads]) for field in scalar_fields
        }
        summary[kind]["counts"] = payloads[0]["counts"]
        summary[kind]["raw"] = payloads
    return {
        "role": "explanatory-only",
        "not_a_speed_number": True,
        "transitions_per_rollout": run.algorithm.n_steps * run.algorithm.n_envs,
        "n_envs": run.algorithm.n_envs,
        "n_steps": run.algorithm.n_steps,
        "note": "features/distribution/sample are nested under policy_forward",
        "summary": summary,
    }


def measure_distribution_validation(repetitions: int) -> dict:
    """Controlled A/B for Torch distribution validation on the policy forward.

    Same process, same model, same observations/masks; only the validation
    setting changes. Also runs one rollout per setting and checks that sampled
    actions/values/log-probs and deterministic argmax are bit-identical.
    """
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.utils import obs_as_tensor

    run = rust_run()
    apply_runtime_settings(run)
    train_scenarios = generate_manifest("train", 16)
    vec = DummyVecEnv(
        [
            make_env(train_scenarios, run, run.algorithm.seed + index)
            for index in range(run.algorithm.n_envs)
        ]
    )
    model = make_model(vec, run.algorithm.seed, run.algorithm)
    model.policy.set_training_mode(False)
    observation = vec.reset()
    masks = get_action_masks(vec)
    obs_tensor = obs_as_tensor(observation, "cpu")
    calls = 2000

    def forward_wall(validate: bool) -> float:
        configure_torch_distributions(validate)
        with torch.no_grad():
            for _ in range(50):
                model.policy(obs_tensor, action_masks=masks)
            started = perf_counter()
            for _ in range(calls):
                model.policy(obs_tensor, action_masks=masks)
            return perf_counter() - started

    def sample(validate: bool):
        configure_torch_distributions(validate)
        torch.manual_seed(123)
        with torch.no_grad():
            actions, values, log_prob = model.policy(obs_tensor, action_masks=masks)
            argmax = model.policy.get_distribution(
                obs_tensor, action_masks=masks
            ).get_actions(deterministic=True)
        return actions, values, log_prob, argmax

    def rollout_once(validate: bool) -> dict:
        configure_torch_distributions(validate)
        callback = _SegmentCallback()
        env = DummyVecEnv(
            [
                make_env(train_scenarios, run, run.algorithm.seed + index)
                for index in range(run.algorithm.n_envs)
            ]
        )
        probe = make_model(env, run.algorithm.seed, run.algorithm)
        started = perf_counter()
        probe.learn(
            total_timesteps=run.algorithm.n_steps * run.algorithm.n_envs, callback=callback
        )
        wall = perf_counter() - started
        env.close()
        return {
            "wall_s": wall,
            "rollout_s": callback.rollout_s,
            "update_s": callback.update_s,
        }

    sample(True)
    forward_wall(True)
    forward_times = {"on": [], "off": []}
    for _ in range(max(1, repetitions)):
        forward_times["on"].append(forward_wall(True) / calls * 1e6)
        forward_times["off"].append(forward_wall(False) / calls * 1e6)

    actions_on, values_on, logprob_on, argmax_on = sample(True)
    actions_off, values_off, logprob_off, argmax_off = sample(False)
    bit_identical = {
        "actions": bool(torch.equal(actions_on, actions_off)),
        "values": bool(torch.equal(values_on, values_off)),
        "log_prob": bool(torch.equal(logprob_on, logprob_off)),
        "argmax": bool(torch.equal(argmax_on, argmax_off)),
    }

    rollout = {"on": [], "off": []}
    for _ in range(max(1, repetitions)):
        rollout["on"].append(rollout_once(True))
        rollout["off"].append(rollout_once(False))
    configure_torch_distributions(True)
    vec.close()

    on_us = statistics.median(forward_times["on"])
    off_us = statistics.median(forward_times["off"])
    on_roll = statistics.median(row["rollout_s"] for row in rollout["on"])
    off_roll = statistics.median(row["rollout_s"] for row in rollout["off"])
    return {
        "batch": run.algorithm.n_envs,
        "forward_calls_per_measure": calls,
        "forward_us": {"on": on_us, "off": off_us, "speedup": on_us / off_us},
        "bit_identical": bit_identical,
        "rollout_s": {
            "on": on_roll,
            "off": off_roll,
            "speedup": on_roll / off_roll,
            "reduction": 1.0 - off_roll / on_roll,
        },
        "raw": {"forward_us": forward_times, "rollout": rollout},
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[runtime] wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "o0-unprofiled",
            "o0-profile",
            "o1-ablation",
            "o1-breakdown",
            "o2-ablation",
            "ppo-update-profile",
            "rollout-profile",
            "distribution-validation",
        ),
        required=True,
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--transitions", type=int, default=12_288)
    parser.add_argument("--eval-days", type=int, default=100, dest="eval_days")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    domain.CONSERVATION_CHECKS = False
    TIMERS.enabled = False
    TIMERS.reset()
    run = rust_run()
    threads = apply_runtime_settings(run)["torch_threads_actual"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "o0-unprofiled":
        payload = {
            "role": "o0-unprofiled",
            "provenance": collect_provenance(threads),
            "timings": measure_unprofiled(args.repetitions, args.transitions, args.eval_days),
        }
        output = args.output or REPORT_DIR / "o0-unprofiled.json"
        _write(output, payload)
        return
    if args.mode == "o0-profile":
        payload = {
            "role": "o0-profile",
            "provenance": collect_provenance(threads),
            "profile": measure_profile(min(args.transitions, 4096), min(args.eval_days, 10)),
        }
        payload["provenance"]["flags"]["timers"] = True
        payload["provenance"]["flags"]["note"] = "profile ranking only; not a speed number"
        output = args.output or REPORT_DIR / "o0-profile.json"
        _write(output, payload)
        return
    if args.mode == "o1-breakdown":
        payload = {
            "role": "o1-breakdown",
            "provenance": collect_provenance(threads),
            "breakdown": measure_o1_breakdown(
                args.repetitions, args.warmup, args.eval_days, BATCH_SIZES
            ),
        }
        output = args.output or REPORT_DIR / "o1-breakdown.json"
        _write(output, payload)
        return
    if args.mode == "distribution-validation":
        payload = {
            "role": "distribution-validation-ab",
            "provenance": collect_provenance(threads),
            "ab": measure_distribution_validation(args.repetitions),
        }
        output = args.output or REPORT_DIR / "distribution-validation.json"
        _write(output, payload)
        return
    if args.mode == "rollout-profile":
        payload = {
            "role": "rollout-profile",
            "provenance": collect_provenance(threads),
            "profile": measure_rollout_profile(args.repetitions),
        }
        payload["provenance"]["flags"]["timers"] = True
        payload["provenance"]["flags"]["note"] = "explanatory; sub-step timers on"
        output = args.output or REPORT_DIR / "rollout-profile.json"
        _write(output, payload)
        return
    if args.mode == "ppo-update-profile":
        payload = {
            "role": "ppo-update-profile",
            "provenance": collect_provenance(threads),
            "profile": measure_ppo_update_profile(args.repetitions),
        }
        payload["provenance"]["flags"]["timers"] = True
        payload["provenance"]["flags"]["note"] = "explanatory; sub-step timers on"
        output = args.output or REPORT_DIR / "ppo-update-profile.json"
        _write(output, payload)
        return
    if args.mode == "o2-ablation":
        payload = {
            "role": "o2-ablation",
            "provenance": collect_provenance(threads),
            "ablation": measure_o2_ablation(args.repetitions, args.warmup, args.eval_days),
        }
        output = args.output or REPORT_DIR / "o2-ablation.json"
        _write(output, payload)
        return
    payload = {
        "role": "o1-ablation",
        "provenance": collect_provenance(threads),
        "ablation": measure_ablation(args.repetitions, args.warmup, args.eval_days),
    }
    output = args.output or REPORT_DIR / "o1-ablation.json"
    _write(output, payload)


if __name__ == "__main__":
    main()
