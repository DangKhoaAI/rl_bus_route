"""Training diagnosis: per-function timers, PPO collect vs update, eval cost."""

from __future__ import annotations

import cProfile
import json
import pstats
from io import StringIO
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl.config import RunConfig
from bus_rl.evaluation.runner import mean_cost
from bus_rl.runtime import apply_runtime_settings
from bus_rl.timing import TIMERS
from bus_rl.training.checkpoint import run_metadata, write_metadata
from bus_rl.training.train import make_env, make_model


class DiagnoseCallback(BaseCallback):
    """Logs rollout collection vs PPO update vs validation wall time."""

    def __init__(self, output: Path):
        super().__init__()
        self.output = Path(output)
        self.events: list[dict] = []
        self._rollout_started: float | None = None
        self._after_rollout: float | None = None
        self._learn_started: float | None = None
        self._last_env_snapshot: list[dict] = []

    def _on_training_start(self) -> None:
        TIMERS.enabled = True
        TIMERS.reset()
        self._learn_started = perf_counter()
        self._log("learn_start", 0.0)

    def _on_rollout_start(self) -> None:
        now = perf_counter()
        if self._after_rollout is not None:
            self._log("ppo_update", now - self._after_rollout)
        self._last_env_snapshot = TIMERS.snapshot()
        self._rollout_started = now

    def _on_rollout_end(self) -> None:
        now = perf_counter()
        elapsed = now - (self._rollout_started or now)
        current = TIMERS.snapshot()
        delta = _delta(current, self._last_env_snapshot)
        self._log("rollout_collect", elapsed, extra={"env": delta[:12]})
        self._after_rollout = now

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        now = perf_counter()
        if self._after_rollout is not None:
            self._log("ppo_update", now - self._after_rollout)
        self._log("learn_total", now - (self._learn_started or now))
        (self.output / "train_events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in self.events) + "\n"
        )

    def _log(self, event: str, seconds: float, extra: dict | None = None) -> None:
        row = {
            "event": event,
            "seconds": seconds,
            "timesteps": int(getattr(self, "num_timesteps", 0) or 0),
        }
        if extra:
            row.update(extra)
        self.events.append(row)
        print(
            f"[diagnose] {event:16s} {seconds:8.3f}s  t={row['timesteps']}",
            flush=True,
        )


def _delta(current: list[dict], previous: list[dict]) -> list[dict]:
    before = {row["name"]: row for row in previous}
    rows = []
    for row in current:
        old = before.get(row["name"], {"seconds": 0.0, "calls": 0})
        seconds = row["seconds"] - old["seconds"]
        calls = row["calls"] - old["calls"]
        if seconds <= 0:
            continue
        rows.append(
            {
                "name": row["name"],
                "seconds": seconds,
                "calls": calls,
                "us_per_call": (seconds / calls) * 1e6 if calls else 0.0,
            }
        )
    rows.sort(key=lambda item: item["seconds"], reverse=True)
    return rows


def _cprofile_top(profiler: cProfile.Profile, limit: int = 30) -> list[dict]:
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
    stats.print_stats(limit)
    text = stream.getvalue()
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "ncalls": parts[0],
                    "tottime": float(parts[1]),
                    "percall": float(parts[2]),
                    "cumtime": float(parts[3]),
                    "function": " ".join(parts[5:]),
                }
            )
        except ValueError:
            continue
        if len(rows) >= limit:
            break
    return rows, text


def diagnose_train(
    run: RunConfig,
    train_scenarios,
    val_scenarios,
    output: Path,
    *,
    eval_limit: int = 10,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    apply_runtime_settings(run)
    TIMERS.enabled = True
    TIMERS.reset()

    loaded_at = perf_counter()
    env = DummyVecEnv(
        [
            make_env(train_scenarios, run, run.algorithm.seed + index)
            for index in range(run.algorithm.n_envs)
        ]
    )
    model = make_model(env, run.algorithm.seed, run.algorithm)
    setup_s = perf_counter() - loaded_at
    print(f"[diagnose] env_model_setup {setup_s:.3f}s  n_envs={run.algorithm.n_envs}", flush=True)

    diagnose = DiagnoseCallback(output)
    profiler = cProfile.Profile()
    started = perf_counter()
    profiler.enable()
    model.learn(total_timesteps=run.algorithm.total_timesteps, callback=CallbackList([diagnose]))
    profiler.disable()
    learn_s = perf_counter() - started

    TIMERS.reset()
    eval_started = perf_counter()
    print(f"[diagnose] validation start  days={eval_limit}", flush=True)
    val_cost = mean_cost(model, val_scenarios, run, limit=eval_limit)
    eval_s = perf_counter() - eval_started
    eval_timers = TIMERS.snapshot()
    print(f"[diagnose] validation       {eval_s:.3f}s  mean_cost={val_cost:.1f}", flush=True)

    model.save(str(output / "last"))
    profile_rows, profile_text = _cprofile_top(profiler)
    (output / "cprofile.txt").write_text(profile_text)
    summary = {
        "setup_s": setup_s,
        "learn_s": learn_s,
        "torch_threads": torch.get_num_threads(),
        "eval_s": eval_s,
        "eval_days": eval_limit,
        "eval_mean_cost": val_cost,
        "timesteps": int(model.num_timesteps),
        "n_envs": run.algorithm.n_envs,
        "n_steps": run.algorithm.n_steps,
        "n_epochs": run.algorithm.n_epochs,
        "decisions_per_s": model.num_timesteps / learn_s if learn_s else 0.0,
        "events": diagnose.events,
        "learn_functions": _sum_event(diagnose.events, "rollout_collect", "ppo_update"),
        "eval_functions": eval_timers,
        "cprofile_top": profile_rows[:20],
        "projected": _project(learn_s, eval_s, model.num_timesteps, eval_limit),
    }
    (output / "diagnosis.json").write_text(json.dumps(summary, indent=2))
    write_metadata(
        output,
        run_metadata(
            run,
            {
                "mode": "diagnose",
                "wall_time_s": setup_s + learn_s + eval_s,
                "total_timesteps_actual": int(model.num_timesteps),
            },
        ),
    )
    return summary


def _sum_event(events: list[dict], *names: str) -> dict[str, float]:
    totals = {name: 0.0 for name in names}
    for event in events:
        if event["event"] in totals:
            totals[event["event"]] += float(event["seconds"])
    return totals


def _project(learn_s: float, eval_s: float, timesteps: int, eval_days: int) -> dict:
    full = 245_760
    scale = full / timesteps if timesteps else 0.0
    evals = full / 12_288
    return {
        "full_timesteps": full,
        "train_only_s": learn_s * scale,
        "val_callbacks_s": (eval_s / max(eval_days, 1)) * 10 * evals,
        "note": "train_only scales linearly; val_callbacks assumes 10-day eval every 12,288 steps",
    }
