"""Validation callback that keeps the lowest-cost checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from bus_rl.evaluation.pool import make_eval_pool
from bus_rl.evaluation.runner import evaluate_scenarios


class BestValidationCallback(BaseCallback):
    def __init__(
        self, scenarios, run, output: Path, forecaster=None, eval_limit: int | None = None
    ):
        super().__init__()
        self.scenarios = list(scenarios)
        self.run = run
        self.output = Path(output)
        self.forecaster = forecaster
        self.eval_limit = eval_limit
        self.best = float("inf")
        self.best_timesteps = None
        self.completed_episodes = 0
        self.history: list[dict] = []
        self.validation_rows: list[dict] = []
        self._next_eval = run.algorithm.eval_freq
        self._pool = None
        self._started = 0.0

    def _subset(self):
        if self.eval_limit is None:
            return list(self.scenarios)
        return list(self.scenarios)[: self.eval_limit]

    def _on_training_start(self) -> None:
        from time import perf_counter

        self._started = perf_counter()
        # Measure the actual untrained policy instead of extrapolating a step-zero score.
        self._evaluate(select_checkpoint=False)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.completed_episodes += int(np.sum(dones))
        if self.run.algorithm.eval_freq <= 0:
            return True
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
            self._next_eval += self.run.algorithm.eval_freq
        return True

    def _evaluate(self, *, select_checkpoint: bool = True) -> float:
        subset = self._subset()
        pool = None
        if self.run.runtime.reuse_eval_pool:
            if self._pool is None:
                self._pool = make_eval_pool(subset, self.run, self.forecaster)
            else:
                self._pool = self._pool.refresh(subset, self.run, self.forecaster)
            pool = self._pool
        from time import perf_counter

        started = perf_counter()
        frame, _ = evaluate_scenarios(
            subset,
            self.run,
            "ppo",
            model=self.model,
            model_seed=self.run.algorithm.seed,
            split="validation",
            forecaster=self.forecaster,
            pool=pool,
        )
        elapsed = perf_counter() - started
        cost = float(frame["total_cost_core"].mean())
        is_best = select_checkpoint and cost < self.best
        self.history.append(
            {
                "timesteps": int(self.num_timesteps),
                "seed": self.run.algorithm.seed,
                "encoder": self.run.algorithm.encoder,
                "history_minutes": self.run.algorithm.history_minutes,
                "val_cost": cost,
                "val_days": len(frame),
                "elapsed_wall_s": elapsed,
                "train_elapsed_wall_s": perf_counter() - self._started,
                "completed_episodes": self.completed_episodes,
                "best_checkpoint": bool(is_best),
                "checkpoint_eligible": select_checkpoint,
            }
        )
        for row in frame.to_dict(orient="records"):
            row.update(
                {
                    "validation_timestep": int(self.num_timesteps),
                    "validation_elapsed_wall_s": elapsed,
                    "completed_episodes": self.completed_episodes,
                }
            )
            self.validation_rows.append(row)
        # Strict less-than: a tie keeps the earlier checkpoint.
        if is_best:
            self.best = cost
            self.best_timesteps = int(self.num_timesteps)
            self.model.save(str(self.output / "best"))
        return cost

    def finalize(self) -> None:
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()
        self.model.save(str(self.output / "last"))
        if self.best == float("inf"):
            self.model.save(str(self.output / "best"))
        pd.DataFrame(self.history).to_csv(self.output / "evaluations.csv", index=False)
        pd.DataFrame(self.validation_rows).to_csv(
            self.output / "validation_metrics.csv", index=False
        )
        if self._pool is not None:
            self._pool.close()
            self._pool = None
