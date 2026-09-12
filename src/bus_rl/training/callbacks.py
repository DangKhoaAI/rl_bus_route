"""Validation callback that keeps the lowest-cost checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from bus_rl.evaluation.pool import make_eval_pool
from bus_rl.evaluation.runner import mean_cost


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
        self.completed_episodes = 0
        self.history: list[dict] = []
        self._next_eval = run.algorithm.eval_freq
        self._pool = None

    def _subset(self):
        if self.eval_limit is None:
            return list(self.scenarios)
        return list(self.scenarios)[: self.eval_limit]

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

    def _evaluate(self) -> float:
        subset = self._subset()
        pool = None
        if self.run.runtime.reuse_eval_pool:
            if self._pool is None:
                self._pool = make_eval_pool(subset, self.run, self.forecaster)
            else:
                self._pool = self._pool.refresh(subset, self.run, self.forecaster)
            pool = self._pool
        # Strict less-than: a tie keeps the earlier checkpoint.
        cost = mean_cost(self.model, subset, self.run, self.forecaster, pool=pool)
        self.history.append({"timesteps": self.num_timesteps, "val_cost": cost})
        if cost < self.best:
            self.best = cost
            self.model.save(str(self.output / "best"))
        return cost

    def finalize(self) -> None:
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()
        self.model.save(str(self.output / "last"))
        if self.best == float("inf"):
            self.model.save(str(self.output / "best"))
        pd.DataFrame(self.history).to_csv(self.output / "evaluations.csv", index=False)
        if self._pool is not None:
            self._pool.close()
            self._pool = None
