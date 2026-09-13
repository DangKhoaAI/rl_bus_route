from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from bus_rl.config import AlgorithmConfig, RunConfig
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.runtime import apply_runtime_settings
from bus_rl.learning.policy import POLICY_KWARGS
from bus_rl.learning.training.callbacks import BestValidationCallback
from bus_rl.learning.training.checkpoint import run_metadata, write_metadata


class ProgressCallback(BaseCallback):
    """One line per PPO rollout so long runs are diagnosable without cProfile."""

    def __init__(self):
        super().__init__()
        self._started = 0.0
        self._last_timesteps = 0

    def _on_training_start(self) -> None:
        self._started = perf_counter()
        self._last_timesteps = 0
        print("[train] start", flush=True)

    def _on_rollout_end(self) -> bool:
        elapsed = perf_counter() - self._started
        delta = int(self.num_timesteps - self._last_timesteps)
        fps = self.num_timesteps / elapsed if elapsed else 0.0
        print(
            f"[train] timesteps={self.num_timesteps}  +{delta}  "
            f"elapsed={elapsed:.1f}s  fps={fps:.1f}",
            flush=True,
        )
        self._last_timesteps = int(self.num_timesteps)
        return True

    def _on_step(self) -> bool:
        return True


def make_model(env, seed: int, algorithm: AlgorithmConfig | None = None):
    algorithm = algorithm or AlgorithmConfig()
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        gamma=algorithm.gamma,
        learning_rate=algorithm.learning_rate,
        n_steps=algorithm.n_steps,
        batch_size=algorithm.batch_size,
        n_epochs=algorithm.n_epochs,
        gae_lambda=algorithm.gae_lambda,
        clip_range=algorithm.clip_range,
        ent_coef=algorithm.ent_coef,
        vf_coef=algorithm.vf_coef,
        max_grad_norm=algorithm.max_grad_norm,
        seed=seed,
        device=algorithm.device,
        policy_kwargs=POLICY_KWARGS,
        verbose=0,
    )


def make_env(scenarios, run: RunConfig, seed: int, forecaster=None):
    def _init():
        env = make_env_for_run(scenarios, run, forecaster=forecaster)
        env.reset(seed=seed)
        return env

    return _init


def fit_algorithm(
    algorithm: AlgorithmConfig,
    *,
    timesteps: int | None = None,
    n_envs: int | None = None,
    seed: int | None = None,
) -> AlgorithmConfig:
    updates: dict = {}
    if seed is not None:
        updates["seed"] = seed
    if n_envs is not None:
        updates["n_envs"] = n_envs
    if timesteps is not None:
        updates["total_timesteps"] = timesteps
        n_envs_final = updates.get("n_envs", algorithm.n_envs)
        if timesteps < algorithm.n_steps * n_envs_final:
            n_envs_final = n_envs if n_envs is not None else 1
            n_steps = max(2, timesteps // n_envs_final)
            updates["n_envs"] = n_envs_final
            updates["n_steps"] = n_steps
            updates["batch_size"] = min(algorithm.batch_size, n_steps * n_envs_final)
            updates["eval_freq"] = timesteps
            updates["n_epochs"] = min(algorithm.n_epochs, 2)
    return replace(algorithm, **updates) if updates else algorithm


def train_run(
    run: RunConfig,
    train_scenarios,
    val_scenarios,
    output: Path,
    *,
    forecaster=None,
    eval_limit: int | None = None,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    apply_runtime_settings(run)
    algorithm = run.algorithm
    if run.runtime.native_batch:
        from bus_rl.execution.environments.rust.batch import NativeBatchVecEnv

        env = NativeBatchVecEnv(train_scenarios, run, algorithm.seed, forecaster=forecaster)
    else:
        env = DummyVecEnv(
            [
                make_env(train_scenarios, run, algorithm.seed + index, forecaster)
                for index in range(algorithm.n_envs)
            ]
        )
    model = make_model(env, algorithm.seed, algorithm)
    validation = BestValidationCallback(
        val_scenarios or train_scenarios[:1],
        run,
        output,
        forecaster=forecaster,
        eval_limit=eval_limit,
    )
    started = perf_counter()
    model.learn(
        total_timesteps=algorithm.total_timesteps,
        callback=CallbackList([ProgressCallback(), validation]),
    )
    validation.finalize()
    wall = perf_counter() - started
    metadata = run_metadata(
        run,
        {
            "seed": algorithm.seed,
            "total_timesteps_requested": algorithm.total_timesteps,
            "total_timesteps_actual": int(model.num_timesteps),
            "episodes_completed": validation.completed_episodes,
            "wall_time_s": wall,
            "best_val_cost": None if validation.best == float("inf") else validation.best,
        },
    )
    if forecaster is not None:
        forecaster.save(output / "forecaster.npz")
    write_metadata(output, metadata)
    return metadata
