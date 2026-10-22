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
from bus_rl.learning.policy import policy_kwargs
from bus_rl.learning.training.callbacks import BestValidationCallback
from bus_rl.learning.training.checkpoint import run_metadata, write_metadata
from bus_rl.learning.training.diagnostics import TrainingDiagnosticsCallback
from bus_rl.provenance import canonical_hash, file_hash


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


def make_model(
    env, seed: int, algorithm: AlgorithmConfig | None = None, *, episode_steps: int = 120
):
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
        target_kl=algorithm.target_kl,
        seed=seed,
        device=algorithm.device,
        policy_kwargs=policy_kwargs(
            algorithm.encoder,
            episode_steps,
            belief_prior_shape=algorithm.belief_prior_shape,
            belief_prior_rate=algorithm.belief_prior_rate,
            belief_decay=algorithm.belief_decay,
        ),
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
    if run.shaping.enabled:
        from bus_rl.learning.shaping import PotentialShapingVecEnv

        env = PotentialShapingVecEnv(env, run.shaping, algorithm.gamma)
    model = make_model(
        env,
        algorithm.seed,
        algorithm,
        episode_steps=run.physical.horizon_s // run.physical.control_interval_s,
    )
    validation = BestValidationCallback(
        val_scenarios or train_scenarios[:1],
        run,
        output,
        forecaster=forecaster,
        eval_limit=eval_limit,
    )
    callbacks = [ProgressCallback(), validation]
    if algorithm.diagnostics_enabled:
        callbacks.append(
            TrainingDiagnosticsCallback(
                output,
                sample_interval=algorithm.diagnostics_interval,
            )
        )
    started = perf_counter()
    model.learn(
        total_timesteps=algorithm.total_timesteps,
        callback=CallbackList(callbacks),
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
            "best_timesteps": validation.best_timesteps,
            "initial_val_cost": validation.history[0]["val_cost"],
            "final_val_cost": validation.history[-1]["val_cost"],
            "checkpoint_selection": "lowest post-training validation core cost; ties keep earliest; step 0 is reference only",
            "train_days": len(train_scenarios),
            "validation_days": len(val_scenarios or train_scenarios[:1]),
            "validation_days_used": len(validation._subset()),
            "train_scenarios_hash": canonical_hash(
                [(scenario.seed, scenario.scenario_hash) for scenario in train_scenarios]
            ),
            "validation_scenarios_hash": canonical_hash(
                [(scenario.seed, scenario.scenario_hash) for scenario in validation._subset()]
            ),
            "evaluations_sha256": file_hash(output / "evaluations.csv"),
            "validation_metrics_sha256": file_hash(output / "validation_metrics.csv"),
            "best_checkpoint_sha256": file_hash(output / "best.zip"),
            "last_checkpoint_sha256": file_hash(output / "last.zip"),
            "policy_parameters": sum(parameter.numel() for parameter in model.policy.parameters()),
        },
    )
    if forecaster is not None:
        forecaster.save(output / "forecaster.npz")
    write_metadata(output, metadata)
    return metadata
