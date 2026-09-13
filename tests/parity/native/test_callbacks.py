"""O1 validation-callback tests against the shipped batched evaluator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from bus_rl.learning.training.callbacks import BestValidationCallback

pytest.importorskip("bus_sim_native")

pytestmark = pytest.mark.native


def test_callback_tie_is_strict_less_than(tmp_path, monkeypatch, reference_bundle):
    run = reference_bundle["run"]
    costs = iter([10.0, 10.0, 9.0])
    monkeypatch.setattr(
        "bus_rl.learning.training.callbacks.mean_cost", lambda *args, **kwargs: next(costs)
    )
    saved: list[str] = []

    class Dummy:
        def save(self, path):
            saved.append(Path(path).name)

    callback = BestValidationCallback(reference_bundle["scenarios"][:2], run, tmp_path)
    callback.model = Dummy()
    callback.num_timesteps = 1
    callback._evaluate()
    callback.num_timesteps = 2
    callback._evaluate()
    callback.num_timesteps = 3
    callback._evaluate()
    assert saved == ["best", "best"]
    assert callback.best == 9.0
    assert [row["val_cost"] for row in callback.history] == [10.0, 10.0, 9.0]
    callback.finalize()
    assert callback._pool is None


def test_callback_restores_model_mode_and_rng(tmp_path, reference_bundle):
    run = replace(
        reference_bundle["run"],
        runtime=replace(reference_bundle["run"].runtime, eval_batch_size=2, reuse_eval_pool=True),
    )
    model = reference_bundle["model"]
    model.policy.set_training_mode(True)
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()
    callback = BestValidationCallback(
        reference_bundle["scenarios"][:2], run, tmp_path, eval_limit=2
    )
    callback.model = model
    callback.num_timesteps = 12_288
    cost = callback._evaluate()
    assert np.isfinite(cost)
    assert model.policy.training is True
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert callback._pool is not None and not callback._pool.closed
    stores = [id(env._scenario_store) for env in callback._pool.envs]
    assert len(set(stores)) == 1
    callback.finalize()
    assert callback._pool is None
    assert (tmp_path / "best.zip").exists()
    assert (tmp_path / "last.zip").exists()
    assert (tmp_path / "evaluations.csv").exists()
