"""Checkpoint metadata and schema/config compatibility checks."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import torch
from sb3_contrib import MaskablePPO

from bus_rl.config import RunConfig
from bus_rl.execution.environments.rust.bridge import native_build_info
from bus_rl.provenance import (
    action_schema_hash,
    assert_physical_compatible,
    assert_schema_compatible,
    device_name,
    git_status,
    lock_hash,
    observation_schema_hash,
    physical_config_hash,
)
from bus_sim.oracle.domain import SimConfig


def run_metadata(run: RunConfig, extra: dict | None = None) -> dict:
    sha, dirty = git_status()
    payload = {
        "obs_version": 2,
        "action_schema_hash": action_schema_hash(),
        "obs_schema_hash": observation_schema_hash(run.physical),
        "physical_config": asdict(run.physical),
        "physical_config_hash": physical_config_hash(run.physical),
        "control": asdict(run.control),
        "control_hash": run.control_hash,
        "reward": asdict(run.reward),
        "reward_hash": run.reward_hash,
        "forecast_enabled": run.forecast.enabled,
        "algorithm": asdict(run.algorithm),
        "runtime": asdict(run.runtime),
        "git_sha": sha,
        "git_dirty": dirty,
        "lock_hash": lock_hash(),
        "device": device_name(),
        "torch": torch.__version__,
        "config_source": run.source,
        "backend": run.runtime.backend,
        "torch_threads": int(run.algorithm.torch_threads),
        "torch_threads_actual": int(torch.get_num_threads()),
        "validate_observation": bool(run.runtime.validate_observation),
        "validate_distributions": bool(run.runtime.validate_distributions),
        "torch_distribution_validate_args": bool(torch.distributions.Distribution._validate_args),
        "eval_batch_size": int(run.runtime.eval_batch_size),
        "reuse_eval_pool": bool(run.runtime.reuse_eval_pool),
        "native_batch": bool(run.runtime.native_batch),
        "native_build": native_build_info() if run.runtime.backend == "rust" else None,
        # R0-R2 parity is accepted, so checkpoints may cross backends once the
        # contract hashes below match.
        "backend_parity_verified": True,
    }
    if extra:
        payload.update(extra)
    return payload


def write_metadata(output: Path, payload: dict) -> None:
    (output / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_metadata(checkpoint: Path) -> dict:
    path = Path(checkpoint)
    meta_path = path.parent / "metadata.json" if path.suffix == ".zip" else path / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"checkpoint metadata.json is missing next to {path}")
    return json.loads(meta_path.read_text())


@contextmanager
def _legacy_model_imports():
    """Temporarily restore the module path embedded in pre-reorganization checkpoints."""
    from bus_rl.learning import policy

    package = ModuleType("bus_rl.models")
    package.__path__ = []
    package.features = policy
    previous_package = sys.modules.get("bus_rl.models")
    previous_features = sys.modules.get("bus_rl.models.features")
    sys.modules["bus_rl.models"] = package
    sys.modules["bus_rl.models.features"] = policy
    try:
        yield
    finally:
        if previous_package is None:
            sys.modules.pop("bus_rl.models", None)
        else:
            sys.modules["bus_rl.models"] = previous_package
        if previous_features is None:
            sys.modules.pop("bus_rl.models.features", None)
        else:
            sys.modules["bus_rl.models.features"] = previous_features


def load_model(checkpoint: Path, env, config: SimConfig, backend: str | None = None):
    metadata = load_metadata(checkpoint)
    assert_schema_compatible(metadata, config)
    assert_physical_compatible(metadata, config)
    if backend is not None:
        assert_backend_compatible(metadata, backend)
    with _legacy_model_imports():
        model = MaskablePPO.load(str(checkpoint), env=env, device="cpu")
    return model, metadata


def assert_backend_compatible(metadata: dict, backend: str) -> None:
    recorded = metadata.get("backend")
    if recorded in (None, backend):
        return
    if not metadata.get("backend_parity_verified", False):
        raise ValueError(
            f"checkpoint backend {recorded!r} cannot load into {backend!r}: "
            "cross-backend parity is not verified"
        )
