"""Checkpoint metadata and schema/config compatibility checks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from bus_rl.config import RunConfig
from bus_rl.domain import SimConfig
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
        "git_sha": sha,
        "git_dirty": dirty,
        "lock_hash": lock_hash(),
        "device": device_name(),
        "torch": torch.__version__,
        "config_source": run.source,
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


def load_model(checkpoint: Path, env, config: SimConfig):
    metadata = load_metadata(checkpoint)
    assert_schema_compatible(metadata, config)
    assert_physical_compatible(metadata, config)
    model = MaskablePPO.load(str(checkpoint), env=env, device="cpu")
    return model, metadata
