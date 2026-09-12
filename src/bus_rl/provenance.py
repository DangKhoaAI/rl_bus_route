"""Run metadata: hashes, git, lockfile, and schema fingerprints."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from bus_rl.control.actions import ACTION_TABLE
from bus_rl.domain import SimConfig


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def action_schema_hash() -> str:
    return canonical_hash([asdict(action) for action in ACTION_TABLE])


def observation_schema_hash(config: SimConfig | None = None) -> str:
    del config
    return canonical_hash(
        {
            "stops": [[4, 2, 8, 7], "float32"],
            "arrival_history": [[4, 2, 8, 5], "float32"],
            "forecast": [[4, 2, 8], "float32"],
            "vehicles": [[16, 27], "float32"],
            "routes": [[4, 8], "float32"],
            "stop_valid": [[4, 2, 8], "float32"],
            "vehicle_valid": [[16], "float32"],
            "route_valid": [[4], "float32"],
            "context": [[3], "float32"],
            "obs_version": 2,
        }
    )


def git_status(cwd: Path | None = None) -> tuple[str, bool]:
    root = cwd or Path.cwd()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
    return sha, dirty


def lock_hash(root: Path | None = None) -> str:
    path = (root or Path.cwd()) / "uv.lock"
    return file_hash(path) if path.exists() else "missing"


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def require_fresh_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")


def physical_config_hash(config: SimConfig) -> str:
    return canonical_hash(asdict(config))


def assert_physical_compatible(expected: dict, actual: SimConfig) -> None:
    actual_hash = physical_config_hash(actual)
    expected_hash = expected.get("physical_config_hash") or canonical_hash(
        expected.get("physical_config", {})
    )
    if expected_hash != actual_hash:
        raise ValueError("physical config does not match checkpoint or dataset")


def assert_schema_compatible(metadata: dict, config: SimConfig) -> None:
    if metadata.get("obs_version") != 2:
        raise ValueError("observation schema version mismatch")
    if metadata.get("action_schema_hash") != action_schema_hash():
        raise ValueError("action schema hash mismatch")
    if metadata.get("obs_schema_hash") != observation_schema_hash(config):
        raise ValueError("observation schema hash mismatch")
