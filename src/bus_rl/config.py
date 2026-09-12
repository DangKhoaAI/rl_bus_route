"""TOML run configuration: physical / control / reward / algorithm namespaces."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from bus_rl.domain import SimConfig
from bus_rl.provenance import canonical_hash
from bus_rl.rewards.costs import RewardConfig


@dataclass(frozen=True)
class ControlConfig:
    enable_reassign: bool = True
    enable_short_turn: bool = True


@dataclass(frozen=True)
class AlgorithmConfig:
    gamma: float = 1.0
    seed: int = 11
    learning_rate: float = 3.0e-4
    n_steps: int = 256
    n_envs: int = 4
    batch_size: int = 256
    n_epochs: int = 4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    total_timesteps: int = 245_760
    eval_freq: int = 12_288
    device: str = "cpu"


@dataclass(frozen=True)
class ForecastConfig:
    enabled: bool = False


@dataclass(frozen=True)
class RunConfig:
    physical: SimConfig
    control: ControlConfig
    reward: RewardConfig
    algorithm: AlgorithmConfig
    forecast: ForecastConfig
    source: str

    @property
    def control_hash(self) -> str:
        return canonical_hash(asdict(self.control))

    @property
    def reward_hash(self) -> str:
        return canonical_hash(asdict(self.reward))


def _load_toml(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_section(cls, payload: dict):
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def load_run_config(path: Path, project_root: Path | None = None) -> RunConfig:
    root = project_root or Path.cwd()
    path = Path(path)
    merged: dict = {}
    base = root / "configs" / "base.toml"
    if base.exists() and path.resolve() != base.resolve():
        merged = _load_toml(base)
    merged = _deep_merge(merged, _load_toml(path))
    physical_payload = merged.get("physical", {})
    return RunConfig(
        physical=_from_section(SimConfig, physical_payload),
        control=_from_section(ControlConfig, merged.get("control", {})),
        reward=_from_section(RewardConfig, merged.get("reward", {})),
        algorithm=_from_section(AlgorithmConfig, merged.get("algorithm", {})),
        forecast=_from_section(ForecastConfig, merged.get("forecast", {})),
        source=str(path),
    )


def parse_counts(text: str | None) -> dict[str, int]:
    from bus_rl.data.scenario import DEFAULT_COUNTS

    if not text:
        return dict(DEFAULT_COUNTS)
    counts: dict[str, int] = {}
    for part in text.split(","):
        name, value = part.split("=", 1)
        counts[name.strip()] = int(value)
    return counts
