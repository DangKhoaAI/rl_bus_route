"""Environment-driven settings for the station API (`uv sync --extra server`)."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseModel):
    # No auth on the API yet, so the default binds to loopback only.
    host: str = "127.0.0.1"
    port: int = 8000
    static_dir: str = "web/station"


class LogSettings(BaseModel):
    level: str = "INFO"


class SimulationSettings(BaseModel):
    config_path: str = "configs/eval.toml"
    # The API replays real held-out days, never an invented dataset; this is where
    # `bus-rl generate --output data/generated/base` writes them.
    manifest_path: str = "data/generated/base/manifest.json"
    # Trained policy served as the `ppo` controller, relative to the working directory (repo root);
    # empty keeps baselines only.
    checkpoint_path: str | None = "runs/core-11/best.zip"
    # Manifest splits the station can replay; the first is the default for a new
    # simulation. Held-out splits only: `train` days are what the policy learned on.
    splits: list[str] = Field(default_factory=lambda: ["validation"], min_length=1)
    max_sessions: int = Field(8, ge=1)
    max_decisions_per_call: int = Field(120, ge=1)


class AppConfig(BaseSettings):
    app_title: str = "Bus RL Station API"
    server: ServerSettings = Field(default_factory=ServerSettings)
    log: LogSettings = Field(default_factory=LogSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


app_config = AppConfig()
