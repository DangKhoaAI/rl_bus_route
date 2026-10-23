"""Composition root for the station API.

Run with `uv run --extra server uvicorn bus_rl.main:app`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from rich.logging import RichHandler

from bus_rl.app_config import app_config
from bus_rl.config import ControlConfig, RunConfig, load_run_config
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.scenarios.io import load_manifest, load_split
from bus_rl.learning.baselines import FixedController, ProportionalController, ThresholdController
from bus_rl.provenance import physical_config_hash
from bus_rl.station.router import router as station_router
from bus_rl.station.service import SimulationService

logging.basicConfig(
    level=app_config.log.level,
    format="[cyan]%(name)s[/] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=True, rich_tracebacks=True, show_path=False)],
)
# Uvicorn installs its own plain handlers before importing the app; hand its records
# to the root handler so server and app lines share one format.
for _name in ("uvicorn", "uvicorn.access"):
    logging.getLogger(_name).handlers.clear()
    logging.getLogger(_name).propagate = True
# The dashboard polls several times a second; per-request access lines would bury
# the decision log.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app_config.simulation
    run = load_run_config(Path(settings.config_path))
    manifest_path = Path(settings.manifest_path)
    if load_manifest(manifest_path).get("physical_config_hash") != physical_config_hash(
        run.physical
    ):
        raise ValueError("physical config does not match the dataset manifest")
    scenarios = {split: load_split(manifest_path, split) for split in settings.splits}
    controllers = {
        "fixed": FixedController(),
        "threshold": ThresholdController(),
        "proportional": ProportionalController(),
    }
    forecaster = None
    if settings.checkpoint_path:
        run, forecaster, controllers["ppo"] = _load_ppo(
            Path(settings.checkpoint_path), run, scenarios[settings.splits[0]]
        )
    app.state.simulation_service = SimulationService(
        run=run,
        scenarios=scenarios,
        controllers=controllers,
        max_sessions=settings.max_sessions,
        max_decisions_per_call=settings.max_decisions_per_call,
        session_idle_s=settings.session_idle_s,
        forecaster=forecaster,
    )
    logger.info(
        "loaded scenarios %s from %s",
        {split: len(days) for split, days in scenarios.items()},
        manifest_path,
    )
    try:
        yield
    finally:
        app.state.simulation_service.close()


def _load_ppo(checkpoint: Path, run: RunConfig, scenarios: list):
    """Load a trained policy as the extra `ppo` controller.

    The checkpoint's control/reward namespaces override the station config exactly as
    `bus-rl evaluate` does, so observations and masks match training. The imports stay
    local: torch loads only when a checkpoint is configured, and the baseline-only
    image keeps it out.
    """
    from dataclasses import replace

    from bus_rl.evaluation.runner import PPOController
    from bus_rl.learning.forecasting import HistoricalForecaster
    from bus_rl.learning.training.checkpoint import load_metadata, load_model
    from bus_sim.oracle.costs import RewardConfig

    metadata = load_metadata(checkpoint)
    reward = RewardConfig(
        **{
            key: value
            for key, value in metadata["reward"].items()
            if key in RewardConfig.__dataclass_fields__
        }
    )
    run = replace(
        run,
        control=ControlConfig(**metadata["control"]),
        reward=reward,
        forecast=replace(run.forecast, enabled=bool(metadata.get("forecast_enabled"))),
    )
    frozen = checkpoint.parent / "forecaster.npz"
    forecaster = HistoricalForecaster.load(frozen) if frozen.exists() else None
    # The probe env only rebuilds the policy's spaces; each simulation makes its own.
    probe = make_env_for_run(scenarios[:1], run, forecaster=forecaster)
    try:
        model, _ = load_model(checkpoint, probe, run.physical, backend=run.runtime.backend)
    finally:
        probe.close()
    logger.info("ppo controller loaded from %s", checkpoint)
    return run, forecaster, PPOController(model)


app = FastAPI(title=app_config.app_title, version="0.1.0", lifespan=lifespan)
app.include_router(station_router)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/intro/")


# The pages ship separately under `web/`; serve them same-origin when present so the
# dashboard calls the API without CORS.
intro_directory = Path(app_config.server.static_dir).parent / "intro"
if intro_directory.is_dir():
    app.mount("/intro", StaticFiles(directory=intro_directory, html=True), name="intro")
if Path(app_config.server.static_dir).is_dir():
    app.mount(
        "/visualize",
        StaticFiles(directory=app_config.server.static_dir, html=True),
        name="visualize",
    )


if __name__ == "__main__":
    # `python -m bus_rl.main` binds where `server` settings say, so `.env` alone sets the port.
    import uvicorn

    uvicorn.run(app, host=app_config.server.host, port=app_config.server.port)
