"""Composition root for the station API.

Run with `uv run --extra server uvicorn bus_rl.main:app`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from bus_rl.app_config import app_config
from bus_rl.config import load_run_config
from bus_rl.execution.scenarios.io import load_manifest, load_split
from bus_rl.learning.baselines import FixedController, ProportionalController, ThresholdController
from bus_rl.provenance import physical_config_hash
from bus_rl.station.router import router as station_router
from bus_rl.station.service import SimulationService

logging.basicConfig(
    level=app_config.log.level,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
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
    scenarios = load_split(manifest_path, settings.split)
    controllers = {
        "fixed": FixedController(),
        "threshold": ThresholdController(),
        "proportional": ProportionalController(),
    }
    app.state.simulation_service = SimulationService(
        run=run,
        scenarios=scenarios,
        controllers=controllers,
        max_sessions=settings.max_sessions,
        max_decisions_per_call=settings.max_decisions_per_call,
    )
    logger.info("loaded %d %s scenarios from %s", len(scenarios), settings.split, manifest_path)
    try:
        yield
    finally:
        app.state.simulation_service.close()


app = FastAPI(title=app_config.app_title, version="0.1.0", lifespan=lifespan)
app.include_router(station_router)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}


# The dashboard ships separately under `web/`; serve it same-origin when present.
if Path(app_config.server.static_dir).is_dir():
    app.mount("/", StaticFiles(directory=app_config.server.static_dir, html=True), name="web")
