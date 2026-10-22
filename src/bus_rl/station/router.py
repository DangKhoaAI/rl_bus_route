"""HTTP endpoints for the station dashboard."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from bus_rl.station.schema import (
    CreateSimulationRequest,
    SimulationSnapshot,
    StationLayout,
    StepRequest,
    StepResponse,
)
from bus_rl.station.service import SimulationService

router = APIRouter(tags=["station"])


def get_simulation_service(request: Request) -> SimulationService:
    return request.app.state.simulation_service


ServiceDependency = Annotated[SimulationService, Depends(get_simulation_service)]


@router.get("/station", response_model=StationLayout)
def station_layout(service: ServiceDependency) -> StationLayout:
    return service.layout()


@router.post("/simulations", response_model=SimulationSnapshot, status_code=status.HTTP_201_CREATED)
def create_simulation(
    body: CreateSimulationRequest,
    service: ServiceDependency,
) -> SimulationSnapshot:
    try:
        return service.create(body.controller, body.split, body.scenario_index)
    except KeyError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, error.args[0])
    except IndexError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"scenario_index {body.scenario_index} out of range",
        )
    except RuntimeError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error))


@router.get("/simulations/{simulation_id}", response_model=SimulationSnapshot)
def get_simulation(
    simulation_id: str,
    service: ServiceDependency,
) -> SimulationSnapshot:
    try:
        return service.get(simulation_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "simulation not found")


@router.post("/simulations/{simulation_id}/steps", response_model=StepResponse)
def step_simulation(
    simulation_id: str,
    body: StepRequest,
    service: ServiceDependency,
) -> StepResponse:
    try:
        return service.step(simulation_id, body.decisions)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "simulation not found")
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error))
    except RuntimeError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error))


@router.delete(
    "/simulations/{simulation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_simulation(
    simulation_id: str,
    service: ServiceDependency,
) -> Response:
    try:
        service.delete(simulation_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "simulation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
