"""Request and response models for the station API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RouteLayout(BaseModel):
    route_id: int
    stop_count: int
    assigned_buses: int


class StationLayout(BaseModel):
    fleet_size: int
    reserve_size: int
    capacity: int
    comfort_capacity: int
    control_interval_s: int
    horizon_s: int
    headway_options_s: list[int]
    routes: list[RouteLayout]
    controllers: list[str]
    scenario_count: int


class CreateSimulationRequest(BaseModel):
    controller: str = "threshold"
    scenario_index: int = Field(0, ge=0)


class StepRequest(BaseModel):
    # The upper bound is `simulation.max_decisions_per_call`, checked in the router.
    decisions: int = Field(1, ge=1)


class Bus(BaseModel):
    bus_id: int
    status: str
    node: int
    direction: int
    load: int
    cooldown_until_s: int


class RouteState(BaseModel):
    route_id: int
    headway_s: int
    waiting: int
    buses: list[Bus]


class CostBreakdown(BaseModel):
    """Interval cost terms as `StepCosts`, plus the reward-weighted total."""

    waiting_pm: float = 0.0
    onboard_pm: float = 0.0
    crowding_pm: float = 0.0
    active_bus_min: float = 0.0
    deadhead_bus_min: float = 0.0
    excessive_wait_pm: float = 0.0
    first_denied_count: int = 0
    abandoned_count: int = 0
    mission_changes: int = 0
    terminal_unfinished_count: int = 0
    total: float = 0.0


class SimulationSnapshot(BaseModel):
    simulation_id: str
    controller: str
    scenario_index: int
    time_s: int
    finished: bool
    routes: list[RouteState]
    reserve_queue: list[Bus]
    interval_cost: CostBreakdown


class Decision(BaseModel):
    time_s: int
    action_index: int
    kind: str


class StepResponse(BaseModel):
    snapshot: SimulationSnapshot
    decisions: list[Decision]
