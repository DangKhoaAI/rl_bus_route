"""Request and response models for the station API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RouteLayout(BaseModel):
    route_id: int
    stop_count: int
    assigned_buses: int


class ScenarioSplit(BaseModel):
    """One manifest split the station replays."""

    name: str
    # The demand variant the split was generated with (base / burst / traffic), per
    # `execution.scenarios.generation.SPLIT_VARIANTS`.
    variant: str
    scenario_count: int


class StationLayout(BaseModel):
    fleet_size: int
    reserve_size: int
    capacity: int
    comfort_capacity: int
    control_interval_s: int
    dwell_s: int
    horizon_s: int
    # Spec §7.1: new passengers stop arriving here; the controller still works to the horizon.
    demand_end_s: int
    # Spec h_guard: a FULL departure gap wider than this is a service violation.
    service_guard_s: int
    headway_options_s: list[int]
    routes: list[RouteLayout]
    controllers: list[str]
    # The first split is the default for a new simulation.
    splits: list[ScenarioSplit]
    # Reward weight per `CostBreakdown` term, keyed by its field name, so a client
    # can split `total` into weighted terms without its own copy of the weights.
    cost_weights: dict[str, float]
    # Spec tau_excess: waiting age at or above this counts as excessive.
    excessive_wait_s: int
    # Milestone flags (spec §1.2). A disabled action stays in the table but is always masked.
    enable_reassign: bool
    enable_short_turn: bool


class CreateSimulationRequest(BaseModel):
    controller: str = "threshold"
    # Null picks the station's default split.
    split: str | None = None
    scenario_index: int = Field(0, ge=0)


class StepRequest(BaseModel):
    # The upper bound is `simulation.max_decisions_per_call`, checked in the router.
    decisions: int = Field(1, ge=1)


class Bus(BaseModel):
    bus_id: int
    status: str
    node: int
    direction: int
    # Node being driven to (-1 when none) and seconds until the current edge, dwell or layover ends.
    next_node: int
    remaining_s: int
    # Full duration of the current edge, or of the next one while dwelling; 0 otherwise.
    edge_s: int
    load: int
    cooldown_until_s: int


class RouteState(BaseModel):
    route_id: int
    headway_s: int
    waiting: int
    buses: list[Bus]


class StopQueue(BaseModel):
    """One (route, direction, stop) queue; waiting ages are in seconds."""

    route_id: int
    direction: int
    stop_index: int
    waiting: int
    mean_wait_s: float
    max_wait_s: int
    # Waiting passengers whose age is at or above the excessive-wait threshold.
    excessive: int


class DepartureClock(BaseModel):
    """One (route, direction, terminal) departure stream; times are in seconds."""

    route_id: int
    direction: int
    node: int
    last_full_s: int
    # Null until the stream has departed at least once; the kernel's negative
    # departure seeds are not a real departure time.
    full_gap_s: int | None
    violates_guard: bool


class ServiceCounters(BaseModel):
    """Conservation counters per spec §3.4.

    `generated` equals waiting + onboard + completed + abandoned; `depot` counts
    the fleet standing at the depot, which the other five do not cover.
    """

    generated: int
    waiting: int
    onboard: int
    completed: int
    abandoned: int
    depot: int


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
    split: str
    scenario_index: int
    time_s: int
    finished: bool
    # "horizon" when the day ended at H, "truncated" when it was cut short (spec §6.4),
    # null while the simulation is running.
    finish_reason: str | None
    counters: ServiceCounters
    routes: list[RouteState]
    stop_queues: list[StopQueue]
    departures: list[DepartureClock]
    reserve_queue: list[Bus]
    # Cost of the most recent control interval, the unit the spec defines in §6.3.
    interval_cost: CostBreakdown
    # Burst split only, and null until the surge starts: the viewer is never told ahead of time.
    burst: Burst | None = None


class Burst(BaseModel):
    """The burst split's extra arrivals at one stop; `added` counts only those already released."""

    route_id: int
    direction: int
    stop_index: int
    start_s: int
    end_s: int
    added: int


class RouteContext(BaseModel):
    """One route as the controller saw it just before deciding."""

    route_id: int
    waiting: int
    buses: int
    headway_s: int


class ActionChoice(BaseModel):
    action_index: int
    kind: str
    bus_id: int | None
    route_id: int | None
    headway_s: int | None
    probability: float


class DecisionContext(BaseModel):
    """What a decision was made from, so a viewer can see why it was taken."""

    routes: list[RouteContext]
    reserve: int
    legal_actions: int
    # A learned policy's most likely legal actions, highest first; empty for rule
    # controllers, whose reason is their rule.
    top_choices: list[ActionChoice]


class Decision(BaseModel):
    time_s: int
    action_index: int
    kind: str
    # The action's arguments from the action table; null where the kind takes none.
    bus_id: int | None
    route_id: int | None
    headway_s: int | None
    # Cost of the control interval this decision opened, so a caller that advances
    # several intervals can still total the episode.
    cost: float
    # Null for NOOP: only commands are explained, and NOOP is most of the day.
    context: DecisionContext | None = None


class StepResponse(BaseModel):
    snapshot: SimulationSnapshot
    decisions: list[Decision]
