"""Typed domain objects shared by the scenario generator and simulator."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from hashlib import sha256

import numpy as np


class Phase(IntEnum):
    DEPOT_IDLE = 0
    DEADHEAD = 1
    TERMINAL_IDLE = 2
    SERVICE_MOVING = 3
    SERVICE_DWELL = 4
    LAYOVER = 5


class Pattern(IntEnum):
    FULL = 0
    SHORT = 1


class PassengerStatus(IntEnum):
    WAITING = 0
    ONBOARD = 1
    COMPLETED = 2
    ABANDONED = 3


CONSERVATION_CHECKS = False


@dataclass(frozen=True)
class Route:
    route_id: int
    stops: tuple[int, ...]
    short_turn_stop: int = 3


@dataclass(frozen=True)
class Network:
    routes: tuple[Route, ...]
    depot_node: int
    edge_base_s: tuple[int, ...]


@dataclass(frozen=True)
class SimConfig:
    route_count: int = 3
    stops_per_route: int = 6
    fleet_size: int = 12
    capacity: int = 40
    comfort_capacity: int = 30
    tick_s: int = 30
    control_interval_s: int = 120
    horizon_s: int = 14_400
    demand_end_s: int = 10_800
    patience_s: int = 2_700
    dwell_s: int = 30
    layover_s: int = 120
    traffic_bucket_s: int = 300
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.route_count > 4 or self.stops_per_route > 8 or self.fleet_size > 16:
            raise ValueError("config exceeds fixed observation/action encoding limits")
        if min(self.route_count, self.stops_per_route, self.fleet_size, self.capacity) <= 0:
            raise ValueError("route, stop, fleet, and capacity counts must be positive")
        if self.control_interval_s % self.tick_s or self.horizon_s % self.tick_s:
            raise ValueError("control interval and horizon must be multiples of tick_s")
        if self.demand_end_s > self.horizon_s or self.demand_end_s % self.tick_s:
            raise ValueError("demand_end_s must be a tick-aligned time within the horizon")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_id: int
    route_id: int | None
    node: int
    direction: int = 1


@dataclass(frozen=True)
class Scenario:
    config: SimConfig
    network: Network
    fleet: tuple[VehicleSpec, ...]
    arrival_tape: np.ndarray
    traffic_tape: np.ndarray
    seed: int
    scenario_hash: str
    enable_reassign: bool = False
    enable_short_turn: bool = False


@dataclass
class PassengerCohort:
    cohort_id: int
    lineage_id: int
    route_id: int
    direction: int
    origin_index: int
    destination_index: int
    arrival_tick: int
    count: int
    status: PassengerStatus = PassengerStatus.WAITING
    first_denied: bool = False
    boarding_tick: int | None = None
    completion_tick: int | None = None
    abandonment_tick: int | None = None


@dataclass
class Vehicle:
    vehicle_id: int
    capacity: int
    route_id: int | None
    node: int
    direction: int
    phase: Phase
    remaining_s: int = 0
    next_node: int | None = None
    passengers: list[PassengerCohort] = field(default_factory=list)
    visit_id: int = 0
    cooldown_until_s: int = 0
    pattern: Pattern = Pattern.FULL
    turn_stop: int | None = None
    pending_extra: bool = False
    load: int = 0


@dataclass(frozen=True)
class Action:
    kind: str = "NOOP"
    bus_id: int | None = None
    route_id: int | None = None
    headway_s: int | None = None


@dataclass
class StepCosts:
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


@dataclass
class WorldState:
    current_time_s: int
    vehicles: dict[int, Vehicle]
    cohorts: list[PassengerCohort]
    finished: list[PassengerCohort] = field(default_factory=list)
    generated_total: int = 0
    waiting_total: int = 0
    onboard_total: int = 0
    completed_total: int = 0
    abandoned_total: int = 0
    next_cohort_id: int = 0
    event_log: list[dict[str, object]] = field(default_factory=list)
    departures: list[dict[str, object]] = field(default_factory=list)
    accepted_actions: list[dict[str, object]] = field(default_factory=list)
    headway_targets_s: dict[int, int] = field(default_factory=dict)
    headway_changed_at_s: dict[int, int] = field(default_factory=dict)
    last_departure_s: dict[tuple[int, int, int], int] = field(default_factory=dict)
    last_full_departure_s: dict[tuple[int, int, int], int] = field(default_factory=dict)
    terminal_settled: bool = False

    @property
    def generated_count(self) -> int:
        return self.generated_total

    @property
    def waiting_count(self) -> int:
        return self.waiting_total

    @property
    def onboard_count(self) -> int:
        return self.onboard_total

    @property
    def completed_count(self) -> int:
        return self.completed_total

    @property
    def abandoned_count(self) -> int:
        return self.abandoned_total

    @property
    def depot_count(self) -> int:
        return sum(v.phase is Phase.DEPOT_IDLE for v in self.vehicles.values())

    def add_waiting(self, cohort: PassengerCohort) -> None:
        self.cohorts.append(cohort)
        self.waiting_total += cohort.count
        self.generated_total += cohort.count
        if cohort.cohort_id >= self.next_cohort_id:
            self.next_cohort_id = cohort.cohort_id + 1

    def iter_cohorts(self):
        yield from self.cohorts
        for bus in self.vehicles.values():
            yield from bus.passengers
        yield from self.finished

    def assert_conservation(self) -> None:
        passengers = (
            self.waiting_total + self.onboard_total + self.completed_total + self.abandoned_total
        )
        if passengers != self.generated_total:
            raise AssertionError(
                f"passenger conservation failed: {self.generated_total} != {passengers}"
            )
        if any(bus.load > bus.capacity for bus in self.vehicles.values()):
            raise AssertionError("vehicle capacity exceeded")
        if not CONSERVATION_CHECKS:
            return
        if sum(cohort.count for cohort in self.cohorts) != self.waiting_total:
            raise AssertionError("waiting counter drifted from the hot list")
        onboard = sum(bus.load for bus in self.vehicles.values())
        if onboard != self.onboard_total:
            raise AssertionError("onboard counter drifted from cached vehicle loads")
        done = sum(cohort.count for cohort in self.finished)
        if done != self.completed_total + self.abandoned_total:
            raise AssertionError("finished counter drifted from the archive list")


def maybe_check_conservation(state: WorldState) -> None:
    if CONSERVATION_CHECKS:
        state.assert_conservation()


def generate_base_network(config: SimConfig) -> Network:
    routes = tuple(
        Route(
            route_id=r,
            stops=tuple(range(r * config.stops_per_route, (r + 1) * config.stops_per_route)),
        )
        for r in range(config.route_count)
    )
    return Network(
        routes=routes,
        depot_node=config.route_count * config.stops_per_route,
        edge_base_s=tuple(180 for _ in range(config.route_count * (config.stops_per_route - 1))),
    )


def scenario_digest(
    config: SimConfig,
    network: Network,
    fleet: Iterable[VehicleSpec],
    arrivals: np.ndarray,
    traffic: np.ndarray,
    seed: int,
) -> str:
    metadata = {
        "config": asdict(config),
        "routes": [asdict(route) for route in network.routes],
        "depot_node": network.depot_node,
        "edge_base_s": network.edge_base_s,
        "fleet": [asdict(vehicle) for vehicle in fleet],
        "seed": seed,
    }
    digest = sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    digest.update(np.ascontiguousarray(arrivals).tobytes())
    digest.update(np.ascontiguousarray(traffic).tobytes())
    return digest.hexdigest()


def initial_state(scenario: Scenario) -> WorldState:
    vehicles: dict[int, Vehicle] = {}
    for spec in scenario.fleet:
        phase = Phase.DEPOT_IDLE if spec.route_id is None else Phase.TERMINAL_IDLE
        vehicles[spec.vehicle_id] = Vehicle(
            vehicle_id=spec.vehicle_id,
            capacity=scenario.config.capacity,
            route_id=spec.route_id,
            node=spec.node,
            direction=spec.direction,
            phase=phase,
        )
    return WorldState(
        current_time_s=0,
        vehicles=vehicles,
        cohorts=[],
        headway_targets_s={route.route_id: 900 for route in scenario.network.routes},
        headway_changed_at_s={route.route_id: -600 for route in scenario.network.routes},
        last_departure_s=_initial_departure_clocks(scenario),
        last_full_departure_s=_initial_departure_clocks(scenario),
    )


def _initial_departure_clocks(scenario: Scenario) -> dict[tuple[int, int, int], int]:
    clocks: dict[tuple[int, int, int], int] = {}
    for route in scenario.network.routes:
        clocks[route.route_id, 1, route.stops[0]] = -900
        clocks[route.route_id, -1, route.stops[-1]] = -900
        clocks[route.route_id, -1, route.stops[route.short_turn_stop]] = -900
    return clocks
