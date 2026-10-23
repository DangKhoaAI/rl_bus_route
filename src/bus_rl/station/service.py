"""Live station simulations: one native env per session, stepped on request."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from time import perf_counter

from bus_rl.config import RunConfig
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.scenarios.generation import SPLIT_VARIANTS
from bus_rl.station import departures, fleet, passengers
from bus_rl.station.schema import (
    Bus,
    CostBreakdown,
    Decision,
    DepartureClock,
    RouteLayout,
    RouteState,
    ScenarioSplit,
    ServiceCounters,
    SimulationSnapshot,
    StationLayout,
    StepResponse,
    StopQueue,
)
from bus_sim.oracle.actions import ACTION_TABLE, HEADWAYS
from bus_sim.oracle.costs import interval_cost
from bus_sim.oracle.domain import Scenario, StepCosts

logger = logging.getLogger(__name__)


@dataclass
class Simulation:
    simulation_id: str
    controller_name: str
    split: str
    scenario_index: int
    scenario: Scenario
    env: object
    controller: object
    observation: dict
    finished: bool = False
    finish_reason: str | None = None
    closed: bool = False
    last_costs: StepCosts = field(default_factory=StepCosts)
    # Endpoints are sync `def`s on FastAPI's threadpool; the native env is not thread-safe.
    lock: threading.Lock = field(default_factory=threading.Lock)


class SimulationService:
    def __init__(
        self,
        run: RunConfig,
        scenarios: dict[str, list[Scenario]],
        controllers: dict[str, object],
        max_sessions: int,
        max_decisions_per_call: int,
        forecaster: object | None = None,
    ) -> None:
        self._run = run
        self._scenarios = scenarios
        self._controllers = controllers
        self._max_sessions = max_sessions
        self._max_decisions_per_call = max_decisions_per_call
        self._forecaster = forecaster
        self._simulations: dict[str, Simulation] = {}
        self._simulations_lock = threading.Lock()

    def layout(self) -> StationLayout:
        physical = self._run.physical
        logger.debug(
            "layout served: fleet=%d routes=%d controllers=%s",
            physical.fleet_size,
            physical.route_count,
            sorted(self._controllers),
        )
        # Every generated day of every split starts from the same fleet assignment
        # and dwell (`generation.py`), so any one day describes the station.
        first_day = next(iter(self._scenarios.values()))[0]
        start_routes = [vehicle.route_id for vehicle in first_day.fleet]
        return StationLayout(
            fleet_size=physical.fleet_size,
            reserve_size=start_routes.count(None),
            capacity=physical.capacity,
            comfort_capacity=physical.comfort_capacity,
            control_interval_s=physical.control_interval_s,
            dwell_s=first_day.config.dwell_s,
            horizon_s=physical.horizon_s,
            headway_options_s=list(HEADWAYS),
            routes=[
                RouteLayout(
                    route_id=route_id,
                    stop_count=physical.stops_per_route,
                    assigned_buses=start_routes.count(route_id),
                )
                for route_id in range(physical.route_count)
            ],
            controllers=sorted(self._controllers),
            splits=[
                ScenarioSplit(name=name, variant=SPLIT_VARIANTS[name], scenario_count=len(days))
                for name, days in self._scenarios.items()
            ],
            demand_end_s=physical.demand_end_s,
            cost_weights=self._cost_weights(),
            excessive_wait_s=passengers.EXCESSIVE_WAIT_S,
            service_guard_s=departures.SERVICE_GUARD_S,
            enable_reassign=self._run.control.enable_reassign,
            enable_short_turn=self._run.control.enable_short_turn,
        )

    def _cost_weights(self) -> dict[str, float]:
        """The weights `interval_cost` applies, keyed by `CostBreakdown` field."""
        reward = self._run.reward
        return {
            "waiting_pm": reward.waiting,
            "onboard_pm": reward.onboard,
            "crowding_pm": reward.crowding,
            "active_bus_min": reward.active,
            "deadhead_bus_min": reward.deadhead,
            "excessive_wait_pm": reward.fairness,
            "first_denied_count": reward.first_denied,
            "abandoned_count": reward.abandoned,
            "mission_changes": reward.mission,
            "terminal_unfinished_count": reward.unfinished,
        }

    def create(
        self, controller_name: str, split: str | None, scenario_index: int
    ) -> SimulationSnapshot:
        """Start a day; raises KeyError, IndexError or RuntimeError (session limit).

        A null `split` picks the default, the first configured one.
        """
        split = split or next(iter(self._scenarios))
        if controller_name not in self._controllers:
            raise KeyError(f"unknown controller {controller_name!r}")
        if split not in self._scenarios:
            raise KeyError(f"unknown split {split!r}")
        controller = self._controllers[controller_name]
        scenario = self._scenarios[split][scenario_index]
        with self._simulations_lock:
            if len(self._simulations) >= self._max_sessions:
                raise RuntimeError(f"session limit {self._max_sessions} reached")
            env = make_env_for_run([scenario], self._run, forecaster=self._forecaster)
            observation, _ = env.reset(seed=0, options={"scenario_index": 0})
            simulation = Simulation(
                simulation_id=uuid.uuid4().hex,
                controller_name=controller_name,
                split=split,
                scenario_index=scenario_index,
                scenario=scenario,
                env=env,
                controller=controller,
                observation=observation,
            )
            self._simulations[simulation.simulation_id] = simulation
        logger.info(
            "simulation %s created: controller=%s split=%s scenario_index=%d",
            simulation.simulation_id,
            controller_name,
            split,
            scenario_index,
        )
        return self._snapshot(simulation)

    def get(self, simulation_id: str) -> SimulationSnapshot:
        simulation = self._simulations[simulation_id]
        with simulation.lock:
            self._ensure_open(simulation)
            return self._snapshot(simulation)

    def step(self, simulation_id: str, decisions: int) -> StepResponse:
        """Advance up to `decisions` control intervals.

        Raises KeyError (unknown id), ValueError (too many decisions) or
        RuntimeError (the day already finished).
        """
        if decisions > self._max_decisions_per_call:
            raise ValueError(f"decisions must be <= {self._max_decisions_per_call}")
        simulation = self._simulations[simulation_id]
        with simulation.lock:
            self._ensure_open(simulation)
            if simulation.finished:
                raise RuntimeError("simulation already finished")
            started = perf_counter()
            records = self._advance(simulation, decisions)
            snapshot = self._snapshot(simulation)
            # Routine ticks stay at DEBUG; only real decisions and the end of the day
            # stand out in the console.
            logger.debug(
                "simulation %s stepped %d decisions in %.1f ms: time_s=%d waiting=%s "
                "reserve=%d cost=%.1f",
                simulation_id,
                len(records),
                (perf_counter() - started) * 1000,
                snapshot.time_s,
                [route.waiting for route in snapshot.routes],
                len(snapshot.reserve_queue),
                snapshot.interval_cost.total,
            )
            for record in records:
                if record.kind != "NOOP":
                    logger.info(
                        "[bold yellow]%s[/] %s t=%ds bus=%s route=%s headway_s=%s cost=%.1f",
                        record.kind,
                        simulation_id[:8],
                        record.time_s,
                        record.bus_id,
                        record.route_id,
                        record.headway_s,
                        record.cost,
                    )
            if snapshot.finished:
                logger.info(
                    "[bold green]FINISHED[/] %s reason=%s time_s=%d counters=%s",
                    simulation_id[:8],
                    snapshot.finish_reason,
                    snapshot.time_s,
                    snapshot.counters,
                )
            return StepResponse(snapshot=snapshot, decisions=records)

    def delete(self, simulation_id: str) -> None:
        with self._simulations_lock:
            simulation = self._simulations.pop(simulation_id)
        self._close(simulation)
        logger.info("simulation %s deleted", simulation_id)

    def close(self) -> None:
        with self._simulations_lock:
            simulations = list(self._simulations.values())
            self._simulations.clear()
        for simulation in simulations:
            self._close(simulation)

    @staticmethod
    def _ensure_open(simulation: Simulation) -> None:
        """A request that waited on the lock of a deleted session gets KeyError."""
        if simulation.closed:
            raise KeyError(simulation.simulation_id)

    @staticmethod
    def _close(simulation: Simulation) -> None:
        # Waits out an in-flight step: closing the native env under it is unsafe.
        with simulation.lock:
            simulation.closed = True
            simulation.env.close()

    def _advance(self, simulation: Simulation, decisions: int) -> list[Decision]:
        env = simulation.env
        costs = StepCosts()
        records: list[Decision] = []
        for _ in range(decisions):
            time_s = int(env.kernel.current_time_s)
            action = simulation.controller.act(simulation.observation, env.action_masks())
            simulation.observation, _, terminated, truncated, info = env.step(action)
            costs = info["costs"]
            records.append(
                Decision(
                    time_s=time_s,
                    action_index=action,
                    kind=ACTION_TABLE[action].kind,
                    bus_id=ACTION_TABLE[action].bus_id,
                    route_id=ACTION_TABLE[action].route_id,
                    headway_s=ACTION_TABLE[action].headway_s,
                    cost=interval_cost(costs, self._run.reward),
                )
            )
            if terminated or truncated:
                # The horizon is the task's own terminal state; anything else is a
                # truncation and the reason has to stay visible (spec §6.4).
                simulation.finish_reason = "horizon" if terminated else "truncated"
                simulation.finished = True
                break
        simulation.last_costs = costs
        return records

    def _snapshot(self, simulation: Simulation) -> SimulationSnapshot:
        """Serve the kernel's full state snapshot; it is the one source for every field."""
        physical = self._run.physical
        state = simulation.env.kernel.debug_snapshot()
        time_s = int(state["time_s"])
        rows = state["vehicles"].tolist()
        comfort = physical.comfort_capacity
        scenario = simulation.scenario
        stop_queues = passengers.queues(
            state["cohorts"].tolist(),
            state["cohort_kind"].tolist(),
            time_s=time_s,
            tick_s=physical.tick_s,
            route_count=physical.route_count,
            stops_per_route=physical.stops_per_route,
        )
        counters = state["counters"].tolist()
        costs = simulation.last_costs
        return SimulationSnapshot(
            simulation_id=simulation.simulation_id,
            controller=simulation.controller_name,
            split=simulation.split,
            scenario_index=simulation.scenario_index,
            time_s=time_s,
            finished=simulation.finished,
            finish_reason=simulation.finish_reason,
            counters=ServiceCounters(
                generated=counters[0],
                waiting=counters[1],
                onboard=counters[2],
                completed=counters[3],
                abandoned=counters[4],
                depot=counters[5],
            ),
            routes=[
                RouteState(
                    route_id=route_id,
                    headway_s=int(state["headway_targets"][route_id]),
                    waiting=sum(cell.waiting for cell in stop_queues if cell.route_id == route_id),
                    buses=[self._bus(row, comfort, scenario, time_s) for row in route_rows],
                )
                for route_id, route_rows in enumerate(
                    fleet.group_by_route(rows, physical.route_count)
                )
            ],
            stop_queues=[StopQueue(**vars(cell)) for cell in stop_queues],
            departures=[
                DepartureClock(**vars(clock))
                for clock in departures.clocks(
                    state["departure_keys"].tolist(),
                    state["last_full_departure"].tolist(),
                    time_s=time_s,
                )
            ],
            reserve_queue=[
                self._bus(row, comfort, scenario, time_s) for row in fleet.reserve_queue(rows)
            ],
            interval_cost=CostBreakdown(
                **vars(costs), total=interval_cost(costs, self._run.reward)
            ),
        )

    @staticmethod
    def _bus(row: list[int], comfort_capacity: int, scenario: Scenario, time_s: int) -> Bus:
        return Bus(
            bus_id=row[fleet.VEHICLE_ID],
            status=fleet.display_status(row[fleet.PHASE], row[fleet.LOAD], comfort_capacity),
            node=row[fleet.NODE],
            direction=row[fleet.DIRECTION],
            next_node=row[fleet.NEXT_NODE],
            remaining_s=row[fleet.REMAINING_S],
            edge_s=fleet.edge_s(row, scenario, time_s),
            load=row[fleet.LOAD],
            cooldown_until_s=row[fleet.COOLDOWN_UNTIL_S],
        )
