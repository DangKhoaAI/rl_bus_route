"""Live station simulations: one native env per session, stepped on request."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from time import monotonic, perf_counter

import numpy as np

from bus_rl.config import RunConfig
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.execution.scenarios.generation import SPLIT_VARIANTS, generate_scenario
from bus_rl.station import departures, fleet, passengers
from bus_rl.station.schema import (
    ActionChoice,
    Burst,
    Bus,
    CostBreakdown,
    Decision,
    DecisionContext,
    DepartureClock,
    RouteContext,
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


@dataclass(frozen=True)
class Surge:
    """Where a burst day adds passengers, and how many at each tick it touches."""

    route_id: int
    direction: int
    stop_index: int
    extras: list[tuple[int, int]]  # (tick, extra passengers), in tick order


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
    last_used_s: float = field(default_factory=monotonic)
    last_costs: StepCosts = field(default_factory=StepCosts)
    surge: Surge | None = None
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
        session_idle_s: float,
        forecaster: object | None = None,
    ) -> None:
        self._run = run
        self._scenarios = scenarios
        self._controllers = controllers
        self._max_sessions = max_sessions
        self._max_decisions_per_call = max_decisions_per_call
        self._session_idle_s = session_idle_s
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
                self._evict_idle()
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
                surge=self._surge(scenario) if SPLIT_VARIANTS[split] == "burst" else None,
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

    @staticmethod
    def _surge(scenario: Scenario) -> Surge:
        """Recover the burst by diffing the tape against the same seed's base day.

        The generator draws the burst after everything else, so the base day is the
        burst day minus exactly the extra arrivals.
        """
        base = generate_scenario(scenario.seed, scenario.config, "base").arrival_tape
        extra = scenario.arrival_tape.astype(np.int32) - base.astype(np.int32)
        cells = np.argwhere(extra > 0)
        _, route_id, direction_index, stop_index, _ = (int(v) for v in cells[0])
        per_tick = extra.sum(axis=(1, 2, 3, 4))
        return Surge(
            route_id=route_id,
            direction=1 if direction_index == 0 else -1,
            stop_index=stop_index,
            extras=[(int(tick), int(per_tick[tick])) for tick in np.flatnonzero(per_tick)],
        )

    def get(self, simulation_id: str) -> SimulationSnapshot:
        simulation = self._simulations[simulation_id]
        with simulation.lock:
            self._touch(simulation)
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
            self._touch(simulation)
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
    def _touch(simulation: Simulation) -> None:
        """Mark a session in use; a request that waited on the lock of a deleted one gets KeyError."""
        if simulation.closed:
            raise KeyError(simulation.simulation_id)
        simulation.last_used_s = monotonic()

    @staticmethod
    def _close(simulation: Simulation) -> None:
        # Waits out an in-flight step: closing the native env under it is unsafe.
        with simulation.lock:
            simulation.closed = True
            simulation.env.close()

    def _evict_idle(self) -> None:
        """Free sessions nobody stepped or read for `session_idle_s`; caller holds `_simulations_lock`.

        A dashboard tab that crashed or lost its network never sends DELETE, so without
        this its session would hold one of the `max_sessions` slots until restart.
        """
        cutoff = monotonic() - self._session_idle_s
        for simulation_id, simulation in list(self._simulations.items()):
            if simulation.last_used_s < cutoff:
                del self._simulations[simulation_id]
                self._close(simulation)
                logger.info("simulation %s evicted after idling", simulation_id)

    def _advance(self, simulation: Simulation, decisions: int) -> list[Decision]:
        env = simulation.env
        costs = StepCosts()
        records: list[Decision] = []
        for _ in range(decisions):
            time_s = int(env.kernel.current_time_s)
            mask = env.action_masks()
            action = simulation.controller.act(simulation.observation, mask)
            # NOOP is a decision too: holding the plan is explained like any command.
            context = self._context(simulation, mask)
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
                    context=context,
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

    def _context(self, simulation: Simulation, mask) -> DecisionContext:
        """The state a decision was made from; call before the env steps."""
        snapshot = self._snapshot(simulation)
        top_choices = []
        probabilities = getattr(simulation.controller, "action_probabilities", None)
        if probabilities is not None:
            p = probabilities(simulation.observation, mask)
            for index in p.argsort()[::-1][:3]:
                if p[index] <= 0:
                    break
                action = ACTION_TABLE[int(index)]
                top_choices.append(
                    ActionChoice(
                        action_index=int(index),
                        kind=action.kind,
                        bus_id=action.bus_id,
                        route_id=action.route_id,
                        headway_s=action.headway_s,
                        probability=round(float(p[index]), 4),
                    )
                )
        return DecisionContext(
            routes=[
                RouteContext(
                    route_id=route.route_id,
                    waiting=route.waiting,
                    buses=len(route.buses),
                    headway_s=route.headway_s,
                )
                for route in snapshot.routes
            ],
            reserve=len(snapshot.reserve_queue),
            legal_actions=int(sum(mask)),
            top_choices=top_choices,
        )

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
            burst=self._burst(simulation.surge, time_s, physical.tick_s),
        )

    @staticmethod
    def _burst(surge: Surge | None, time_s: int, tick_s: int) -> Burst | None:
        """The surge as released by `time_s`; null before its first tick has been simulated."""
        if surge is None or surge.extras[0][0] * tick_s >= time_s:
            return None
        end_s = (surge.extras[-1][0] + 1) * tick_s
        return Burst(
            route_id=surge.route_id,
            direction=surge.direction,
            stop_index=surge.stop_index,
            start_s=surge.extras[0][0] * tick_s,
            end_s=end_s if end_s <= time_s else None,
            added=sum(extra for tick, extra in surge.extras if tick * tick_s < time_s),
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
