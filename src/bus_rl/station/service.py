"""Live station simulations: one native env per session, stepped on request."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from time import perf_counter

from bus_rl.config import RunConfig
from bus_rl.execution.environments.factory import make_env_for_run
from bus_rl.station import fleet
from bus_rl.station.schema import (
    Bus,
    CostBreakdown,
    Decision,
    RouteLayout,
    RouteState,
    SimulationSnapshot,
    StationLayout,
    StepResponse,
)
from bus_sim.oracle.actions import ACTION_TABLE, HEADWAYS
from bus_sim.oracle.costs import add_costs, interval_cost
from bus_sim.oracle.domain import StepCosts

logger = logging.getLogger(__name__)


@dataclass
class Simulation:
    simulation_id: str
    controller_name: str
    scenario_index: int
    env: object
    controller: object
    observation: dict
    finished: bool = False
    last_costs: StepCosts = field(default_factory=StepCosts)
    # Endpoints are sync `def`s on FastAPI's threadpool; the native env is not thread-safe.
    lock: threading.Lock = field(default_factory=threading.Lock)


class SimulationService:
    def __init__(
        self,
        run: RunConfig,
        scenarios: list,
        controllers: dict[str, object],
        max_sessions: int,
        max_decisions_per_call: int,
    ) -> None:
        self._run = run
        self._scenarios = scenarios
        self._controllers = controllers
        self._max_sessions = max_sessions
        self._max_decisions_per_call = max_decisions_per_call
        self._simulations: dict[str, Simulation] = {}
        self._simulations_lock = threading.Lock()

    def layout(self) -> StationLayout:
        physical = self._run.physical
        # Every generated day starts from the same fleet assignment (`generation.py`).
        start_routes = [vehicle.route_id for vehicle in self._scenarios[0].fleet]
        return StationLayout(
            fleet_size=physical.fleet_size,
            reserve_size=start_routes.count(None),
            capacity=physical.capacity,
            comfort_capacity=physical.comfort_capacity,
            control_interval_s=physical.control_interval_s,
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
            scenario_count=len(self._scenarios),
        )

    def create(self, controller_name: str, scenario_index: int) -> SimulationSnapshot:
        """Start a day; raises KeyError, IndexError or RuntimeError (session limit)."""
        controller = self._controllers[controller_name]
        scenario = self._scenarios[scenario_index]
        with self._simulations_lock:
            if len(self._simulations) >= self._max_sessions:
                raise RuntimeError(f"session limit {self._max_sessions} reached")
            env = make_env_for_run([scenario], self._run)
            observation, _ = env.reset(seed=0, options={"scenario_index": 0})
            simulation = Simulation(
                simulation_id=uuid.uuid4().hex,
                controller_name=controller_name,
                scenario_index=scenario_index,
                env=env,
                controller=controller,
                observation=observation,
            )
            self._simulations[simulation.simulation_id] = simulation
        logger.info(
            "simulation %s created: controller=%s scenario_index=%d",
            simulation.simulation_id,
            controller_name,
            scenario_index,
        )
        return self._snapshot(simulation)

    def get(self, simulation_id: str) -> SimulationSnapshot:
        simulation = self._simulations[simulation_id]
        with simulation.lock:
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
            if simulation.finished:
                raise RuntimeError("simulation already finished")
            started = perf_counter()
            records = self._advance(simulation, decisions)
            logger.info(
                "simulation %s stepped %d decisions in %.1f ms",
                simulation_id,
                len(records),
                (perf_counter() - started) * 1000,
            )
            return StepResponse(snapshot=self._snapshot(simulation), decisions=records)

    def delete(self, simulation_id: str) -> None:
        with self._simulations_lock:
            simulation = self._simulations.pop(simulation_id)
        simulation.env.close()
        logger.info("simulation %s deleted", simulation_id)

    def close(self) -> None:
        with self._simulations_lock:
            simulations = list(self._simulations.values())
            self._simulations.clear()
        for simulation in simulations:
            simulation.env.close()

    def _advance(self, simulation: Simulation, decisions: int) -> list[Decision]:
        env = simulation.env
        costs = StepCosts()
        records: list[Decision] = []
        for _ in range(decisions):
            time_s = int(env.kernel.current_time_s)
            action = simulation.controller.act(simulation.observation, env.action_masks())
            simulation.observation, _, terminated, truncated, info = env.step(action)
            costs = add_costs(costs, info["costs"])
            records.append(
                Decision(time_s=time_s, action_index=action, kind=ACTION_TABLE[action].kind)
            )
            if terminated or truncated:
                simulation.finished = True
                break
        simulation.last_costs = costs
        return records

    def _snapshot(self, simulation: Simulation) -> SimulationSnapshot:
        env = simulation.env
        trace = env.kernel.trace_snapshot()
        rows = env.kernel.debug_snapshot()["vehicles"].tolist()
        comfort = self._run.physical.comfort_capacity
        routes = [
            RouteState(
                route_id=route_id,
                headway_s=int(trace["headway_targets"][route_id]),
                waiting=int(trace["queues"][route_id]),
                buses=[self._bus(row, comfort) for row in route_rows],
            )
            for route_id, route_rows in enumerate(
                fleet.group_by_route(rows, self._run.physical.route_count)
            )
        ]
        costs = simulation.last_costs
        return SimulationSnapshot(
            simulation_id=simulation.simulation_id,
            controller=simulation.controller_name,
            scenario_index=simulation.scenario_index,
            time_s=int(trace["time_s"]),
            finished=simulation.finished,
            routes=routes,
            reserve_queue=[self._bus(row, comfort) for row in fleet.reserve_queue(rows)],
            interval_cost=CostBreakdown(
                **vars(costs), total=interval_cost(costs, self._run.reward)
            ),
        )

    @staticmethod
    def _bus(row: list[int], comfort_capacity: int) -> Bus:
        return Bus(
            bus_id=row[fleet.VEHICLE_ID],
            status=fleet.display_status(row[fleet.PHASE], row[fleet.LOAD], comfort_capacity),
            node=row[fleet.NODE],
            direction=row[fleet.DIRECTION],
            load=row[fleet.LOAD],
            cooldown_until_s=row[fleet.COOLDOWN_UNTIL_S],
        )
