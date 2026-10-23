"""Pure rules that turn kernel vehicle rows into the station's fleet view.

A row is one line of the kernel's 14-column vehicle matrix
(`crates/bus-sim-core/src/snapshot.rs::vehicle_row`).
"""

from __future__ import annotations

from collections.abc import Sequence

from bus_sim.oracle.domain import Phase, Scenario
from bus_sim.oracle.travel import edge_duration_s

VEHICLE_ID, CAPACITY, ROUTE_ID, NODE, DIRECTION, PHASE = 0, 1, 2, 3, 4, 5
REMAINING_S, NEXT_NODE = 6, 7
COOLDOWN_UNTIL_S, LOAD = 9, 13
DEPOT_ROUTE_ID = -1

_PHASE_STATUS = {
    Phase.DEPOT_IDLE: "depot",
    Phase.DEADHEAD: "deadhead",
    Phase.TERMINAL_IDLE: "terminal",
    Phase.SERVICE_MOVING: "moving",
    Phase.SERVICE_DWELL: "dwell",
    Phase.LAYOVER: "terminal",
}


def group_by_route(rows: Sequence[Sequence[int]], route_count: int) -> list[list[Sequence[int]]]:
    """Rows per route; a bus deadheading onto a route already carries its route id."""
    routes: list[list[Sequence[int]]] = [[] for _ in range(route_count)]
    for row in rows:
        if row[ROUTE_ID] != DEPOT_ROUTE_ID:
            routes[row[ROUTE_ID]].append(row)
    return routes


def reserve_queue(rows: Sequence[Sequence[int]]) -> list[Sequence[int]]:
    """Depot buses in display order: idle and ready first, then by id.

    This is only the order the station shows; which bus leaves is still the
    controller's `DISPATCH(b, r)` choice.
    """
    depot = [row for row in rows if row[ROUTE_ID] == DEPOT_ROUTE_ID]
    return sorted(
        depot,
        key=lambda row: (row[PHASE] != Phase.DEPOT_IDLE, row[COOLDOWN_UNTIL_S], row[VEHICLE_ID]),
    )


def display_status(phase: int, load: int, comfort_capacity: int) -> str:
    """One status word per bus; any bus above comfort load shows as crowded."""
    if load > comfort_capacity:
        return "crowded"
    return _PHASE_STATUS[Phase(phase)]


def edge_s(row: Sequence[int], scenario: Scenario, time_s: int) -> int:
    """Full duration of the edge a moving bus is on, or of the edge a dwelling bus enters next.

    The kernel keeps only the seconds left on an edge, so the dashboard cannot place a bus
    along it. Durations are sampled from the traffic tape at entry, which is the latest tick
    whose sampled duration ends exactly when the bus is due. Other phases have no edge: 0.
    """
    phase, route_id, remaining = row[PHASE], row[ROUTE_ID], row[REMAINING_S]
    index = row[NODE] % scenario.config.stops_per_route
    if phase == Phase.SERVICE_DWELL:
        return edge_duration_s(scenario, route_id, index, time_s + remaining)
    if phase != Phase.SERVICE_MOVING:
        return 0
    tick = scenario.config.tick_s
    for entry in range(time_s - time_s % tick, -1, -tick):
        duration = edge_duration_s(scenario, route_id, index, entry)
        if entry + duration == time_s + remaining:
            return duration
    return remaining
