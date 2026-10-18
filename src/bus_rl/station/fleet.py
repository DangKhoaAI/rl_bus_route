"""Pure rules that turn kernel vehicle rows into the station's fleet view.

A row is one line of the kernel's 14-column vehicle matrix
(`crates/bus-sim-core/src/snapshot.rs::vehicle_row`).
"""

from __future__ import annotations

from collections.abc import Sequence

from bus_sim.oracle.domain import Phase

VEHICLE_ID, CAPACITY, ROUTE_ID, NODE, DIRECTION, PHASE = 0, 1, 2, 3, 4, 5
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
