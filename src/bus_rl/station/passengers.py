"""Pure rules that turn kernel cohort rows into the station's queue view.

A row is one line of the kernel's 14-column cohort matrix, prefixed with the id
of the bus carrying it (`crates/bus-sim-core/src/snapshot.rs::cohort_row`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

BUS_ID, COHORT_ID, LINEAGE_ID, ROUTE_ID, DIRECTION = 0, 1, 2, 3, 4
ORIGIN_INDEX, DESTINATION_INDEX, ARRIVAL_TICK, COUNT, STATUS = 5, 6, 7, 8, 9
FIRST_DENIED, BOARDING_TICK, COMPLETION_TICK, ABANDONMENT_TICK = 10, 11, 12, 13

# Cohort kind 0 is the waiting queue; onboard cohorts hang off vehicles and
# finished ones are archived.
WAITING = 0

# Spec tau_excess: waiting age at or above this counts as excessive.
# Mirrors `crates/bus-sim-core/src/costs.rs::EXCESSIVE_WAIT_S`.
EXCESSIVE_WAIT_S = 900


@dataclass(frozen=True)
class QueueStats:
    """One (route, direction, stop) queue; waiting ages are in seconds."""

    route_id: int
    direction: int
    stop_index: int
    waiting: int
    mean_wait_s: float
    max_wait_s: int
    excessive: int


def queues(
    rows: Sequence[Sequence[int]],
    kinds: Sequence[int],
    *,
    time_s: int,
    tick_s: int,
    route_count: int,
    stops_per_route: int,
) -> list[QueueStats]:
    """Every (route, direction, stop) cell, empty ones included, in display order.

    Spec §6.1 indexes the queue by (route, direction, origin, destination); the
    station shows the origin sum, which is what the dispatcher reacts to.
    """
    counts: dict[tuple[int, int, int], int] = {}
    weighted_age: dict[tuple[int, int, int], int] = {}
    max_age: dict[tuple[int, int, int], int] = {}
    excessive: dict[tuple[int, int, int], int] = {}
    for row, kind in zip(rows, kinds, strict=True):
        if kind != WAITING:
            continue
        key = (row[ROUTE_ID], row[DIRECTION], row[ORIGIN_INDEX])
        count = row[COUNT]
        age_s = time_s - row[ARRIVAL_TICK] * tick_s
        counts[key] = counts.get(key, 0) + count
        weighted_age[key] = weighted_age.get(key, 0) + count * age_s
        max_age[key] = max(max_age.get(key, 0), age_s)
        if age_s >= EXCESSIVE_WAIT_S:
            excessive[key] = excessive.get(key, 0) + count
    cells: list[QueueStats] = []
    for route_id in range(route_count):
        for direction in (1, -1):
            for stop_index in range(stops_per_route):
                key = (route_id, direction, stop_index)
                waiting = counts.get(key, 0)
                cells.append(
                    QueueStats(
                        route_id=route_id,
                        direction=direction,
                        stop_index=stop_index,
                        waiting=waiting,
                        mean_wait_s=(round(weighted_age[key] / waiting, 1) if waiting else 0.0),
                        max_wait_s=max_age.get(key, 0),
                        excessive=excessive.get(key, 0),
                    )
                )
    return cells
