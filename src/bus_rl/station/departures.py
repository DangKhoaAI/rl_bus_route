"""Pure rules that turn the kernel's departure clocks into the station's service view.

The kernel keys departures by (route, direction, terminal node) and keeps the
last ANY and the last FULL departure of each stream
(`crates/bus-sim-core/src/snapshot.rs::full_snapshot`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Spec h_guard: a FULL departure gap wider than this is reported as a violation.
# Mirrors `crates/bus-sim-core/src/guards.rs`.
SERVICE_GUARD_S = 1200


@dataclass(frozen=True)
class DepartureClock:
    """One (route, direction, terminal) departure stream; times are in seconds."""

    route_id: int
    direction: int
    node: int
    last_full_s: int
    full_gap_s: int | None
    violates_guard: bool


def clocks(
    keys: Sequence[Sequence[int]],
    last_full: Sequence[int],
    *,
    time_s: int,
) -> list[DepartureClock]:
    """Report the current gap since the last FULL departure of each stream.

    A stream the kernel has never seen a FULL departure from keeps its seed value
    (-900s for the seeded terminals, a large negative sentinel for the rest), so
    the gap is reported as unknown rather than as a violation the evidence does
    not support.
    """
    return [
        DepartureClock(
            route_id=row[0],
            direction=row[1],
            node=row[2],
            last_full_s=full_s,
            full_gap_s=time_s - full_s if full_s >= 0 else None,
            violates_guard=full_s >= 0 and time_s - full_s > SERVICE_GUARD_S,
        )
        for row, full_s in zip(keys, last_full, strict=True)
    ]
