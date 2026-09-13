"""Deterministic, backend-agnostic snapshots of simulator state.

Snapshots are pure data (NumPy arrays / JSON-compatible scalars) so they can be
written by the Python oracle and, later, produced by the Rust kernel and
compared field by field. Nothing in this module mutates the world.
"""

from __future__ import annotations

import numpy as np

# Fixed layouts. Order is part of the contract; never reorder without bumping
# the fixture schema version.
VEHICLE_FIELDS = (
    "vehicle_id",
    "capacity",
    "route_id",  # -1 == None
    "node",
    "direction",
    "phase",
    "remaining_s",
    "next_node",  # -1 == None
    "visit_id",
    "cooldown_until_s",
    "pattern",
    "turn_stop",  # -1 == None
    "pending_extra",
    "load",
)
COHORT_FIELDS = (
    "cohort_id",
    "lineage_id",
    "route_id",
    "direction",
    "origin_index",
    "destination_index",
    "arrival_tick",
    "count",
    "status",
    "first_denied",
    "boarding_tick",  # -1 == None
    "completion_tick",  # -1 == None
    "abandonment_tick",  # -1 == None
)
COUNTER_FIELDS = (
    "generated",
    "waiting",
    "onboard",
    "completed",
    "abandoned",
    "depot",
)
COST_FIELDS = (
    "waiting_pm",
    "onboard_pm",
    "crowding_pm",
    "active_bus_min",
    "deadhead_bus_min",
    "excessive_wait_pm",
    "first_denied_count",
    "abandoned_count",
    "mission_changes",
    "terminal_unfinished_count",
)
COHORT_KINDS = {"waiting": 0, "onboard": 1, "finished": 2}


def _optional(value: int | None) -> int:
    return -1 if value is None else int(value)


def vehicle_row(vehicle) -> list[int]:
    return [
        vehicle.vehicle_id,
        vehicle.capacity,
        _optional(vehicle.route_id),
        vehicle.node,
        vehicle.direction,
        int(vehicle.phase),
        vehicle.remaining_s,
        _optional(vehicle.next_node),
        vehicle.visit_id,
        vehicle.cooldown_until_s,
        int(vehicle.pattern),
        _optional(vehicle.turn_stop),
        int(vehicle.pending_extra),
        vehicle.load,
    ]


def cohort_row(cohort) -> list[int]:
    return [
        cohort.cohort_id,
        cohort.lineage_id,
        cohort.route_id,
        cohort.direction,
        cohort.origin_index,
        cohort.destination_index,
        cohort.arrival_tick,
        cohort.count,
        int(cohort.status),
        int(cohort.first_denied),
        _optional(cohort.boarding_tick),
        _optional(cohort.completion_tick),
        _optional(cohort.abandonment_tick),
    ]


def counter_array(state) -> np.ndarray:
    return np.array(
        [
            state.generated_count,
            state.waiting_count,
            state.onboard_count,
            state.completed_count,
            state.abandoned_count,
            state.depot_count,
        ],
        dtype=np.int64,
    )


def vehicle_array(state) -> np.ndarray:
    ids = sorted(state.vehicles)
    return np.array([vehicle_row(state.vehicles[i]) for i in ids], dtype=np.int64)


def _cohort_arrays(state) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[int]] = []
    kinds: list[int] = []
    for cohort in state.cohorts:
        rows.append([-1] + cohort_row(cohort))
        kinds.append(COHORT_KINDS["waiting"])
    for vehicle_id in sorted(state.vehicles):
        for cohort in state.vehicles[vehicle_id].passengers:
            rows.append([vehicle_id] + cohort_row(cohort))
            kinds.append(COHORT_KINDS["onboard"])
    for cohort in state.finished:
        rows.append([-1] + cohort_row(cohort))
        kinds.append(COHORT_KINDS["finished"])
    if not rows:
        return np.zeros((0, len(COHORT_FIELDS) + 1), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.array(rows, dtype=np.int64), np.array(kinds, dtype=np.int64)


def numpy_state_snapshot(state) -> dict[str, np.ndarray]:
    """Full world snapshot. Onboard rows are prefixed with the carrying bus id."""
    cohorts, kinds = _cohort_arrays(state)
    routes = sorted(state.headway_targets_s)
    departure_keys = _departure_keys(state)
    return {
        "time_s": np.array(state.current_time_s, dtype=np.int64),
        "counters": counter_array(state),
        "vehicles": vehicle_array(state),
        "cohorts": cohorts,
        "cohort_kind": kinds,
        "headway_targets": np.array([state.headway_targets_s[r] for r in routes], dtype=np.int64),
        "headway_changed_at": np.array(
            [state.headway_changed_at_s[r] for r in routes], dtype=np.int64
        ),
        "terminal_settled": np.array(int(state.terminal_settled), dtype=np.int64),
        "departure_keys": np.array(departure_keys, dtype=np.int64).reshape(-1, 3),
        "last_departure": np.array(
            [state.last_departure_s.get(key, -(10**9)) for key in departure_keys],
            dtype=np.int64,
        ),
        "last_full_departure": np.array(
            [state.last_full_departure_s.get(key, -(10**9)) for key in departure_keys],
            dtype=np.int64,
        ),
    }


def _departure_keys(state) -> list[tuple[int, int, int]]:
    keys = set(state.last_departure_s) | set(state.last_full_departure_s)
    return sorted(keys)


def mask_snapshot(mask) -> np.ndarray:
    return np.asarray(mask, dtype=bool)


def _array_divergence(path: str, expected, actual, rtol: float, atol: float) -> dict | None:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        return {
            "path": path,
            "reason": "shape",
            "expected": list(expected.shape),
            "actual": list(actual.shape),
        }
    if expected.dtype.kind in "biu" or actual.dtype.kind in "biu":
        equal = np.array_equal(expected, actual)
        if equal:
            return None
        flat = int(np.argmax(np.asarray(expected != actual).reshape(-1)))
    else:
        close = np.isclose(
            expected.astype(np.float64), actual.astype(np.float64), rtol=rtol, atol=atol
        )
        close = close | (np.isnan(expected) & np.isnan(actual))
        if bool(np.all(close)):
            return None
        diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        flat = int(np.argmax(diff.reshape(-1)))
    index = np.unravel_index(flat, expected.shape)
    return {
        "path": f"{path}{[int(i) for i in index]}",
        "expected": expected[index].item() if expected.ndim else expected.item(),
        "actual": actual[index].item() if actual.ndim else actual.item(),
    }


def compare_records(
    expected: dict,
    actual: dict,
    *,
    obs_rtol: float = 1e-6,
    obs_atol: float = 1e-6,
    float_rtol: float = 1e-9,
    float_atol: float = 1e-9,
    limit: int = 20,
) -> list[dict]:
    """Return up to ``limit`` divergences, first one first.

    Exact categories (masks, counters, IDs, status, event lists) use equality.
    Observation tensors use ``obs_*`` tolerance; rewards/costs use the tighter
    cost tolerance from the migration spec.
    """
    divergences: list[dict] = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            divergences.append(
                {
                    "path": key,
                    "reason": "missing" if key not in actual else "unexpected",
                    "expected": "present" if key in expected else "absent",
                    "actual": "present" if key in actual else "absent",
                }
            )
            if len(divergences) >= limit:
                return divergences
            continue
        left, right = expected[key], actual[key]
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                divergences.append({"path": key, "reason": "type"})
                continue
            divergences.extend(
                compare_records(
                    left,
                    right,
                    obs_rtol=obs_rtol,
                    obs_atol=obs_atol,
                    float_rtol=float_rtol,
                    float_atol=float_atol,
                    limit=limit - len(divergences),
                )
            )
        elif isinstance(left, list) or isinstance(right, list):
            if left != right:
                divergences.append({"path": key, "expected": left, "actual": right})
        elif "obs" in key:
            obs_key = key.split("__", 1)[1] if "__" in key else key
            found = _array_divergence(f"obs[{obs_key}]", left, right, obs_rtol, obs_atol)
            if found:
                divergences.append(found)
        elif key == "reward" or "cost" in key:
            found = _array_divergence(key, left, right, float_rtol, float_atol)
            if found:
                divergences.append(found)
        else:
            found = _array_divergence(key, left, right, 0.0, 0.0)
            if found:
                divergences.append(found)
        if len(divergences) >= limit:
            break
    return divergences[:limit]


def first_divergence(expected: dict, actual: dict, **kwargs) -> dict | None:
    found = compare_records(expected, actual, limit=1, **kwargs)
    return found[0] if found else None
