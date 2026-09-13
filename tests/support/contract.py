"""Frozen observation/kernel contract shared by the parity tests."""

from __future__ import annotations

from bus_sim.parity.snapshot import COST_FIELDS

# Observation keys in the frozen oracle order; mirror
# `bus_sim.oracle.observation.observe`.
OBS_KEYS = (
    "stops",
    "arrival_history",
    "forecast",
    "vehicles",
    "routes",
    "stop_valid",
    "vehicle_valid",
    "route_valid",
    "context",
)

# Keys the R1 kernel owns and must match the oracle exactly. Observation and
# mask parity belong to R2.
KERNEL_COMPARE_KEYS = (
    (
        "time_s",
        "counters",
        "vehicles",
        "cohort_unique",
        "cohort_order",
        "cohort_ptr",
        "headway_targets",
        "headway_changed_at",
        "departure_keys",
        "last_departure",
        "last_full_departure",
        "terminal_settled",
        "reward",
        "terminated",
        "truncated",
        "mean_wait",
        "tick__vehicles",
        "tick__counters",
        "tick__time_s",
        "tick__events",
    )
    + tuple(f"costs__{field}" for field in COST_FIELDS)
    + tuple(f"tick__costs__{field}" for field in COST_FIELDS)
)
