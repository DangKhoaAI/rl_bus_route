"""Frozen observation contract shared by the parity tests."""

from __future__ import annotations

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
