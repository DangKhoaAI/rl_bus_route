"""Differential-testing primitives for the Rust migration (R0 fixtures)."""

from bus_rl.parity.snapshot import (
    COHORT_FIELDS,
    VEHICLE_FIELDS,
    compare_records,
    mask_snapshot,
    numpy_state_snapshot,
)

__all__ = [
    "COHORT_FIELDS",
    "VEHICLE_FIELDS",
    "compare_records",
    "mask_snapshot",
    "numpy_state_snapshot",
]
