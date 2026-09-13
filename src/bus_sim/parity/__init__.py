"""Runtime parity primitives: scenario catalog, state snapshots, controllers."""

from bus_sim.parity.snapshot import (
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
