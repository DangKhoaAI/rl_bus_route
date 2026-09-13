"""Frozen interface constants at the Python↔Rust (`bus_sim_native`) boundary.

These describe the *contract*, not the physics. They mirror the Rust side
(`crates/bus-sim-python/src/lib.rs`, `crates/bus-sim-core/src/actions.rs`) and
the Python consumer (`src/bus_rl/execution/environments/rust/`). Change one
side without the other and these tests fail.
"""

from __future__ import annotations

# `bus_sim_core::actions::ACTION_COUNT`; the Gym action space is Discrete(221).
MASK_SIZE = 221

# `Kernel.step_contract()["costs"]` length; order is pinned to
# `bus_sim.parity.snapshot.COST_FIELDS` by test_kernel_contract.
COST_COUNT = 10

# Observation keys produced by `obs_dict` in the bridge and `observe` in
# `bus_sim.oracle.observation`.
OBS_KEYS = frozenset(
    {
        "stops",
        "arrival_history",
        "forecast",
        "vehicles",
        "routes",
        "stop_valid",
        "vehicle_valid",
        "route_valid",
        "context",
    }
)

# Batched observation shapes are `(N, *single_env_shape)`.
BATCH_OBS_SHAPES = {
    "stops": (4, 2, 8, 7),
    "arrival_history": (4, 2, 8, 5),
    "forecast": (4, 2, 8),
    "vehicles": (16, 27),
    "routes": (4, 8),
    "stop_valid": (4, 2, 8),
    "vehicle_valid": (16,),
    "route_valid": (4,),
    "context": (3,),
}
