"""Backend selection for Gym environments (`runtime.backend`)."""

from __future__ import annotations

from bus_rl.backend.native import native_available, require_native
from bus_rl.config import RunConfig


def make_env_for_run(scenarios, run: RunConfig, forecaster=None):
    """Build the env selected by ``run.runtime.backend``.

    ``rust`` fails loudly when the extension is missing; the Python oracle stays
    available through explicit ``backend = "python"`` selection.
    """
    backend = run.runtime.backend
    if backend == "python":
        from bus_rl.env.bus_dispatch import BusDispatchEnv

        return BusDispatchEnv(
            scenarios,
            run.physical,
            forecaster=forecaster,
            reward=run.reward,
            control=run.control,
        )
    if backend == "rust":
        require_native()
        from bus_rl.env.native_bus_dispatch import NativeBusDispatchEnv

        return NativeBusDispatchEnv(
            scenarios,
            run.physical,
            forecaster=forecaster,
            reward=run.reward,
            control=run.control,
        )
    raise ValueError(f"unknown backend: {backend!r}")


def backend_status(run: RunConfig) -> dict:
    backend = run.runtime.backend
    return {
        "backend": backend,
        "native_available": native_available(),
        "fallback": backend == "rust" and not native_available(),
    }
