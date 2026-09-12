"""Native (`bus_sim`) backend helpers: scenario serialization and provenance."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bus_rl.domain import Scenario, scenario_digest

NATIVE_MODULE = "bus_sim"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
NATIVE_BUILD_REPORT = PROJECT_ROOT / "reports" / "rust-migration" / "native-build.json"


def native_available() -> bool:
    return importlib.util.find_spec(NATIVE_MODULE) is not None


def require_native() -> None:
    if not native_available():
        raise RuntimeError(
            "runtime.backend='rust' requires the native `bus_sim` extension; "
            "build it with `python scripts/build_native.py`"
        )


def scenario_payload(
    scenario: Scenario,
    enable_reassign: bool | None = None,
    enable_short_turn: bool | None = None,
) -> str:
    """Serialize the immutable scenario fields the kernel owns."""
    return json.dumps(
        {
            "config": asdict(scenario.config),
            "network": {
                "routes": [
                    {
                        "route_id": route.route_id,
                        "stops": list(route.stops),
                        "short_turn_stop": route.short_turn_stop,
                    }
                    for route in scenario.network.routes
                ],
                "depot_node": scenario.network.depot_node,
                "edge_base_s": list(scenario.network.edge_base_s),
            },
            "fleet": [
                {
                    "vehicle_id": spec.vehicle_id,
                    "route_id": spec.route_id,
                    "node": spec.node,
                    "direction": spec.direction,
                }
                for spec in scenario.fleet
            ],
            "enable_reassign": (
                scenario.enable_reassign if enable_reassign is None else enable_reassign
            ),
            "enable_short_turn": (
                scenario.enable_short_turn if enable_short_turn is None else enable_short_turn
            ),
        }
    )


def _scenario_tapes(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Return `(arrivals int8, traffic float32)` for a scenario.

    Metadata-only scenarios (Rust runs do not keep the 600 dense tapes in
    Python) load their tapes from `scenario.path` once, at store construction.
    """
    if scenario.arrival_tape.size:
        return (
            np.ascontiguousarray(scenario.arrival_tape, dtype=np.int8),
            np.ascontiguousarray(scenario.traffic_tape, dtype=np.float32),
        )
    if scenario.path is None:
        raise ValueError("scenario has neither tapes nor a path")
    with np.load(Path(scenario.path) / "tapes.npz", allow_pickle=False) as arrays:
        arrivals = np.ascontiguousarray(arrays["arrivals"], dtype=np.int8)
        traffic = np.ascontiguousarray(arrays["traffic"], dtype=np.float32)
    digest = scenario_digest(
        scenario.config, scenario.network, scenario.fleet, arrivals, traffic, scenario.seed
    )
    if digest != scenario.scenario_hash:
        raise ValueError(f"scenario hash does not match persisted data: {scenario.path}")
    return arrivals, traffic


def build_kernel(scenario: Scenario, *, conservation: bool = True):
    """Pack a scenario into a native kernel. Tapes are copied once here."""
    require_native()
    import bus_sim

    arrivals, traffic = _scenario_tapes(scenario)
    kernel = bus_sim.Kernel(scenario_payload(scenario), arrivals, traffic)
    kernel.set_conservation_checks(conservation)
    return kernel


class NativeScenarioStore:
    """Owns packed native scenarios so many env kernels share immutable tapes.

    Without this, ``n_envs`` envs each copy every scenario tape; with a shared
    store the tapes exist once per distinct scenario list.
    """

    def __init__(self, scenarios) -> None:
        require_native()
        import bus_sim

        self._store = bus_sim.ScenarioStore()
        for scenario in scenarios:
            arrivals, traffic = _scenario_tapes(scenario)
            self._store.add(scenario_payload(scenario), arrivals, traffic)

    def __len__(self) -> int:
        return len(self._store)

    def kernel(self, index: int, *, conservation: bool = True):
        import bus_sim

        kernel = bus_sim.Kernel.from_store(self._store, index)
        kernel.set_conservation_checks(conservation)
        return kernel

    def batch_kernel(self, capacity: int, *, conservation: bool = True):
        """One native kernel owning ``capacity`` episode slots over this store."""
        import bus_sim

        kernel = bus_sim.BatchKernel.from_store(self._store, int(capacity))
        kernel.set_conservation_checks(conservation)
        return kernel


_STORE_CACHE: dict[tuple, NativeScenarioStore] = {}
_STORE_CACHE_LIMIT = 4


def shared_store(scenarios) -> NativeScenarioStore:
    """Reuse one packed store for an identical scenario list within a process."""
    scenarios = tuple(scenarios)
    key = tuple(
        (scenario.scenario_hash, bool(scenario.enable_reassign), bool(scenario.enable_short_turn))
        for scenario in scenarios
    )
    cached = _STORE_CACHE.get(key)
    if cached is not None:
        return cached
    store = NativeScenarioStore(scenarios)
    if len(_STORE_CACHE) >= _STORE_CACHE_LIMIT:
        _STORE_CACHE.pop(next(iter(_STORE_CACHE)))
    _STORE_CACHE[key] = store
    return store


def native_build_info(root: Path | None = None) -> dict | None:
    path = (root / "reports/rust-migration/native-build.json") if root else NATIVE_BUILD_REPORT
    if not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text())
    return {
        "crate": payload.get("crate"),
        "crate_version": payload.get("crate_version"),
        "library_sha256": payload.get("library_sha256"),
        "rustc": payload.get("rustc"),
        "cargo": payload.get("cargo"),
        "git_sha": payload.get("git", {}).get("sha"),
    }
