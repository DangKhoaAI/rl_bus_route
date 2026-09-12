"""Native (`bus_sim`) backend helpers: scenario serialization and provenance."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bus_rl.domain import Scenario

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


def build_kernel(scenario: Scenario, *, conservation: bool = True):
    """Pack a scenario into a native kernel. Tapes are copied once here."""
    require_native()
    import bus_sim

    arrivals = np.ascontiguousarray(scenario.arrival_tape, dtype=np.int32)
    traffic = np.ascontiguousarray(scenario.traffic_tape, dtype=np.float32)
    kernel = bus_sim.Kernel(scenario_payload(scenario), arrivals, traffic)
    kernel.set_conservation_checks(conservation)
    return kernel


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
