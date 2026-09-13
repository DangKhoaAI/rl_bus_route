"""Portable JSON + NPZ scenario persistence. Pickle is deliberately disabled."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bus_rl.provenance import canonical_hash
from bus_sim.oracle.domain import Network, Route, Scenario, SimConfig, VehicleSpec, scenario_digest


def _validate(scenario: Scenario) -> None:
    config = scenario.config
    expected = (
        config.horizon_s // config.tick_s,
        config.route_count,
        2,
        config.stops_per_route,
        config.stops_per_route,
    )
    if scenario.arrival_tape.shape != expected or np.any(scenario.arrival_tape < 0):
        raise ValueError("arrival tape has invalid shape or negative counts")
    if np.any(scenario.arrival_tape[config.demand_end_s // config.tick_s :]):
        raise ValueError("arrivals exist outside the demand window")
    for route_id in range(config.route_count):
        for origin in range(config.stops_per_route):
            if np.any(scenario.arrival_tape[:, route_id, 0, origin, : origin + 1]):
                raise ValueError("positive-direction destination must be downstream")
            if np.any(scenario.arrival_tape[:, route_id, 1, origin, origin:]):
                raise ValueError("negative-direction destination must be downstream")


def save_scenario(scenario: Scenario, directory: Path) -> None:
    _validate(scenario)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "config": asdict(scenario.config),
        "network": {
            "routes": [asdict(r) for r in scenario.network.routes],
            "depot_node": scenario.network.depot_node,
            "edge_base_s": scenario.network.edge_base_s,
        },
        "fleet": [asdict(v) for v in scenario.fleet],
        "seed": scenario.seed,
        "scenario_hash": scenario.scenario_hash,
    }
    (directory / "scenario.json").write_text(json.dumps(metadata, sort_keys=True, indent=2))
    np.savez_compressed(
        directory / "tapes.npz", arrivals=scenario.arrival_tape, traffic=scenario.traffic_tape
    )


def load_scenario(directory: Path, *, with_tapes: bool = True) -> Scenario:
    directory = Path(directory)
    metadata = json.loads((directory / "scenario.json").read_text())
    config = SimConfig(**metadata["config"])
    network_data = metadata["network"]
    network = Network(
        tuple(Route(**route) for route in network_data["routes"]),
        network_data["depot_node"],
        tuple(network_data["edge_base_s"]),
    )
    fleet = tuple(VehicleSpec(**vehicle) for vehicle in metadata["fleet"])
    if not with_tapes:
        # Metadata-only scenario: the native store loads the tapes from `path`
        # once, so a Rust run never holds all 600 dense tapes.
        empty_arrivals = np.zeros((0,), dtype=np.int8)
        empty_traffic = np.zeros((0,), dtype=np.float32)
        empty_arrivals.setflags(write=False)
        empty_traffic.setflags(write=False)
        return Scenario(
            config,
            network,
            fleet,
            empty_arrivals,
            empty_traffic,
            metadata["seed"],
            metadata["scenario_hash"],
            path=directory,
        )
    with np.load(directory / "tapes.npz", allow_pickle=False) as arrays:
        arrivals, traffic = arrays["arrivals"], arrays["traffic"]
    # Cast so scenarios saved with an older (int32) tape dtype load identically.
    arrivals = np.ascontiguousarray(arrivals, dtype=np.int8)
    digest = scenario_digest(config, network, fleet, arrivals, traffic, metadata["seed"])
    if digest != metadata["scenario_hash"]:
        raise ValueError("scenario hash does not match persisted data")
    arrivals.setflags(write=False)
    traffic.setflags(write=False)
    scenario = Scenario(config, network, fleet, arrivals, traffic, metadata["seed"], digest)
    _validate(scenario)
    return scenario


def save_manifest(
    directory: Path, splits: dict[str, tuple[Scenario, ...]], config: SimConfig
) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    entries: dict[str, list[dict[str, object]]] = {}
    for split, scenarios in splits.items():
        items = []
        for index, scenario in enumerate(scenarios):
            relative = f"{split}/{index:04d}"
            save_scenario(scenario, directory / relative)
            items.append({"seed": scenario.seed, "hash": scenario.scenario_hash, "path": relative})
        entries[split] = items
    payload = {
        "schema_version": 2,
        "physical_config": asdict(config),
        "physical_config_hash": canonical_hash(asdict(config)),
        "splits": entries,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))
    return path


def load_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported manifest schema")
    return payload


def load_split(manifest_path: Path, split: str, *, with_tapes: bool = True) -> list[Scenario]:
    manifest_path = Path(manifest_path)
    payload = load_manifest(manifest_path)
    if split not in payload["splits"]:
        raise ValueError(f"split {split!r} is not in the manifest")
    root = manifest_path.parent
    return [
        load_scenario(root / item["path"], with_tapes=with_tapes)
        for item in payload["splits"][split]
    ]
