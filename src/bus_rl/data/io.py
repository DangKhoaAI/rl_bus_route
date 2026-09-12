"""Portable JSON + NPZ scenario persistence. Pickle is deliberately disabled."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bus_rl.domain import Network, Route, Scenario, SimConfig, VehicleSpec, scenario_digest


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


def load_scenario(directory: Path) -> Scenario:
    metadata = json.loads((directory / "scenario.json").read_text())
    with np.load(directory / "tapes.npz", allow_pickle=False) as arrays:
        arrivals, traffic = arrays["arrivals"], arrays["traffic"]
    config = SimConfig(**metadata["config"])
    network_data = metadata["network"]
    network = Network(
        tuple(Route(**route) for route in network_data["routes"]),
        network_data["depot_node"],
        tuple(network_data["edge_base_s"]),
    )
    fleet = tuple(VehicleSpec(**vehicle) for vehicle in metadata["fleet"])
    digest = scenario_digest(config, network, fleet, arrivals, traffic, metadata["seed"])
    if digest != metadata["scenario_hash"]:
        raise ValueError("scenario hash does not match persisted data")
    arrivals.setflags(write=False)
    traffic.setflags(write=False)
    scenario = Scenario(config, network, fleet, arrivals, traffic, metadata["seed"], digest)
    _validate(scenario)
    return scenario
