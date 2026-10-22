from __future__ import annotations

import importlib

import pytest

from bus_rl.cli import main as cli_main
from tests.support.paths import CONFIGS, ROOT

pytest.importorskip("fastapi")
pytestmark = pytest.mark.native


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    from fastapi.testclient import TestClient

    data = tmp_path_factory.mktemp("station") / "data"
    cli_main(
        [
            "generate",
            "--config",
            str(CONFIGS / "base.toml"),
            "--output",
            str(data),
            "--counts",
            "train=1,validation=1,test_id=1,test_ood_burst=1",
        ]
    )
    monkeypatch_module.chdir(ROOT)
    monkeypatch_module.setenv("SIMULATION__MANIFEST_PATH", str(data / "manifest.json"))
    monkeypatch_module.setenv("SIMULATION__SPLITS", '["validation", "test_ood_burst"]')
    monkeypatch_module.setenv("SIMULATION__MAX_SESSIONS", "2")
    monkeypatch_module.setenv("SIMULATION__MAX_DECISIONS_PER_CALL", "200")
    import bus_rl.app_config
    import bus_rl.main

    importlib.reload(bus_rl.app_config)
    main = importlib.reload(bus_rl.main)
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def test_station_layout_is_three_routes_of_three_with_three_reserves(client):
    layout = client.get("/station").json()
    assert layout["fleet_size"] == 12
    assert layout["reserve_size"] == 3
    assert [route["assigned_buses"] for route in layout["routes"]] == [3, 3, 3]
    assert "threshold" in layout["controllers"]


def test_simulation_runs_a_full_day_then_rejects_more_steps(client):
    created = client.post("/simulations", json={"controller": "threshold"})
    assert created.status_code == 201
    snapshot = created.json()
    assert snapshot["time_s"] == 0
    assert len(snapshot["reserve_queue"]) == 3
    assert [len(route["buses"]) for route in snapshot["routes"]] == [3, 3, 3]
    simulation_id = snapshot["simulation_id"]

    first = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 1}).json()
    assert first["snapshot"]["time_s"] == 120
    assert first["decisions"][0]["time_s"] == 0
    assert first["snapshot"]["interval_cost"]["total"] >= 0
    moving = [
        bus
        for route in first["snapshot"]["routes"]
        for bus in route["buses"]
        if bus["status"] in ("moving", "crowded")
    ]
    assert moving and all(
        bus["next_node"] >= 0 and 0 < bus["remaining_s"] <= bus["edge_s"] for bus in moving
    )

    rest = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 200}).json()
    assert rest["snapshot"]["finished"] is True
    assert rest["snapshot"]["time_s"] == 14_400
    assert len(rest["decisions"]) == 14_400 // 120 - 1

    finished = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 1})
    assert finished.status_code == 409
    assert client.delete(f"/simulations/{simulation_id}").status_code == 204
    assert client.get(f"/simulations/{simulation_id}").status_code == 404


def test_snapshot_serves_spec_counters_queues_and_departures(client):
    layout = client.get("/station").json()
    assert layout["splits"] == [
        {"name": "validation", "variant": "base", "scenario_count": 1},
        {"name": "test_ood_burst", "variant": "burst", "scenario_count": 1},
    ]
    assert layout["demand_end_s"] == 10_800
    assert layout["service_guard_s"] == 1200
    assert layout["excessive_wait_s"] == 900
    assert (layout["enable_reassign"], layout["enable_short_turn"]) == (True, True)

    simulation_id = client.post("/simulations", json={"controller": "threshold"}).json()[
        "simulation_id"
    ]
    try:
        start = client.get(f"/simulations/{simulation_id}").json()
        assert start["finish_reason"] is None
        assert len(start["stop_queues"]) == 3 * 2 * 6
        assert sum(cell["waiting"] for cell in start["stop_queues"]) == 0
        assert start["counters"]["generated"] == 0
        assert {clock["route_id"] for clock in start["departures"]} == {0, 1, 2}
        assert all(clock["violates_guard"] is False for clock in start["departures"])

        stepped = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 40}).json()
        snapshot = stepped["snapshot"]
        counters = snapshot["counters"]
        assert counters["generated"] == (
            counters["waiting"]
            + counters["onboard"]
            + counters["completed"]
            + counters["abandoned"]
        )
        assert sum(cell["waiting"] for cell in snapshot["stop_queues"]) == counters["waiting"]
        assert sum(cell["waiting"] for cell in snapshot["stop_queues"]) == sum(
            route["waiting"] for route in snapshot["routes"]
        )
        assert len(stepped["decisions"]) == 40
        assert all(decision["cost"] >= 0 for decision in stepped["decisions"])
        assert snapshot["interval_cost"]["total"] == stepped["decisions"][-1]["cost"]
        weighted = sum(
            weight * snapshot["interval_cost"][term]
            for term, weight in layout["cost_weights"].items()
        )
        assert weighted == pytest.approx(snapshot["interval_cost"]["total"])
        assert all(
            (decision["kind"] == "SET_HEADWAY") == (decision["headway_s"] is not None)
            for decision in stepped["decisions"]
        )

        rest = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 200}).json()
        assert rest["snapshot"]["finish_reason"] == "horizon"
        assert rest["snapshot"]["counters"]["generated"] > 0
    finally:
        client.delete(f"/simulations/{simulation_id}")


def test_simulation_replays_the_requested_split_and_controller(client):
    default = client.post("/simulations", json={"controller": "fixed"}).json()
    burst = client.post(
        "/simulations", json={"controller": "proportional", "split": "test_ood_burst"}
    ).json()
    try:
        assert (default["split"], default["controller"]) == ("validation", "fixed")
        assert (burst["split"], burst["controller"]) == ("test_ood_burst", "proportional")
    finally:
        for snapshot in (default, burst):
            client.delete(f"/simulations/{snapshot['simulation_id']}")


def test_errors_map_to_http_status(client):
    assert client.post("/simulations", json={"controller": "nope"}).status_code == 404
    assert client.post("/simulations", json={"scenario_index": 5}).status_code == 422
    assert client.post("/simulations", json={"split": "train"}).status_code == 404
    assert client.post("/simulations/missing/steps", json={"decisions": 1}).status_code == 404

    first = client.post("/simulations", json={}).json()["simulation_id"]
    too_many = client.post(f"/simulations/{first}/steps", json={"decisions": 201})
    assert too_many.status_code == 422
    second = client.post("/simulations", json={}).json()["simulation_id"]
    assert client.post("/simulations", json={}).status_code == 503
    for simulation_id in (first, second):
        client.delete(f"/simulations/{simulation_id}")


def test_health_and_dashboard_are_served(client):
    assert client.get("/health").json() == {"status": "ok"}
    if (ROOT / "web/station/index.html").exists():
        assert client.get("/").status_code == 200
