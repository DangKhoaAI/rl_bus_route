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
            "train=1,validation=1,test_id=1",
        ]
    )
    monkeypatch_module.chdir(ROOT)
    monkeypatch_module.setenv("SIMULATION__MANIFEST_PATH", str(data / "manifest.json"))
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

    rest = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 200}).json()
    assert rest["snapshot"]["finished"] is True
    assert rest["snapshot"]["time_s"] == 14_400
    assert len(rest["decisions"]) == 14_400 // 120 - 1

    finished = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 1})
    assert finished.status_code == 409
    assert client.delete(f"/simulations/{simulation_id}").status_code == 204
    assert client.get(f"/simulations/{simulation_id}").status_code == 404


def test_errors_map_to_http_status(client):
    assert client.post("/simulations", json={"controller": "nope"}).status_code == 404
    assert client.post("/simulations", json={"scenario_index": 5}).status_code == 422
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
