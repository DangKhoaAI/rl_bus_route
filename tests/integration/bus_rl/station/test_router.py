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


def test_burst_surfaces_only_once_it_has_started(client):
    created = client.post("/simulations", json={"split": "test_ood_burst"}).json()
    simulation_id = created["simulation_id"]
    try:
        assert created["burst"] is None
        snapshot = client.post(
            f"/simulations/{simulation_id}/steps", json={"decisions": 200}
        ).json()["snapshot"]
        burst = snapshot["burst"]
        assert burst["start_s"] < burst["end_s"] <= snapshot["time_s"]
        # 5–10 minutes of 8–20 extra passengers per tick.
        assert 8 * 5 <= burst["added"]
        assert burst["direction"] in (1, -1)
    finally:
        client.delete(f"/simulations/{simulation_id}")
    base = client.post("/simulations", json={"split": "validation"}).json()
    client.delete(f"/simulations/{base['simulation_id']}")
    assert base["burst"] is None


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


def test_idle_session_is_evicted_when_the_limit_is_reached(client, monkeypatch):
    from time import monotonic

    import bus_rl.station.service as service_module

    idle = client.post("/simulations", json={}).json()["simulation_id"]
    # Jump an hour ahead: every session already open is now idle past the timeout.
    monkeypatch.setattr(service_module, "monotonic", lambda: monotonic() + 3600)
    fresh = [client.post("/simulations", json={}) for _ in range(2)]
    try:
        assert [response.status_code for response in fresh] == [201, 201]
        assert client.get(f"/simulations/{idle}").status_code == 404
    finally:
        for response in fresh:
            client.delete(f"/simulations/{response.json()['simulation_id']}")


def _commands(client, controller):
    created = client.post("/simulations", json={"controller": controller}).json()
    simulation_id = created["simulation_id"]
    try:
        step = client.post(f"/simulations/{simulation_id}/steps", json={"decisions": 200})
        return step.json()["decisions"]
    finally:
        client.delete(f"/simulations/{simulation_id}")


def test_commands_carry_the_state_they_were_decided_from(client):
    decisions = _commands(client, "threshold")
    commands = [d for d in decisions if d["kind"] != "NOOP"]
    assert commands
    assert all(d["context"] is None for d in decisions if d["kind"] == "NOOP")
    for command in commands:
        context = command["context"]
        assert [route["route_id"] for route in context["routes"]] == [0, 1, 2]
        # Recalls can park more than the three planned reserves.
        assert 0 <= context["reserve"] <= 12
        assert context["legal_actions"] >= 1
        # A rule controller has no probabilities; its reason is the rule itself.
        assert context["top_choices"] == []


def test_ppo_commands_list_the_policy_top_choices(client):
    if "ppo" not in client.get("/station").json()["controllers"]:
        pytest.skip("no PPO checkpoint configured")
    commands = [d for d in _commands(client, "ppo") if d["kind"] != "NOOP"]
    assert commands
    for command in commands:
        choices = command["context"]["top_choices"]
        probabilities = [choice["probability"] for choice in choices]
        # Deterministic predict takes the most likely legal action.
        assert choices[0]["action_index"] == command["action_index"]
        assert probabilities == sorted(probabilities, reverse=True)
        assert 0 < sum(probabilities) <= 1.0001
