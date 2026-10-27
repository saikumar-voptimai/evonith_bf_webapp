"""Furnace status API (SV-2.1, #293): the snapshot route equals the Streamlit V-Board for the same timestamp, answers
from the worker's live snapshot while it is fresh, takes one InfluxDB snapshot query per 10-s slot for any number of
requests otherwise; trends, their RFC 7807 errors, the role and the units of every numeric field.

InfluxDB is the counting fake of ``tests/test_services_status.py``; Redis the in-memory fake of
``tests/test_services_live_status.py``. No plant source is reached.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from data import furnace_status as fs
from data.furnace_status import repository as fs_repository
from services import live_status, status
from services.events import LiveStatusData, TagValue, WorkerStatus
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_services_live_status import FakeRedis
from tests.test_services_status import NOW, CountingFetcher, assert_snapshot_equals_page, plant_frames


@pytest.fixture(autouse=True)
def _fresh_caches():
    status.clear_cache()
    yield
    status.clear_cache()


@pytest.fixture()
def fetcher(monkeypatch) -> CountingFetcher:
    fake = CountingFetcher(plant_frames(), delay_s=0.05)
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", fake)
    monkeypatch.setattr(status, "clock", lambda: NOW)
    return fake


@pytest.fixture()
def app():
    return build(redis_url=None)


@pytest.fixture()
def client(app):
    with TestClient(app, base_url="https://testserver") as c:
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def get(client: TestClient, user: str, url: str, **kw: Any):
    return client.get(url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


# ---------------------------------------------------------------- role


def test_status_routes_need_a_session(client) -> None:
    for url in ("/api/status/snapshot", "/api/status/trend?parameters=raft"):
        r = client.get(url)
        assert r.status_code == 401 and r.headers["content-type"].startswith("application/problem+json"), url


# ---------------------------------------------------------------- snapshot


def test_snapshot_route_equals_the_streamlit_page(client, fetcher) -> None:
    r = get(client, "op", "/api/status/snapshot")
    assert r.status_code == 200, r.text
    view = status.StatusSnapshotView.model_validate(r.json())
    page = fs.load_status_snapshot(NOW)  # what the V-Board renders for the same timestamp
    assert_snapshot_equals_page(view, page)
    assert view.freshness.source == "request" and view.freshness.computed_at == NOW
    by_key = {p["key"]: p for p in r.json()["parameters"]}
    assert by_key["raft"]["state"] == "low" and by_key["raft"]["limits"]["low"] == 2100.0
    assert by_key["blast_pressure"]["state"] == "high" and by_key["hot_blast_volume"]["unit"] == "Nm³/h"
    assert r.json()["operating_state"]["state"] == "normal"
    assert len(r.json()["parameters"]) == len(fs.PARAMETERS)


def test_many_requests_in_a_slot_take_one_snapshot_query(client, fetcher) -> None:
    """Acceptance criterion on the API path (no worker snapshot): 24 concurrent and 12 later requests of three users
    within one 10-s slot, one InfluxDB snapshot query."""
    users = [u for u in PASSWORDS for _ in range(8)]
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda u: get(client, u, "/api/status/snapshot").status_code, users))
    assert codes == [200] * 24
    fs_repository.clear_cache()  # past the 5-s request cache: only the slot keeps the count at one
    for user in list(PASSWORDS) * 4:
        assert get(client, user, "/api/status/snapshot").status_code == 200
    assert fetcher.snapshot_queries == 1


def test_snapshot_comes_from_the_worker_while_fresh(app, client, fetcher, monkeypatch) -> None:
    redis = FakeRedis()
    app.state.live.redis = redis  # as after a successful connect
    produced = status.snapshot_view(fs.load_status_snapshot(NOW), NOW - timedelta(seconds=5), source="producer")
    fs_repository.clear_cache()
    fetcher.calls.clear()
    asyncio.run(live_status.write_latest(redis, produced))

    for _ in range(5):
        r = get(client, "op", "/api/status/snapshot")
        assert r.status_code == 200 and r.json()["freshness"]["source"] == "producer"
    assert status.StatusSnapshotView.model_validate(r.json()) == produced and fetcher.calls == []

    stale = produced.model_copy(
        update={"freshness": produced.freshness.model_copy(update={"computed_at": NOW - timedelta(seconds=30)})}
    )
    asyncio.run(live_status.write_latest(redis, stale))  # the worker stopped producing 30 s ago
    r = get(client, "op", "/api/status/snapshot")
    assert r.json()["freshness"]["source"] == "request" and fetcher.snapshot_queries == 1

    async def down(key):
        raise ConnectionError("redis down")

    monkeypatch.setattr(redis, "get", down)  # Redis not reachable: the route takes (here: reuses) its own snapshot
    r = get(client, "op", "/api/status/snapshot")
    assert r.status_code == 200 and r.json()["freshness"]["source"] == "request" and fetcher.snapshot_queries == 1


# ---------------------------------------------------------------- trend


def test_trend_route_returns_the_service_trend(client, fetcher) -> None:
    r = get(client, "op", "/api/status/trend", params={"parameters": "pci_rate, raft,pci_rate", "interval": "1h"})
    assert r.status_code == 200, r.text
    body = status.StatusTrend.model_validate(r.json())
    expected = status.build_trend(["pci_rate", "raft"], "1h", now=NOW)
    assert body == expected
    assert [s.key for s in body.series] == ["pci_rate", "raft"] and body.series[0].raw_points == 61
    assert body.max_points == status.DEFAULT_TREND_POINTS and body.interval == "1h"

    r = get(
        client,
        "sup",
        "/api/status/trend",
        params={
            "parameters": "raft",
            "interval": "custom",
            "start": "2026-10-26T06:00:00",
            "end": "2026-10-26T14:00:00",
            "max_points": 200,
        },
    )
    assert r.status_code == 200, r.text
    (series,) = r.json()["series"]
    assert r.json()["start"].startswith("2026-10-26T00:30:00") and series["raw_points"] == 481
    assert len(series["times"]) <= 200


@pytest.mark.parametrize(
    ("params", "detail"),
    [
        ({"parameters": " , "}, "Name at least one parameter."),
        ({"parameters": ",".join(f"p{i}" for i in range(9))}, "At most 8 parameters per request."),
        ({"parameters": "raft,nope"}, "Unknown parameter: nope."),
        ({"parameters": "raft", "interval": "custom"}, "A custom interval needs start and end."),
        (
            {"parameters": "raft", "interval": "custom", "start": "2026-10-18T00:00", "end": "2026-10-26T00:00"},
            "at most 7 days",
        ),
        ({"parameters": "raft", "interval": "2h"}, "The request does not match the schema."),
        ({"parameters": "raft", "max_points": 50}, "The request does not match the schema."),
        ({}, "The request does not match the schema."),
    ],
)
def test_trend_errors_are_problem_details(client, fetcher, params, detail) -> None:
    r = get(client, "op", "/api/status/trend", params=params)
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")
    assert detail in r.json()["detail"]
    assert fetcher.calls == []


# ---------------------------------------------------------------- units

UNITS = ("unit", "m", "s", "min", "count", "fraction")


def _numeric(prop: dict[str, Any]) -> bool:
    options = [prop, *prop.get("anyOf", [])]
    items = [o["items"] for o in options if o.get("type") == "array" and isinstance(o.get("items"), dict)]
    types = {o.get("type") for o in options} | {i.get("type") for item in items for i in [item, *item.get("anyOf", [])]}
    return bool(types & {"number", "integer"})


def _states_unit(description: str | None) -> bool:
    """The last comma clause names the unit: a unit, or the parameter's unit (``unit`` field of the object)."""
    if not description or "," not in description:
        return False
    clause = description.rsplit(",", 1)[-1].strip()
    if clause.startswith(("in unit", "in the parameter's unit", "in the unit of the deciding parameter")):
        return True
    return clause.split(" ")[0] in UNITS


@pytest.mark.parametrize(
    "schema",
    [
        "StatusLimits",
        "StatusParameter",
        "StatusZone",
        "StatusOperatingState",
        "StatusFreshness",
        "StatusSnapshotView",
        "StatusTrendStats",
        "StatusTrendSeries",
        "StatusTrend",
    ],
)
def test_every_numeric_status_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = [
        f"{name}: {prop.get('description')!r}"
        for name, prop in spec.get("properties", {}).items()
        if _numeric(prop) and not _states_unit(prop.get("description"))
    ]
    assert not missing, missing


@pytest.mark.parametrize("model", [TagValue, LiveStatusData, WorkerStatus])
def test_every_numeric_live_status_field_states_its_unit(model) -> None:
    props = model.model_json_schema()["properties"]
    missing = [
        f"{name}: {prop.get('description')!r}"
        for name, prop in props.items()
        if _numeric(prop) and not _states_unit(prop.get("description"))
    ]
    assert not missing, missing


def test_trend_query_parameters_are_documented() -> None:
    operation = build().openapi()["paths"]["/api/status/trend"]["get"]
    by_name = {p["name"]: p for p in operation["parameters"]}
    assert set(by_name) == {"parameters", "interval", "start", "end", "max_points"}
    assert by_name["interval"]["schema"]["enum"] == ["1h", "4h", "8h", "16h", "24h", "custom"]
    assert by_name["max_points"]["description"].split(",")[-1].strip().startswith("count")
