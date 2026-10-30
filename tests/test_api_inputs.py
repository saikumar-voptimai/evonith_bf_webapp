"""Live inputs API (SV-7.3, #326): ``GET /api/inputs/live`` serves the four groups in one response with the snapshot
id, the as-of times, the stale flags and the limits of ``config/live_inputs.yml``; any number of requests within a
group's TTL cause one upstream read; the role; units on every numeric field.

Neon and InfluxDB are the counting fakes of ``tests/test_services_live_inputs.py``; the operator inputs the YAML file.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from services import live_inputs as li
from services.live_inputs import LiveInputsView
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_services_live_inputs import (
    NOW,
    T_SAMPLE,
    T_STOCK,
    CountingFetcher,
    FakeProvider,
    counting_influx,
    install_fakes,
    reset_caches,
)

URL = "/api/inputs/live"
UNITS = ("s", "min", "d", "t", "%", "count", "g/Nm³", "in unit")
SCHEMAS = (
    "LiveInputsView",
    "LiveInputsRef",
    "GroupLimits",
    "StockGroup",
    "StockMaterial",
    "RmGroup",
    "RmMaterial",
    "ProcessGroup",
    "ProcessValue",
    "BlastMoisture",
    "OperatorInputsGroup",
)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    install_fakes(monkeypatch)
    yield
    reset_caches()


@pytest.fixture()
def fetcher(monkeypatch) -> CountingFetcher:
    return counting_influx(monkeypatch)


@pytest.fixture()
def client():
    app = build(redis_url=None)
    with TestClient(app, base_url="https://testserver") as c:
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def get(client: TestClient, user: str, url: str = URL, **kw: Any):
    return client.get(url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_the_route_needs_a_session(client) -> None:
    r = client.get(URL)
    assert r.status_code == 401 and r.headers["content-type"].startswith("application/problem+json")


def test_the_four_groups_in_one_response_with_the_snapshot_id_and_the_limits(client, fetcher) -> None:
    r = get(client, "op")
    assert r.status_code == 200, r.text
    view = LiveInputsView.model_validate(r.json())
    body = r.json()
    assert set(body) == {"snapshot_id", "read_at", "stale", *li.GROUP_KEYS}
    assert body["stale"] == [] and _at(body["read_at"]) == NOW
    assert view.snapshot_id == li.snapshot_id([view.stock.as_of, view.rm_composition.as_of,
                                               view.process_conditions.as_of, view.operator_inputs.as_of])  # fmt: skip
    # The limits of config/live_inputs.yml are echoed per group: the SPA holds no copy.
    assert {k: body[k]["limits"] for k in li.GROUP_KEYS} == {
        k: {"ttl_s": s.ttl_s, "stale_after_min": s.stale_after_min} for k, s in li.SPECS.items()
    }
    assert {k: body[k]["label"] for k in li.GROUP_KEYS} == {k: s.label for k, s in li.SPECS.items()}
    # Each group: source, as-of time and the values with their units.
    stock = body["stock"]
    assert _at(stock["as_of"]) == _at(T_STOCK) and stock["source"] == "Neon offline_feed.raw_material_stock"
    assert {m["id"]: m["stock_t"] for m in stock["materials"]}["nmdc_donimalai"] == 1250.0
    assert stock["total_t"] == 1890.0
    rm = body["rm_composition"]
    assert _at(rm["as_of"]) == _at(T_SAMPLE) and (rm["chemistry_mode"], rm["window_days"]) == ("latest", 30)
    assert {m["id"]: m["fe_t_pct"] for m in rm["materials"]}["nmdc_donimalai"] == 63.4
    process = body["process_conditions"]
    first = process["parameters"][0]
    assert (first["key"], first["unit"], first["label"]) == ("hot_blast_volume", "Nm³/h", "Hot Blast Volume")
    assert first["value"] is not None and process["telemetry"] in ("live", "partial")
    assert process["blast_moisture"]["value_g_nm3"] > 0 and process["blast_moisture"]["day"] == "2026-10-15"
    inputs = body["operator_inputs"]
    assert inputs["source"].startswith("YAML file") and inputs["version"] is None and inputs["inputs"]["ore_editor"]


def test_requests_within_the_ttl_share_one_upstream_read(client, fetcher) -> None:
    first = get(client, "op").json()
    second = get(client, "sup").json()
    assert first["snapshot_id"] == second["snapshot_id"]
    assert {k: second[k]["read_at"] for k in li.GROUP_KEYS} == {k: first[k]["read_at"] for k in li.GROUP_KEYS}
    assert FakeProvider.calls == {"stock": 1, "chemistry": 1} and fetcher.snapshot_queries == 1


def test_the_chemistry_mode_and_window_of_the_request(client, fetcher) -> None:
    r = get(client, "op", params={"chemistry_mode": "avg", "chemistry_window_days": 14})
    assert r.status_code == 200, r.text
    rm = r.json()["rm_composition"]
    assert (rm["chemistry_mode"], rm["window_days"]) == ("avg", 14)
    assert rm["materials"][0]["source"] == "offline_db_avg_non_zero"
    assert get(client, "op").json()["rm_composition"]["chemistry_mode"] == "latest"
    assert FakeProvider.calls["chemistry"] == 2 and FakeProvider.calls["stock"] == 1
    assert get(client, "op", params={"chemistry_mode": "mean"}).status_code == 422
    assert get(client, "op", params={"chemistry_window_days": 0}).status_code == 422


def test_groups_older_than_their_limit_are_flagged(client, fetcher, monkeypatch) -> None:
    monkeypatch.setattr(li, "clock", lambda: NOW + timedelta(days=2))
    body = get(client, "op").json()
    # Stock (12 h) and the process readings (15 min) are past their limits; the composition (7 d) and the operator
    # inputs (never by age) are not.
    assert body["stale"] == ["stock", "process_conditions"]
    assert body["stock"]["stale"] and "limit 12 h" in body["stock"]["stale_reason"]
    assert body["process_conditions"]["stale"] and "limit 0.25 h" in body["process_conditions"]["stale_reason"]
    assert not body["rm_composition"]["stale"] and not body["operator_inputs"]["stale"]
    assert body["stock"]["age_min"] == pytest.approx(2 * 24 * 60 + (NOW - _at(T_STOCK)).total_seconds() / 60)

    li.clear()
    FakeProvider.fail_stock = True
    failed = get(client, "op").json()
    assert failed["stock"]["error"].startswith("Stock query failed") and failed["stock"]["as_of"] is None
    assert "stock" in failed["stale"] and failed["stock"]["stale_reason"].startswith("The last read failed")


@pytest.mark.parametrize("schema", SCHEMAS)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = []
    for name, prop in spec.get("properties", {}).items():
        types = {prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}
        if not types & {"number", "integer"}:
            continue
        description = prop.get("description") or ""
        if not ("," in description and description.rsplit(",", 1)[-1].strip() in UNITS):
            missing.append(f"{name}: {description!r}")
    assert not missing, missing


def test_the_run_results_carry_the_snapshot_reference_in_the_schema() -> None:
    schemas = build().openapi()["components"]["schemas"]
    assert "LiveInputsRef" in str(schemas["BmoRunResult"]["properties"]["live_inputs"])
    assert "LiveInputsRef" in str(schemas["RunCase"]["properties"]["live_inputs"])
