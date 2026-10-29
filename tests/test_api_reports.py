"""Shift report API (SV-2.3, #295): list by date and shift, a report with data, tables, narrative and status, the live
report queued as a job (202), roles, and units in the schemas; on a SQLite twin store and an in-memory job queue."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import store as store_service
from services.jobs import JobStore
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie

IST = "Asia/Kolkata"


class MemoryRedis:
    """The Redis commands ``JobStore`` uses, in memory."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.sorted: dict[str, dict[str, float]] = {}

    async def set(self, key: str, value: Any, nx: bool = False, ex: int | None = None) -> bool | None:
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def get(self, key: str) -> Any:
        return self.values.get(key)

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self.sorted.setdefault(key, {}).update(mapping)
        return len(mapping)


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    yield store
    store_service.reset()
    store.dispose()


@pytest.fixture()
def client(twin):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        app.state.live.store = JobStore(MemoryRedis())
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def call(client: TestClient, method: str, user: str | None, url: str, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.request(method, url, headers=headers, **kw)


def ist(text: str) -> datetime:
    return pd.Timestamp(text, tz=IST).tz_convert("UTC").to_pydatetime()


def report_row(key: str, generated: str, **values: Any) -> dict[str, Any]:
    day, letter = key.split("_")
    start = ist(f"{day} {dict(A='06:00', B='14:00', C='22:00')[letter]}")
    stats = {"mean": 231000.0, "std": 2500.0}
    temps = {"q1": 180.0, "q2": 182.0, "q3": 179.0, "q4": 185.0, "spread_std": 2.65}
    data = {
        "shift_date": day,
        "shift_label": letter,
        "shift_start_ist": pd.Timestamp(start).tz_convert(IST).isoformat(),
        "shift_end_ist": (pd.Timestamp(start) + pd.Timedelta(hours=8)).tz_convert(IST).isoformat(),
        "report_type": "Shift",
        "window_id": f"{day}_SHIFT_{letter}",
        "duration_h": 8.0,
        "status": "STABLE",
        "status_flags": [],
        "production_rate": 112.4,
        "fuel_rate": 528.4,
        "total_charges": 46,
        **{k: stats for k in ("blast_volume", "blast_temp", "blast_pressure", "furnace_top_dp", "furnace_bottom_dp",
                              "furnace_total_dp", "o2_flow", "o2_enrichment", "permeability", "etaco", "raft")},
        **{k: temps for k in ("uptake", "lower_stack", "belly", "skin_flow", "bosh")},
        "used_materials": {"Flux": "- Dolomite - 36.80 mt"},
    }  # fmt: skip
    return {
        "shift_key": key,
        "shift_start": start,
        "shift_end": start + timedelta(hours=8),
        "kind": "closed",
        "trigger": "schedule",
        "requested_by": "scheduler",
        "generated_at": ist(generated),
        "furnace_status": "STABLE",
        "production_rate_t_h": 112.4,
        "fuel_rate_kg_thm": 528.4,
        "etaco_pct": 47.1,
        "narrative_status": "ok",
        "narrative": "Stable shift; watch the belly spread.",
        "narrative_tries": 1,
        "narrative_model": "fake/model-1",
        "data": data,
        "document": {
            "title": f"Live Report ({day}_SHIFT_{letter})",
            "pre_blocks": ["**SHIFT HANDOVER REPORT - BF2 EVONITH STEEL**"],
            "sections": [
                {"title": "Fuel Rate", "headers": ["Fuel rate (kg/thm)"], "rows": [["528.40"]], "placement": "left"}
            ],  # fmt: skip
            "notes": [{"text": "**Report Analysis**\n\nStable shift.", "placement": "right", "kind": "analysis"}],
            "generated_at_ist": pd.Timestamp(ist(generated)).tz_convert(IST).isoformat(),
        },
        **values,
    }


def test_reports_are_listed_by_date_and_shift(twin, client) -> None:
    twin.reports.add(report_row("2026-10-26_B", "2026-10-26 22:21"))
    twin.reports.add(report_row("2026-10-26_C", "2026-10-27 06:21"))  # dated the 26th: shift C starts at 22:00
    twin.reports.add(report_row("2026-10-27_A", "2026-10-27 14:21"))
    twin.reports.add(report_row("2026-10-27_A", "2026-10-27 11:00", kind="live", trigger="user", requested_by="op"))
    every = call(client, "GET", "op", "/api/reports").json()
    expected = [("2026-10-27_A", "closed"), ("2026-10-27_A", "live"), ("2026-10-26_C", "closed"),
                ("2026-10-26_B", "closed")]  # fmt: skip
    assert [(r["shift_key"], r["kind"]) for r in every] == expected
    day = call(client, "GET", "op", "/api/reports", params={"date": "2026-10-26"}).json()
    assert [r["shift_key"] for r in day] == ["2026-10-26_C", "2026-10-26_B"]
    one = call(client, "GET", "op", "/api/reports", params={"date": "2026-10-27", "shift": "A", "kind": "live"}).json()
    assert len(one) == 1 and one[0]["requested_by"] == "op" and one[0]["shift"] == "A"
    first = every[0]
    assert first["shift_date"] == "2026-10-27" and first["status"] == "complete" and first["fuel_rate_kg_thm"] == 528.4
    assert pd.Timestamp(first["shift_start"]) == pd.Timestamp(ist("2026-10-27 06:00"))
    assert "data" not in first and "narrative" not in first  # the list carries the headline only
    assert call(client, "GET", "op", "/api/reports", params={"shift": "D"}).status_code == 422


def test_a_report_has_its_data_tables_narrative_and_status(twin, client) -> None:
    ok = twin.reports.add(report_row("2026-10-27_A", "2026-10-27 14:21"))
    failed = twin.reports.add(
        report_row(
            "2026-10-27_B",
            "2026-10-27 22:22",
            narrative_status="failed",
            narrative=None,
            narrative_tries=2,
            narrative_error="try 1: RuntimeError: 503; try 2: RuntimeError: 503",
        )
    )
    body = call(client, "GET", "op", f"/api/reports/{ok.id}").json()
    assert body["status"] == "complete" and body["narrative"].startswith("Stable shift")
    assert body["data"]["blast_volume"] == {"mean": 231000.0, "std": 2500.0}
    assert body["data"]["uptake"]["spread_std"] == 2.65 and body["data"]["duration_h"] == 8.0
    assert body["document"]["sections"][0]["rows"] == [["528.40"]]
    without = call(client, "GET", "sup", f"/api/reports/{failed.id}").json()
    assert without["status"] == "no_narrative" and without["narrative"] is None
    assert (
        without["narrative_status"] == "failed"
        and without["narrative_tries"] == 2
        and "503" in without["narrative_error"]
    )
    missing = call(client, "GET", "op", f"/api/reports/{uuid.uuid4()}")
    assert missing.status_code == 404 and missing.headers["content-type"].startswith("application/problem+json")
    assert call(client, "GET", "op", "/api/reports/not-an-id").status_code == 404


def test_the_live_report_is_queued_as_a_job(twin, client) -> None:
    r = call(client, "POST", "op", "/api/reports/live")
    assert r.status_code == 202
    job = r.json()["job"]
    assert job["kind"] == "report.live" and job["status"] == "queued" and job["submitted_by"] == "op"
    assert job["params"] == {"narrative": True} and r.json()["duplicate"] is False
    again = call(client, "POST", "sup", "/api/reports/live", json={"narrative": True}).json()
    assert again["duplicate"] and again["job"]["id"] == job["id"]  # one live report at a time
    other = call(client, "POST", "op", "/api/reports/live", json={"narrative": False}).json()
    assert not other["duplicate"] and other["job"]["params"] == {"narrative": False}
    assert call(client, "POST", "op", "/api/reports/live", json={"shift": "A"}).status_code == 422

    pending = call(client, "GET", "op", f"/api/reports/{job['id']}")  # by job id, before the worker has run
    assert pending.status_code == 404 and "queued" in pending.json()["detail"]
    row = twin.reports.add(report_row("2026-10-27_B", "2026-10-27 16:00", kind="live", trigger="user"))
    jobs = client.app.state.live.store  # type: ignore[attr-defined]
    asyncio.run(jobs.finish(job["id"], "done", result={"report_id": str(row.id)}))  # the in-memory queue has no loop
    done = call(client, "GET", "op", f"/api/reports/{job['id']}").json()
    assert done["id"] == str(row.id) and done["kind"] == "live"


def test_the_reports_need_a_session_and_a_store(client) -> None:
    assert call(client, "GET", None, "/api/reports").status_code == 401
    assert call(client, "POST", None, "/api/reports/live").status_code == 401
    store_service.use(None)
    r = call(client, "GET", "op", "/api/reports")
    assert r.status_code == 503 and "BF_TWIN_DB_URL" in r.json()["detail"]


def test_every_numeric_field_of_the_report_schemas_states_its_unit(client) -> None:
    schemas = client.get("/api/openapi.json").json()["components"]["schemas"]
    names = ("ShiftReportSummary", "ShiftReportDetail", "ShiftReportDataView", "TempRowView", "ParamStatsView")
    missing = []
    for name in names:
        for field, spec in schemas[name]["properties"].items():
            kinds = {s.get("type") for s in spec.get("anyOf", [spec])}
            if kinds & {"number", "integer"} and not spec.get("description"):
                missing.append(f"{name}.{field}")
    assert not missing, missing
    paths = client.get("/api/openapi.json").json()["paths"]
    assert set(paths["/api/reports/live"]) == {"post"} and "202" in paths["/api/reports/live"]["post"]["responses"]


# ---------------------------------------------------------------------------------- Shift Reports page (SV-2.4, #296)
def test_the_thresholds_are_served_from_the_configuration(client, monkeypatch) -> None:
    import config.config_loader as loader

    body = call(client, "GET", "op", "/api/reports/thresholds").json()
    assert body == {
        "etaco_warn_pct": 42.0,
        "etaco_crit_pct": 40.0,
        "fuel_rate_warn_kg_thm": 530.0,
        "fuel_rate_crit_kg_thm": 570.0,
        "raft_lo_c": 2100.0,
        "raft_hi_c": 2350.0,
        "permeability_lo": 1000.0,
        "permeability_hi": 1600.0,
        "attention_from": 1,
        "unstable_from": 3,
    }
    shipped = loader.load_config("shift_report.yml")
    changed = {
        **shipped,
        "status_thresholds": {**shipped["status_thresholds"], "fuel_rate": {"warn": 520, "crit": 560}},
    }
    monkeypatch.setattr(loader, "load_config", lambda name="": changed)
    assert call(client, "GET", "op", "/api/reports/thresholds").json()["fuel_rate_warn_kg_thm"] == 520.0
    monkeypatch.setattr(loader, "load_config", lambda name="": {"status_thresholds": {}})
    broken = call(client, "GET", "op", "/api/reports/thresholds")
    assert broken.status_code == 500 and broken.headers["content-type"].startswith("application/problem+json")
    assert call(client, "GET", None, "/api/reports/thresholds").status_code == 401


def test_a_report_carries_its_kpis_against_the_thresholds(twin, client) -> None:
    row = report_row("2026-10-27_A", "2026-10-27 14:21")
    row["data"] = {**row["data"], "fuel_rate": 548.7, "etaco": {"mean": 46.3, "std": 0.8}}
    row["data"]["raft"] = {"mean": 2380.5, "std": 20.0}
    row["data"]["permeability"] = {"mean": 1320.0, "std": 60.0}
    added = twin.reports.add(row)
    body = call(client, "GET", "op", f"/api/reports/{added.id}").json()
    kpis = {k["key"]: k for k in body["kpis"]}
    assert [k["key"] for k in body["kpis"]] == ["etaco", "fuel_rate", "raft", "permeability"]
    assert kpis["fuel_rate"]["level"] == "warning" and kpis["fuel_rate"]["unit"] == "kg/tHM"
    assert kpis["fuel_rate"]["flag"] == "Fuel Rate 548.70 kg/tHM (> 530 attention)"
    assert kpis["etaco"]["level"] == "ok" and kpis["etaco"]["value"] == 46.3 and kpis["etaco"]["crit"] == 40.0
    assert kpis["raft"]["level"] == "warning" and (kpis["raft"]["lo"], kpis["raft"]["hi"]) == (2100.0, 2350.0)
    assert kpis["permeability"]["level"] == "ok" and kpis["permeability"]["unit"] == ""
    schemas = client.get("/api/openapi.json").json()["components"]["schemas"]
    for name in ("KpiStatusView", "ReportThresholds"):
        for field, spec in schemas[name]["properties"].items():
            kinds = {s.get("type") for s in spec.get("anyOf", [spec])}
            assert not (kinds & {"number", "integer"}) or spec.get("description"), f"{name}.{field}"
