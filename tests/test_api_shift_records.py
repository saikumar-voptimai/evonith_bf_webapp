"""Shift handover record API (SV-2.5, #297): list and detail of the frozen records, edits refused (no write route),
the record linked from its report, roles and units; on a SQLite twin store."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from data.twin.repositories import Actor
from services import shift_records
from tests.test_api_reports import call, client, ist, report_row, twin  # noqa: F401 - fixtures
from tests.test_services_shift_records import C_END, C_START, run_row, snapshot

H = timedelta(hours=1)


def freeze(store, key: str) -> None:
    store.shifts.freeze(
        shift_records.record_values(store, shift_records.Shift.parse(key), closed_at=ist("2026-10-27 06:11"))
    )


@pytest.fixture()
def frozen(twin):  # noqa: F811
    for i in range(1, 9):
        twin.snapshots.upsert(snapshot(C_START + i * H, 400.0 + i))
    run = twin.optimiser.add(run_row(C_START + H))
    twin.optimiser.decide(run.id, "accepted", Actor("sv1", "supervisor"), at=C_START + 2 * H, comment="go")
    freeze(twin, "2026-10-26_C")
    freeze(twin, "2026-10-26_B")  # no snapshots, no decision
    return run


def test_records_are_listed_newest_first_and_read_with_their_snapshots(twin, client, frozen) -> None:  # noqa: F811
    rows = call(client, "GET", "op", "/api/shift-records").json()
    assert [r["shift_key"] for r in rows] == ["2026-10-26_C", "2026-10-26_B"]
    c = rows[0]
    assert c["k_trace"] == 408.0 and c["windows_stored"] == 8 and c["windows_expected"] == 8 and c["shift_window"]
    assert c["plan_decision"] == "accepted" and c["plan_run_id"] == str(frozen.id)
    assert c["report_ref"] == "/reports?date=2026-10-26&shift=C" and c["href"] == "/api/shift-records/2026-10-26_C"
    assert pd.Timestamp(c["closed_at"]) == pd.Timestamp(ist("2026-10-27 06:11")) and "hourly" not in c
    assert [r["shift_key"] for r in call(client, "GET", "op", "/api/shift-records", params={"shift": "B"}).json()] == [
        "2026-10-26_B"
    ]
    assert call(client, "GET", "op", "/api/shift-records", params={"date": "2026-10-27"}).json() == []

    body = call(client, "GET", "op", "/api/shift-records/2026-10-26_C").json()
    assert len(body["hourly"]) == 8 and body["missing_windows"] == []
    assert body["shift_window_snapshot"]["k_trace"] == 408.0
    assert pd.Timestamp(body["shift_window_snapshot"]["window_end"]) == pd.Timestamp(C_END)
    assert body["plan"]["decision"]["comment"] == "go" and body["plan"]["run"]["cost_rs_thm"] == 24810.5
    assert body["frozen_by"] is None and body["report_id"] is None  # no report.shift yet
    empty = call(client, "GET", "op", "/api/shift-records/2026-10-26_B").json()
    assert empty["shift_window_snapshot"] is None and len(empty["missing_windows"]) == 8 and empty["plan"] is None

    report = twin.reports.add(report_row("2026-10-26_C", "2026-10-27 06:21"))
    twin.reports.add(report_row("2026-10-26_C", "2026-10-27 05:00", kind="live", trigger="user"))
    assert call(client, "GET", "op", "/api/shift-records/2026-10-26_C").json()["report_id"] == str(report.id)

    missing = call(client, "GET", "op", "/api/shift-records/2026-10-27_A")
    assert missing.status_code == 404 and missing.headers["content-type"].startswith("application/problem+json")
    assert call(client, "GET", "op", "/api/shift-records/2026-10-27_D").status_code == 422


def test_the_api_refuses_edits(twin, client, frozen) -> None:  # noqa: F811
    for method in ("PUT", "PATCH", "DELETE", "POST"):
        r = call(client, method, "adm", "/api/shift-records/2026-10-26_C", json={"k_trace": 1.0})
        assert r.status_code == 405, method
        assert r.headers["content-type"].startswith("application/problem+json")
    assert call(client, "POST", "adm", "/api/shift-records", json={}).status_code == 405
    assert call(client, "GET", "op", "/api/shift-records/2026-10-26_C").json()["k_trace"] == 408.0
    paths = client.get("/api/openapi.json").json()["paths"]
    assert set(paths["/api/shift-records"]) == {"get"} and set(paths["/api/shift-records/{shift_key}"]) == {"get"}


def test_a_report_links_the_record_of_its_shift(twin, client, frozen) -> None:  # noqa: F811
    report = twin.reports.add(report_row("2026-10-26_C", "2026-10-27 06:21"))
    body = call(client, "GET", "op", f"/api/reports/{report.id}").json()
    assert body["record"]["shift_key"] == "2026-10-26_C" and body["record"]["href"] == "/api/shift-records/2026-10-26_C"
    assert pd.Timestamp(body["record"]["closed_at"]) == pd.Timestamp(ist("2026-10-27 06:11"))
    later = twin.reports.add(report_row("2026-10-27_A", "2026-10-27 14:21"))
    assert call(client, "GET", "op", f"/api/reports/{later.id}").json()["record"] is None


def test_the_records_need_a_session_and_state_their_units(client) -> None:  # noqa: F811
    assert call(client, "GET", None, "/api/shift-records").status_code == 401
    assert call(client, "GET", None, "/api/shift-records/2026-10-26_C").status_code == 401
    schemas = client.get("/api/openapi.json").json()["components"]["schemas"]
    missing = []
    for name in ("ShiftRecordSummary", "ShiftRecordDetail", "FrozenRun", "ShiftRecordLink"):
        for field, spec in schemas[name]["properties"].items():
            kinds = {s.get("type") for s in spec.get("anyOf", [spec])}
            if kinds & {"number", "integer"} and not spec.get("description"):
                missing.append(f"{name}.{field}")
    assert not missing, missing
