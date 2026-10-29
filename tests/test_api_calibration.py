"""Calibration console API (SV-4.6, #309): administrators only; the console's calibration equals
``data.burden_engine.calibration.live_calibration`` on the same window (relative difference <= 1e-6, through JSON);
a manual recalibration queues ``calibration.nightly`` and is audited, also through ``POST /api/jobs``; numeric fields
state their unit. SQLite twin store, in-memory Redis."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from api.routers.calibration import CalibrationConsole
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import day_calibration, job_audit
from services import store as store_service
from services.jobs import JobStore
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_bmo import IndexedRedis
from tests.test_api_live import session_cookie
from tests.test_services_calibration_console import DAY, FIELDS, VALUES, nightly, reference
from utils.burden_engine import strata_network

URL = "/api/admin/calibration"
RUN = "/api/admin/calibration/run"
KIND = "calibration.nightly"


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
def folders(tmp_path, monkeypatch):
    cal, jac = tmp_path / "calibration", tmp_path / "strata_flow"
    monkeypatch.setattr(day_calibration, "STORE_DIR", cal)
    monkeypatch.setattr(strata_network, "CACHE_DIR", jac)
    monkeypatch.setattr(day_calibration, "today", lambda: DAY)
    return cal, jac


@pytest.fixture()
def client(twin, folders):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        app.state.live.store = JobStore(IndexedRedis())
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def call(client: TestClient, method: str, user: str | None, url: str, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.request(method, url, headers=headers, **kw)


# ------------------------------------------------------------------------------------------------------- roles

ROLE_MATRIX = {None: 401, "op": 403, "sup": 403}


@pytest.mark.parametrize("actor", list(ROLE_MATRIX))
@pytest.mark.parametrize(("method", "url"), [("GET", URL), ("POST", RUN)])
def test_the_console_answers_administrators_only(client, actor, method, url) -> None:
    r = call(client, method, actor, url)
    assert r.status_code == ROLE_MATRIX[actor], r.text
    assert r.headers["content-type"].startswith("application/problem+json")


# ------------------------------------------------------------------------------------------------------ parity


def test_the_console_equals_live_calibration_on_the_same_window(client, folders) -> None:
    """Acceptance: the result equals data/burden_engine/calibration.py on the same window."""
    nightly(folders[0], network={"jacobian": "computed (17 network solves, 19 s)", "cache_file": "jacobian_a.json"})
    ref = reference()
    r = call(client, "GET", "adm", URL)
    assert r.status_code == 200, r.text
    body = r.json()
    cal = body["calibration"]
    assert body["day"] == DAY.isoformat() and body["current"] and cal["source"] == "live"
    for field, key in zip(FIELDS, VALUES, strict=True):
        assert abs(cal[field] - ref.cal[key]) <= 1e-6 * abs(ref.cal[key]), field
    assert [cal["window_start"], cal["window_end"]] == list(ref.window)
    assert cal["n_shifts"] == ref.n_shifts and cal["shifts"] == ref.shifts
    assert [m["shift_key"] for m in cal["mask"] if m["state"] == "used"] == ref.shifts
    assert body["jacobian_cache"]["status"] == "missing" and body["jacobian_cache"]["day_file"] == "jacobian_a.json"
    assert body["days"][0]["jacobian_cached"] is False and body["runs"] == [] and body["runs_source"] == "twin.jobs"


def test_a_day_without_a_calibration_answers_404_and_an_empty_store_an_empty_console(client) -> None:
    body = call(client, "GET", "adm", URL).json()
    assert body["day"] is None and body["calibration"] is None and body["days"] == []
    assert body["jacobian_cache"]["status"] == "none" and body["job"]["schedule"] == "02:30"
    r = call(client, "GET", "adm", URL, params={"day": "2026-10-01"})
    assert r.status_code == 404 and r.json()["title"] == "Calibration not stored"
    assert call(client, "GET", "adm", URL, params={"day": "15 Oct"}).status_code == 422


# -------------------------------------------------------------------------------------------- recalibration


def _audit(twin: TwinStore) -> list[Any]:
    rows, _ = twin.audit.page(limit=50, offset=0)
    return rows


def test_a_recalibration_is_queued_and_audited(client, twin) -> None:
    r = call(client, "POST", "adm", RUN)
    assert r.status_code == 202, r.text
    body = r.json()
    job = body["job"]
    assert job["kind"] == KIND and job["status"] == "queued" and job["submitted_by"] == "adm"
    assert not body["duplicate"]
    (entry,) = _audit(twin)
    assert entry.id == body["audit_id"] and (entry.actor, entry.role) == ("adm", "admin")
    assert (entry.action, entry.object_type, entry.object_id) == ("job.start", "job", job["id"])
    assert entry.details == {
        "kind": KIND,
        "title": "Nightly calibration",
        "duplicate": False,
        "origin": "calibration console",
    }
    assert entry.before is None and entry.after["id"] == job["id"] and entry.after["kind"] == KIND
    assert entry.client_ip
    # the same request while the job is queued returns it, and is recorded as well
    again = call(client, "POST", "adm", RUN).json()
    assert again["duplicate"] and again["job"]["id"] == job["id"]
    latest = _audit(twin)[0]
    assert latest.details["duplicate"] and latest.before == latest.after
    # the console lists the run (Redis without a twin store)
    store_service.use(None)
    body = call(client, "GET", "adm", URL).json()
    assert body["runs_source"] == "redis" and [r["id"] for r in body["runs"]] == [job["id"]]
    assert body["runs"][0]["scheduled"] is False


def test_the_generic_job_route_audits_the_calibration_too(client, twin) -> None:
    r = call(client, "POST", "adm", "/api/jobs", json={"kind": KIND, "params": {}})
    assert r.status_code == 202, r.text
    (entry,) = _audit(twin)
    assert entry.object_id == r.json()["job"]["id"] and entry.details["origin"] == "jobs"
    kinds = {k["name"]: k for k in call(client, "GET", "adm", "/api/jobs/kinds").json()}
    assert kinds[KIND]["audited"] and not kinds["system.selftest"]["audited"]
    # a kind without the flag writes no entry
    assert call(client, "POST", "adm", "/api/jobs", json={"kind": "system.selftest", "params": {}}).status_code == 202
    assert len(_audit(twin)) == 1


def test_without_the_twin_store_a_recalibration_is_refused(client) -> None:
    store_service.use(None)
    r = call(client, "POST", "adm", RUN)
    assert r.status_code == 503 and "BF_TWIN_DB_URL" in r.json()["detail"]
    r = call(client, "POST", "adm", "/api/jobs", json={"kind": KIND, "params": {}})
    assert r.status_code == 503 and "audit log" in r.json()["detail"]
    jobs = client.app.state.live.store  # type: ignore[attr-defined]
    assert not jobs.redis.sorted.get("twin:jobs")  # nothing queued


def test_a_job_whose_entry_cannot_be_written_is_cancelled_before_it_starts(client, twin, monkeypatch) -> None:
    def down(*args: Any, **kwargs: Any) -> None:
        raise OperationalError("INSERT INTO twin.audit_log", {}, ConnectionError("server closed the connection"))

    monkeypatch.setattr(twin.audit, "record", down)
    r = call(client, "POST", "adm", RUN)
    assert r.status_code == 503, r.text
    jobs = client.app.state.live.store  # type: ignore[attr-defined]
    (job_id,) = jobs.redis.sorted["twin:jobs"]
    import asyncio

    record = asyncio.run(jobs.get(job_id))
    assert record.status == "cancelled" and record.cancel_requested and record.error == job_audit.NOT_STARTED


def test_the_runs_come_from_the_job_archive(client, twin) -> None:
    t0 = datetime(2026, 10, 14, 21, 0, tzinfo=UTC)
    twin.jobs.upsert(
        {
            "id": "j-night",
            "kind": KIND,
            "status": "done",
            "params": {},
            "input_hash": "h",
            "submitted_by": "scheduler",
            "submitted_at": t0,
            "started_at": t0,
            "finished_at": t0 + timedelta(seconds=42),
            "tries": 1,
            "result": {"day": "2026-10-15"},
            "scheduled_for": t0,
            "run_s": 42.0,
        }
    )
    (run,) = call(client, "GET", "adm", URL).json()["runs"]
    assert (run["id"], run["scheduled"], run["run_s"], run["day"]) == ("j-night", True, 42.0, "2026-10-15")
    assert datetime.fromisoformat(run["finished_at"]) == t0 + timedelta(seconds=42)


# ------------------------------------------------------------------------------------------------------- units

UNITS = ("count", "dimensionless", "GJ/h", "kg/tHM", "%", "MJ/tHM", "t/h", "s", "bytes", "days", "unit")
SCHEMAS = [
    "CalibrationConsole",
    "DayCalibration",
    "MaskShift",
    "StabilityTest",
    "NetworkStats",
    "CalibrationDay",
    "JacobianFile",
    "JacobianCache",
    "NightlyJob",
    "CalibrationRun",
    "CalibrationRunStarted",
]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    numeric = [
        (name, prop)
        for name, prop in spec.get("properties", {}).items()
        if ({prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}) & {"number", "integer"}
        and not name.endswith("id")
    ]
    missing = [
        f"{name}: {prop.get('description')!r}"
        for name, prop in numeric
        if (prop.get("description") or "").rsplit(",", 1)[-1].strip().split(" ")[0] not in UNITS
    ]
    assert not missing, missing


def test_the_day_query_is_an_ist_date() -> None:
    (param,) = build().openapi()["paths"][URL]["get"]["parameters"]
    assert param["name"] == "day" and "IST" in param["description"]
    assert date.fromisoformat(DAY.isoformat()) == DAY


# ------------------------------------------------------------------------------------------------------- UI audit

UI_AUDIT_CONSOLE = (
    Path(__file__).resolve().parents[1] / "frontend" / "scripts" / "fixtures" / "calibration-console.json"
)


def test_the_ui_audit_console_is_what_the_route_returns() -> None:
    """The UI audit's mock of GET /api/admin/calibration validates against its model and is consistent: the used
    shifts of the mask are the calibration's, the shown day is the newest stored, the Jacobian status names its file."""
    data = json.loads(UI_AUDIT_CONSOLE.read_text(encoding="utf-8"))
    body = CalibrationConsole.model_validate(data)
    cal = body.calibration
    assert cal is not None and cal.mask is not None and body.day == body.days[0].day
    assert [m.shift_key for m in cal.mask if m.state == "used"] == cal.shifts
    assert {m.state for m in cal.mask} == {"used", "unstable", "missing"}
    assert body.jacobian_cache.status == "cached" and body.jacobian_cache.day_file == cal.network.cache_file
    assert {d.source for d in body.days} == {"live", "fallback"} and {r.status for r in body.runs} == {"done", "failed"}
