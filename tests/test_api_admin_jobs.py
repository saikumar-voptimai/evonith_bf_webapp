"""Jobs and system health API (SV-4.8, #311): administrators only; cancel and retry are audited, also the cancel of
``POST /api/jobs/{id}/cancel``; every source of ``GET /api/admin/health`` carries its last timestamp and age; numeric
fields state their unit. SQLite twin store, in-memory Redis."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.routers import admin_jobs
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import host_health, source_health
from services import store as store_service
from services.jobs import JobStore
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_bmo import IndexedRedis
from tests.test_api_live import session_cookie

HISTORY = "/api/admin/jobs"
HEALTH = "/api/admin/health"
SELFTEST = "system.selftest"


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
def jobs() -> JobStore:
    return JobStore(IndexedRedis())


@pytest.fixture()
def client(twin, jobs, monkeypatch):
    fake_host = host_health.collect(
        tegrastats=lambda: None,
        figures=lambda: {
            "cpu_pct": 20.0,
            "ram_used_mb": 2000.0,
            "ram_total_mb": 8000.0,
            "swap_used_mb": 0.0,
            "swap_total_mb": 0.0,
            "temperatures": [{"sensor": "cpu", "c": 51.0}],
        },  # fmt: skip
        disk_rows=lambda: [{"path": "/", "used_gb": 40.0, "total_gb": 100.0, "pct": 40.0, "state": "ok"}],
    )
    monkeypatch.setattr(host_health, "snapshot", lambda: fake_host)
    source_health.HEALTH.reset()
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        app.state.live.store = jobs
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c
    source_health.HEALTH.reset()


def call(client: TestClient, method: str, user: str | None, url: str, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.request(method, url, headers=headers, **kw)


def audit(twin: TwinStore) -> list[Any]:
    rows, _ = twin.audit.page(limit=50, offset=0)
    return list(reversed(rows))


# ------------------------------------------------------------------------------------------------------- roles
ROLE_MATRIX = {None: 401, "op": 403, "sup": 403}


@pytest.mark.parametrize("actor", list(ROLE_MATRIX))
@pytest.mark.parametrize(
    ("method", "url"),
    [("GET", HISTORY), ("GET", HEALTH), ("POST", f"{HISTORY}/x/cancel"), ("POST", f"{HISTORY}/x/retry")],
)
def test_the_view_answers_administrators_only(client, actor, method, url) -> None:
    r = call(client, method, actor, url)
    assert r.status_code == ROLE_MATRIX[actor], r.text
    assert r.headers["content-type"].startswith("application/problem+json")


UNITS = (", s", "%", "°C", "MB", "GB", "ms", "count", "tasks", "fraction", "unit")


def test_numeric_fields_state_their_unit() -> None:
    models = [
        admin_jobs.JobHistoryRow,
        admin_jobs.SourceRow,
        admin_jobs.ServiceRow,
        admin_jobs.HostHealth,
        admin_jobs.DiskRow,
        admin_jobs.TemperatureRow,
        admin_jobs.HostLimit,
    ]
    for model in models:
        for name, field in model.model_fields.items():
            annotation = str(field.annotation)
            if ("float" in annotation or "int" in annotation) and "list" not in annotation:
                text = field.description or ""
                assert any(u in text for u in UNITS), (model.__name__, name)


# ------------------------------------------------------------------------------------------------ job control
def test_a_cancel_from_the_view_is_audited_and_ends_a_queued_job(client, jobs, twin) -> None:
    record, _ = asyncio.run(jobs.submit(SELFTEST, {"steps": 2}, "adm"))
    r = call(client, "POST", "adm", f"{HISTORY}/{record.id}/cancel")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ended"] and body["job"]["status"] == "cancelled" and body["job"]["cancel_requested"]
    (entry,) = audit(twin)
    assert entry.id == body["audit_id"] and (entry.action, entry.object_id, entry.actor) == (
        "job.cancel",
        record.id,
        "adm",
    )
    assert entry.details["origin"] == "admin jobs" and entry.client_ip
    again = call(client, "POST", "adm", f"{HISTORY}/{record.id}/cancel")
    assert again.status_code == 409 and "cancelled" in again.json()["detail"]
    assert call(client, "POST", "adm", f"{HISTORY}/{'0' * 32}/cancel").status_code == 404
    assert len(audit(twin)) == 1


def test_the_jobs_cancel_route_is_audited_too(client, jobs, twin) -> None:
    """Acceptance: job control audited, also from the pages' cancel button (supervisors)."""
    record, _ = asyncio.run(jobs.submit(SELFTEST, {}, "adm"))
    assert call(client, "POST", "op", f"/api/jobs/{record.id}/cancel").status_code == 403
    r = call(client, "POST", "sup", f"/api/jobs/{record.id}/cancel")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "cancelled" and r.json()["error"] == "Cancelled before start."
    (entry,) = audit(twin)
    assert (entry.action, entry.actor, entry.role, entry.details["origin"]) == ("job.cancel", "sup", "supervisor",
                                                                                "jobs")  # fmt: skip


def test_a_cancel_without_the_twin_store_is_refused(client, jobs, twin) -> None:
    record, _ = asyncio.run(jobs.submit(SELFTEST, {}, "adm"))
    store_service.use(None)
    r = call(client, "POST", "sup", f"/api/jobs/{record.id}/cancel")
    assert r.status_code == 503 and "audit log" in r.json()["detail"]
    assert asyncio.run(jobs.get(record.id)).status == "queued"


def test_a_retry_is_audited_and_queues_the_same_parameters(client, jobs, twin) -> None:
    record, _ = asyncio.run(jobs.submit(SELFTEST, {"steps": 3}, "op"))
    assert call(client, "POST", "adm", f"{HISTORY}/{record.id}/retry").status_code == 409  # still queued
    asyncio.run(jobs.finish(record.id, "failed", error="boom"))
    r = call(client, "POST", "adm", f"{HISTORY}/{record.id}/retry")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["retry_of"] == record.id and not body["duplicate"] and body["job"]["params"]["steps"] == 3
    (entry,) = audit(twin)
    assert (entry.action, entry.object_id) == ("job.retry", body["job"]["id"]) and entry.id == body["audit_id"]
    assert entry.details["retry_of"] == record.id
    # the same retry while the new job is queued returns it, recorded as well
    again = call(client, "POST", "adm", f"{HISTORY}/{record.id}/retry").json()
    assert again["duplicate"] and again["job"]["id"] == body["job"]["id"] and len(audit(twin)) == 2
    assert call(client, "POST", "adm", f"{HISTORY}/{'0' * 32}/retry").status_code == 404


def test_a_retry_of_parameters_the_kind_no_longer_accepts_is_422(client, twin) -> None:
    twin.jobs.upsert(dict(id="d" * 32, kind=SELFTEST, status="failed", params={"speed": 1}, input_hash="h",
                          submitted_by="adm", submitted_at=datetime.now(UTC) - timedelta(days=9)))  # fmt: skip
    r = call(client, "POST", "adm", f"{HISTORY}/{'d' * 32}/retry")
    assert r.status_code == 422 and r.json()["errors"][0]["loc"] == ["body", "params", "speed"]
    assert audit(twin) == []


# ---------------------------------------------------------------------------------------------------- history
def test_the_history_lists_jobs_with_durations_and_controls(client, jobs, twin) -> None:
    now = datetime.now(UTC)
    failed, _ = asyncio.run(jobs.submit(SELFTEST, {"steps": 1}, "op"))
    asyncio.run(jobs.update(failed.id, status="running", started_at=now - timedelta(seconds=40)))
    asyncio.run(jobs.finish(failed.id, "failed", error="boom", run_s=7.0))
    queued, _ = asyncio.run(jobs.submit(SELFTEST, {"steps": 2}, "op"))
    twin.jobs.upsert(dict(id="e" * 32, kind="twin.evaluate", status="done", params={}, input_hash="h",
                          submitted_by="scheduler", submitted_at=now - timedelta(days=10), run_s=95.0,
                          scheduled_for=now - timedelta(days=10)))  # fmt: skip

    body = call(client, "GET", "adm", HISTORY).json()
    assert body["sources"] == ["twin.jobs", "redis"] and body["limit"] == 200
    assert [r["id"] for r in body["rows"]] == [queued.id, failed.id]  # 7 days by default
    q, f = body["rows"]
    assert q["can_cancel"] and not q["can_retry"] and q["wait_s"] >= 0 and q["live"]
    assert f["can_retry"] and not f["can_cancel"] and f["run_s"] == 7.0 and f["error"] == "boom"

    month = call(client, "GET", "adm", HISTORY, params={"days": 30}).json()["rows"]
    assert month[-1]["id"] == "e" * 32 and month[-1]["scheduled"] and month[-1]["run_s"] == 95.0
    failed_only = call(client, "GET", "adm", HISTORY, params={"days": 30, "status": "failed"}).json()["rows"]
    assert [r["id"] for r in failed_only] == [failed.id]
    kind_only = call(client, "GET", "adm", HISTORY, params={"days": 30, "kind": "twin.evaluate"}).json()["rows"]
    assert [r["id"] for r in kind_only] == ["e" * 32]
    assert call(client, "GET", "adm", HISTORY, params={"days": 93}).status_code == 422
    assert call(client, "GET", "adm", HISTORY, params={"status": "lost"}).status_code == 422


# ----------------------------------------------------------------------------------------------------- health
def test_every_source_shows_its_last_timestamp_and_age(client, twin) -> None:
    """Acceptance: every source shows last timestamp and age."""
    now = datetime.now(UTC)
    for i, key in enumerate(source_health.SOURCES):
        source_health.HEALTH.record_ok(key, data_end=now - timedelta(hours=i + 0.25))
    end = now.replace(minute=0, second=0, microsecond=0)
    twin.snapshots.upsert(dict(window_start=end - timedelta(hours=8), window_end=end, evaluated_at=end,
                               status="complete", quantities={}, flags={}, versions={}))  # fmt: skip

    r = call(client, "GET", "adm", HEALTH)
    assert r.status_code == 200, r.text
    body = r.json()
    checked = datetime.fromisoformat(body["checked_at"])
    keys = [s["key"] for s in body["sources"]]
    assert keys == [*source_health.SOURCES, "evaluation"]
    for s in body["sources"]:
        last = datetime.fromisoformat(s["last_at"])
        assert s["age_s"] == pytest.approx((checked - last).total_seconds(), abs=1e-3), s["key"]
    by = {s["key"]: s for s in body["sources"]}
    assert by["influxdb"]["state"] == "ok" and by["hourly"]["state"] == "ok" and by["evaluation"]["group"] == "twin"

    services = {s["key"]: s for s in body["services"]}
    assert list(services) == ["api", "redis", "twin_store", "worker"]
    assert services["api"]["state"] == "ok" and services["api"]["last_at"] and services["api"]["age_s"] >= 0
    assert services["redis"]["state"] == "off"  # no BF_REDIS_URL in the test app
    assert services["twin_store"]["state"] == "down"  # SQLite created without alembic
    assert services["worker"]["state"] == "missing"
    host = body["host"]
    assert host["source"] == "psutil" and host["ram_pct"] == 25.0 and host["temperatures"][0]["c"] == 51.0
    assert body["counts"]["ok"] >= 3 and body["host_error"] is None


def test_a_failed_source_is_down_with_its_error_and_the_host_error_is_reported(client, monkeypatch) -> None:
    source_health.HEALTH.record_failure("dpr", "ConnectionError: Neon unreachable")

    def broken() -> dict:
        raise OSError("psutil")

    monkeypatch.setattr(host_health, "snapshot", broken)
    store_service.use(None)
    body = call(client, "GET", "adm", HEALTH).json()
    by = {s["key"]: s for s in body["sources"]}
    assert by["dpr"]["state"] == "down" and "Neon unreachable" in by["dpr"]["error"]
    assert by["evaluation"]["state"] == "off" and by["influxdb"]["state"] == "missing"
    assert body["host"] is None and body["host_error"] == "OSError: psutil"
    assert {s["key"]: s["state"] for s in body["services"]}["twin_store"] == "off"
