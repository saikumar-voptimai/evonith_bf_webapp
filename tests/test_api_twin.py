"""Twin and optimiser API (SV-1.6, #286): routes on a SQLite twin store, units in the schemas, decisions audited,
30 days of hourly series within 500 ms, and an optimiser run from the route through the worker into the store."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import store as store_service
from services import twin_views
from tests.pg_server import needs_postgres
from tests.redis_server import needs_redis, redis_server
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import Users, session_cookie

IST = "Asia/Kolkata"


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
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def get(client: TestClient, user: str, url: str, **kw: Any):
    return client.get(url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


def post(client: TestClient, user: str, url: str, **kw: Any):
    return client.post(url, headers={"Cookie": client.cookies_by_user[user]}, **kw)  # type: ignore[attr-defined]


def ist(text: str) -> datetime:
    return pd.Timestamp(text, tz=IST).tz_convert("UTC").to_pydatetime()


def snapshot(end: datetime, **values: Any) -> dict[str, Any]:
    return {
        "window_start": end - timedelta(hours=8),
        "window_end": end,
        "evaluated_at": end + timedelta(minutes=5),
        "status": "complete",
        "k_trace": 305.0,
        "k_forge": 303.0,
        "k_strata": 304.0,
        "k_strata_sd": 2.0,
        "k_charged": 306.0,
        "omega_imp": 0.80,
        "omega_strata": 0.79,
        "omega_strata_sd": 0.01,
        "omega_cal": 0.80,
        "quantities": {"hmt_c": 1478.0, "tau_raceway_h": 6.1, **{f"q{i}_x": float(i) for i in range(30)}},
        "flags": {"notes": []},
        "versions": {"model": "test"},
        **values,
    }


class Frozen(datetime):
    at = datetime(2026, 10, 27, 6, 0, tzinfo=UTC)  # 11:30 IST

    @classmethod
    def now(cls, tz=None):
        return cls.at if tz is None else cls.at.astimezone(tz)


# ----------------------------------------------------------------------------------------------------------- now
def test_now_reports_the_latest_evaluation_and_its_lag(twin, client, monkeypatch) -> None:
    monkeypatch.setattr(twin_views, "datetime", Frozen)
    assert twin_views.expected_window_end(Frozen.at) == ist("2026-10-27 11:00")
    empty = get(client, "op", "/api/twin/now").json()
    assert empty["snapshot"] is None and empty["stale"] and empty["alarms_open"] == 0

    twin.snapshots.upsert(snapshot(ist("2026-10-27 10:00")))
    twin.snapshots.upsert(snapshot(ist("2026-10-27 11:00"), k_trace=307.5))
    twin.alarms.raise_alarm("omega_drift", subject="high", severity="warning", message="drift")
    body = get(client, "op", "/api/twin/now").json()
    assert body["snapshot"]["k_trace"] == 307.5 and body["snapshot"]["quantities"]["hmt_c"] == 1478.0
    assert pd.Timestamp(body["snapshot"]["window_end"]) == pd.Timestamp(ist("2026-10-27 11:00"))
    assert body["lag_h"] == 0 and not body["stale"] and body["alarms_open"] == 1
    assert body["age_s"] == pytest.approx(25 * 60)  # evaluated at 11:05 IST, now 11:30

    Frozen.at = datetime(2026, 10, 27, 8, 0, tzinfo=UTC)  # 13:30 IST: the 12:00 and 13:00 windows are missing
    try:
        late = get(client, "op", "/api/twin/now").json()
    finally:
        Frozen.at = datetime(2026, 10, 27, 6, 0, tzinfo=UTC)
    assert late["lag_h"] == 2 and late["stale"]


def test_without_a_store_the_twin_routes_answer_503(client) -> None:
    store_service.use(None)
    r = get(client, "op", "/api/twin/now")
    assert r.status_code == 503 and "BF_TWIN_DB_URL" in r.json()["detail"]
    assert get(client, "op", "/api/optimiser/latest").status_code == 503


def test_the_twin_needs_a_session(client) -> None:
    assert client.get("/api/twin/now").status_code == 401


# -------------------------------------------------------------------------------------------------------- series
def test_series_hourly_and_per_shift(twin, client) -> None:
    ends = pd.date_range("2026-10-24 01:00", "2026-10-27 00:00", freq="h", tz=IST)  # 72 hourly windows
    for end in ends:
        twin.snapshots.upsert(snapshot(end.tz_convert("UTC").to_pydatetime(), k_trace=300.0 + end.hour))
    params = {"start": ends[0].isoformat(), "end": ends[-1].isoformat()}
    hourly = get(client, "op", "/api/twin/series", params={**params, "quantities": "hmt_c, tau_raceway_h"}).json()
    assert hourly["count"] == 72 and hourly["resolution"] == "hour"
    first = hourly["rows"][0]
    assert first["quantities"] == {"hmt_c": 1478.0, "tau_raceway_h": 6.1} and first["shift_key"] is None
    shifts = get(client, "op", "/api/twin/series", params={**params, "resolution": "shift"}).json()
    keys = [r["shift_key"] for r in shifts["rows"]]
    assert keys == [
        "2026-10-23_C",
        "2026-10-24_A",
        "2026-10-24_B",
        "2026-10-24_C",
        "2026-10-25_A",
        "2026-10-25_B",
        "2026-10-25_C",
        "2026-10-26_A",
        "2026-10-26_B",
    ]  # fmt: skip  (the window ending 06:00 is the C shift of the day before)
    assert [r["k_trace"] for r in shifts["rows"][:3]] == [306.0, 314.0, 322.0]  # windows ending 06:00, 14:00, 22:00
    assert shifts["rows"][0]["quantities"] == {}
    bad = get(client, "op", "/api/twin/series", params={"start": ends[-1].isoformat(), "end": ends[0].isoformat()})
    assert bad.status_code == 422
    far = {"start": "2026-01-01T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
    assert get(client, "op", "/api/twin/series", params=far).status_code == 422


def _thirty_days(store: TwinStore, client: TestClient) -> float:
    from data.twin.models import Snapshot

    end = ist("2026-10-27 06:00")
    with store.sessions.begin() as session:  # 720 rows in one transaction
        for k in range(720):
            session.add(Snapshot(**snapshot(end - timedelta(hours=k))))
    params = {"start": (end - timedelta(days=30)).isoformat(), "end": end.isoformat(), "quantities": "hmt_c"}
    get(client, "op", "/api/twin/series", params=params)  # first request: connection and imports
    t0 = time.perf_counter()
    r = get(client, "op", "/api/twin/series", params=params)
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200 and r.json()["count"] == 720
    return elapsed


def test_thirty_days_of_hourly_series_within_500_ms(twin, client) -> None:
    elapsed = _thirty_days(twin, client)
    assert elapsed < 0.5, f"{elapsed:.3f} s"


@pytest.fixture(scope="module")
def pg_url():
    from tests.pg_server import fresh_database, postgres_server
    from tests.test_twin_store import alembic_config

    with postgres_server() as server, fresh_database(server) as url:
        from alembic import command

        command.upgrade(alembic_config(url), "head")
        yield url


@needs_postgres
def test_thirty_days_of_hourly_series_within_500_ms_on_postgresql(pg_url) -> None:
    store = TwinStore.from_url(pg_url)
    store_service.use(store)
    try:
        app = build()
        with TestClient(app, base_url="https://testserver") as client:
            client.cookies_by_user = {"op": session_cookie(app, "op")}  # type: ignore[attr-defined]
            elapsed = _thirty_days(store, client)
    finally:
        store_service.reset()
        store.dispose()
    assert elapsed < 0.5, f"{elapsed:.3f} s"


# -------------------------------------------------------------------------------------------------------- alarms
def test_alarms_open_and_history(twin, client) -> None:
    twin.alarms.raise_alarm("omega_drift", subject="high", severity="warning", message="ω above STRATA")
    twin.alarms.raise_alarm("source_stale", subject="influx", severity="critical", message="no data")
    twin.alarms.clear_alarm("source_stale", subject="influx")
    body = get(client, "op", "/api/twin/alarms").json()
    assert [a["kind"] for a in body["open"]] == ["omega_drift"]
    assert sorted(a["kind"] for a in body["history"]) == ["omega_drift", "source_stale"]
    cleared = next(a for a in body["history"] if a["kind"] == "source_stale")
    assert cleared["cleared_at"] is not None and cleared["severity"] == "critical"
    only = get(client, "op", "/api/twin/alarms", params={"kind": "source_stale"}).json()
    assert only["open"] == [] and len(only["history"]) == 1


# ------------------------------------------------------------------------------------------------------- transit
def test_transit_from_the_hourly_dataset(twin, client, monkeypatch) -> None:
    from services import coke_energy as ce
    from tests.twin_data import synthetic_frames

    fd, *_ = synthetic_frames(start="2026-05-01", days=4)
    monkeypatch.setattr(ce, "data_keys", lambda cfg, nonce: ce.DataKeys("x", 1, "c", 0, "2026-05-01", "{}", "d", 30.0))
    monkeypatch.setattr(ce, "CACHED_LOADERS", ce.Loaders(lambda p, m: fd, None, None, None, None))
    body = get(client, "op", "/api/twin/transit", params={"at": "2026-05-03T06:30:00Z"}).json()
    assert [b["zone"] for b in body["bands"]] == ["above_cz", "in_cz", "below_cz"]
    assert body["complete"] and body["missing_hours"] == 0
    assert 1.0 < body["tau_cz_h"] < body["tau_raceway_h"] < 12.0
    assert body["coke_in_transit_t"] == pytest.approx(sum(b["coke_inventory_t"] for b in body["bands"]))
    assert body["lag"] is None


# ----------------------------------------------------------------------------------------------------- optimiser
def run_row(**values: Any) -> dict[str, Any]:
    started = values.pop("started_at", datetime(2026, 10, 27, 0, 45, tzinfo=UTC))
    return {
        "trigger": "schedule",
        "requested_by": "scheduler",
        "mode": "step2",
        "omega_mode": "strata",
        "started_at": started,
        "finished_at": started + timedelta(seconds=40),
        "status": "ok",
        "production_t_d": 2270.0,
        "cost_rs_thm": 29852.4,
        "saving_rs_thm": 180.0,
        "coke_kg_thm": 303.0,
        "coke_sd_kg_thm": 1.2,
        "fuel_rate_kg_thm": 564.7,
        "certainty": "High",
        "inputs": {"mode": "step2"},
        "result": {"mode": "fixed"},
        **values,
    }


def test_decisions_are_audited_and_the_latest_applies(twin, client) -> None:
    run = twin.optimiser.add(run_row())
    url = f"/api/optimiser/runs/{run.id}/decision"
    assert post(client, "op", url, json={"decision": "accepted"}).status_code == 403  # supervisor or admin
    first = post(client, "sup", url, json={"decision": "accepted", "comment": " Pellet within trial limit "})
    assert first.status_code == 201, first.text
    assert first.json()["decided_by"] == "sup" and first.json()["comment"] == "Pellet within trial limit"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_[ABC]", first.json()["shift_key"])
    second = post(client, "adm", url, json={"decision": "rejected", "comment": "Si trending up"})
    assert second.status_code == 201
    latest = get(client, "op", "/api/optimiser/latest").json()
    assert latest["decision"]["decision"] == "rejected" and [d["decision"] for d in latest["decisions"]] == [
        "accepted",
        "rejected",
    ]
    audit = twin.audit.query(action="plan.decide")
    assert [(a.actor, a.details["decision"]) for a in reversed(audit)] == [("sup", "accepted"), ("adm", "rejected")]
    assert all(a.object_id == str(run.id) and a.client_ip for a in audit)

    failed = twin.optimiser.add(run_row(status="infeasible", started_at=datetime(2026, 10, 27, 1, 0, tzinfo=UTC)))
    assert (
        post(client, "sup", f"/api/optimiser/runs/{failed.id}/decision", json={"decision": "accepted"}).status_code
        == 409
    )
    missing = post(client, "sup", f"/api/optimiser/runs/{uuid.uuid4()}/decision", json={"decision": "accepted"})
    assert missing.status_code == 404
    assert post(client, "sup", url, json={"decision": "maybe"}).status_code == 422


def test_latest_and_recent_runs(twin, client) -> None:
    assert get(client, "op", "/api/optimiser/latest").status_code == 404
    twin.optimiser.add(run_row())
    twin.optimiser.add(run_row(mode="step1", status="infeasible", started_at=datetime(2026, 10, 27, 1, 0, tzinfo=UTC)))
    latest = get(client, "op", "/api/optimiser/latest").json()
    assert latest["mode"] == "step2" and latest["cost_rs_thm"] == 29852.4 and latest["inputs"] == {"mode": "step2"}
    assert get(client, "op", "/api/optimiser/latest", params={"mode": "step1"}).status_code == 404  # none ok
    runs = get(client, "op", "/api/optimiser/runs").json()
    assert [r["mode"] for r in runs] == ["step1", "step2"] and runs[0]["status"] == "infeasible"


# ----------------------------------------------------------------------------------------------------------- units
UNITS = ("kg/tHM", "t/d", "Rs/tHM", "°C", "%", "mass", "count", "dimensionless", "kg/t", "charges/h", "h", "min", "s",
         "t")  # fmt: skip


def _numeric(prop: dict[str, Any]) -> bool:
    types = {prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}
    return bool(types & {"number", "integer"})


@pytest.mark.parametrize(
    "schema",
    [
        "TwinNow",
        "TwinSeriesRow",
        "TwinTransit",
        "TransitBand",
        "TransitLag",
        "TwinSnapshotData",
        "OptimiserRunSummary",
        "OptimiserRunDetail",
        "OptimiserRunParams",
        "SiForecastPoint",
        "SiForecastData",
        "ChargedCokePoint",
        "ChargedCokeForecastData",
    ],
)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = []
    for name, prop in spec.get("properties", {}).items():
        if not _numeric(prop) or name == "id" or name.endswith("_id"):
            continue
        unit = (prop.get("description") or "").rsplit(",", 1)[-1].strip()
        if not (prop.get("description") and "," in prop["description"] and unit.split(" ")[0] in UNITS):
            missing.append(f"{name}: {prop.get('description')!r}")
    assert not missing, missing


# ------------------------------------------------------------------------------------------- run through the worker
@needs_redis
def test_a_run_started_through_the_api_is_stored_by_the_worker(twin, tmp_path, monkeypatch) -> None:
    """Step 2 with Fixed-ω on the reference yard and the shipped calibration (no plant data in the test)."""
    from arq.connections import RedisSettings
    from arq.worker import create_worker

    from worker.settings import WorkerSettings

    monkeypatch.setenv("BF_WORKER_POOL", "0")  # solve in the worker process
    with redis_server(tmp_path) as url:
        app = build(redis_url=url)
        with TestClient(app, base_url="https://testserver") as client:
            users = Users(app, client)
            body = {"mode": "step2", "omega_mode": "fixed"}
            assert users.post("op", "/api/optimiser/runs", json=body).status_code == 403
            assert users.post("sup", "/api/optimiser/runs", json={"mode": "step9"}).status_code == 422
            started = users.post("sup", "/api/optimiser/runs", json=body)
            assert started.status_code == 202, started.text
            job_id = started.json()["job"]["id"]
            again = users.post("adm", "/api/optimiser/runs", json=body).json()
            assert again["duplicate"] and again["job"]["id"] == job_id
            queued = users.get("op", f"/api/optimiser/runs/{job_id}").json()
            assert queued["job"]["status"] == "queued" and queued["run"] is None

            async def work() -> None:
                worker = create_worker(
                    WorkerSettings, redis_settings=RedisSettings.from_dsn(url), poll_delay=0.05, handle_signals=False,
                    burst=True, cron_jobs=[],
                )  # fmt: skip
                await worker.main()
                await worker.close()

            asyncio.run(work())
            done = users.get("op", f"/api/optimiser/runs/{job_id}").json()
            assert done["job"]["status"] == "done", done["job"]
            run = done["run"]
            assert run["status"] == "ok" and run["mode"] == "step2" and run["omega_mode"] == "fixed"
            assert run["trigger"] == "user" and run["requested_by"] == "sup" and run["job_id"] == job_id
            assert run["production_t_d"] == pytest.approx(2270.0) and 250 < run["coke_kg_thm"] < 350
            assert run["inputs"]["yard_source"] == "reference" and run["inputs"]["calibration"]["source"] == "fallback"
            assert done["job"]["result"]["run_id"] == run["id"]
            by_run = users.get("op", f"/api/optimiser/runs/{run['id']}").json()
            assert by_run["run"]["id"] == run["id"] and by_run["job"]["id"] == job_id
            decided = users.post("sup", f"/api/optimiser/runs/{run['id']}/decision", json={"decision": "accepted"})
            assert decided.status_code == 201
            assert users.get("op", "/api/optimiser/latest").json()["decision"]["decision"] == "accepted"
    assert twin.jobs.get(job_id) is not None  # the job record the run refers to
