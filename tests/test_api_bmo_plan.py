"""Blend Mix Optimiser plan board (SV-7.4, #327): the Step 2 runs the /bmo page shows, the implied production of the
latest Step 1 run, the decision that applies, and the page's Fixed-ω run from its constraints queued as
``optimiser.run``. Fakes only: no network solve, no worker."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.live import job_store
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore
from services import bmo_plan
from services import store as store_service
from services.jobs import OptimiserRunParams
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie
from tests.test_api_optimiser import FakeJobs

T0 = datetime(2026, 10, 30, 0, 45, tzinfo=UTC)  # 06:15 IST, the shift's Step 2 slot


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
def api(twin):
    jobs = FakeJobs()
    app = build()
    app.dependency_overrides[job_store] = lambda: jobs
    with TestClient(app, base_url="https://testserver") as client:
        cookies = {u: session_cookie(app, u) for u in PASSWORDS}
        yield client, cookies, jobs


def call(api, method: str, user: str | None, url: str, **kw: Any):
    client, cookies, _ = api
    headers = {"Cookie": cookies[user]} if user else {}
    return client.request(method, url, headers=headers, **kw)


def add_run(
    store: TwinStore,
    *,
    at: datetime,
    mode: str = "step2",
    omega: str = "strata",
    trigger: str = "schedule",
    status: str = "ok",
    production: float | None = 2270.0,
    certainty: str | None = "High",
    **extra: Any,
):
    values = dict(
        id=uuid.uuid4(),
        job_id=None,
        trigger=trigger,
        requested_by="scheduler" if trigger == "schedule" else "op",
        mode=mode,
        omega_mode=omega,
        started_at=at,
        finished_at=at + timedelta(minutes=2),
        status=status,
        production_t_d=production,
        cost_rs_thm=30139.0 if status == "ok" else None,
        saving_rs_thm=155.0 if status == "ok" else None,
        coke_kg_thm=305.7 if status == "ok" else None,
        coke_sd_kg_thm=4.1 if omega == "strata" and status == "ok" else None,
        fuel_rate_kg_thm=567.0 if status == "ok" else None,
        certainty=certainty if omega == "strata" and status == "ok" else None,
        inputs={},
        result={},
        versions={},
    )
    values.update(extra)
    return store.optimiser.add(values)


def test_the_board_names_the_shift_run_the_latest_run_and_the_implied_production(twin) -> None:
    step1 = add_run(twin, at=T0 - timedelta(hours=8, minutes=-15), mode="step1", production=2310.0)
    shift = add_run(twin, at=T0)  # the shift's scheduled STRATA-ω Step 2 run
    add_run(twin, at=T0, omega="fixed")  # its Fixed-ω twin
    add_run(twin, at=T0 + timedelta(hours=1), omega="fixed", trigger="user", status="infeasible", production=None)
    page = add_run(twin, at=T0 + timedelta(hours=2), omega="fixed", trigger="user", production=2250.0)
    twin.optimiser.decide(page.id, "accepted", Actor(name="sup", role="supervisor", client_ip=None),
                          shift_key="2026-10-30_A", comment="Apply from 10:00")  # fmt: skip

    b = bmo_plan.board(twin, now=T0 + timedelta(hours=3))
    assert b.implied is not None and (b.implied.production_t_d, b.implied.run_id) == (2310.0, str(step1.id))
    assert b.implied.omega_mode == "strata" and b.implied.trigger == "schedule"
    assert b.shift_run is not None and b.shift_run.id == str(shift.id) and b.shift_run.certainty == "High"
    assert b.latest_run is not None and b.latest_run.id == str(page.id) and b.latest_run.certainty is None
    assert b.latest_run.decision is not None and b.latest_run.decision.decided_by == "sup"
    assert b.latest_run.decision.comment == "Apply from 10:00" and b.shift_run.decision is None
    assert [r.status for r in b.recent] == ["ok", "infeasible", "ok", "ok"]  # newest first, Step 1 left out
    assert [q.quantity for q in b.slag_bands] == ["b2", "t_basicity", "ib4"]


def test_an_old_scheduled_run_is_not_the_shift_run_but_stays_the_latest(twin) -> None:
    old = add_run(twin, at=T0 - timedelta(hours=30))
    b = bmo_plan.board(twin, now=T0)
    assert b.shift_run is None and b.latest_run is not None and b.latest_run.id == str(old.id) and b.recent == []
    assert b.implied is None


def test_an_empty_store_gives_an_empty_board(twin) -> None:
    b = bmo_plan.board(twin, now=T0)
    assert (b.implied, b.shift_run, b.latest_run, b.recent) == (None, None, None, [])


def test_a_step1_run_without_production_implies_nothing(twin) -> None:
    add_run(twin, at=T0, mode="step1", production=None)
    assert bmo_plan.implied_production(twin) is None


def test_the_board_is_served_with_the_decision_that_applies(api, twin) -> None:
    run = add_run(twin, at=T0)
    twin.optimiser.decide(run.id, "rejected", Actor(name="sup", role="supervisor", client_ip=None))
    twin.optimiser.decide(run.id, "accepted", Actor(name="sup", role="supervisor", client_ip=None))
    r = call(api, "GET", "op", "/api/bmo/plan")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shift_run"]["id"] == str(run.id) == body["latest_run"]["id"]
    assert body["shift_run"]["decision"]["decision"] == "accepted" and body["implied"] is None
    assert body["shift_run"]["production_t_d"] == 2270.0 and body["shift_run"]["coke_kg_thm"] == 305.7


def test_a_page_run_is_queued_as_a_fixed_omega_step2_run_with_the_page_limits(api) -> None:
    body = {"production_t_d": 2250, "limits": {"slag_max_kg_thm": 330, "cao_sio2_min": 1.06, "charges_per_h": 6.4}}
    r = call(api, "POST", "op", "/api/bmo/plan/runs", json=body)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert job["kind"] == "optimiser.run" and job["submitted_by"] == "op" and r.json()["duplicate"] is False
    want = OptimiserRunParams(
        mode="step2", omega_mode="fixed", production_t_d=2250, calibration="day", yard="live",
        limits={"slag_max_kg_thm": 330, "cao_sio2_min": 1.06, "charges_per_h": 6.4},  # type: ignore[arg-type]
    )  # fmt: skip
    assert job["params"] == want.model_dump(mode="json")
    assert job["params"]["certainty"] == "low" and job["params"]["settings"] is None

    default = call(api, "POST", "op", "/api/bmo/plan/runs")
    assert default.status_code == 202 and default.json()["job"]["params"]["production_t_d"] is None

    recompute = call(api, "POST", "op", "/api/bmo/plan/runs", json={"mode": "step1", "production_t_d": 2250})
    params = recompute.json()["job"]["params"]
    assert params["mode"] == "step1" and params["omega_mode"] == "fixed" and params["production_t_d"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"production_t_d": 1200},
        {"mode": "max_production"},
        {"omega_mode": "strata"},
        {"limits": {"cao_sio2_min": 3}},
        {"limits": {"unknown": 1}},
    ],
)
def test_a_page_run_request_out_of_range_is_refused(api, body) -> None:
    r = call(api, "POST", "op", "/api/bmo/plan/runs", json=body)
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")


def test_the_plan_routes_need_a_session_and_are_open_to_users(api) -> None:
    assert call(api, "GET", None, "/api/bmo/plan").status_code == 401
    assert call(api, "POST", None, "/api/bmo/plan/runs").status_code == 401
    assert call(api, "GET", "op", "/api/bmo/plan").status_code == 200
