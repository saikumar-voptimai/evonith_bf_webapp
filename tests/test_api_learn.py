"""Validation and STRATA-Learn console API (SV-4.5, #308): administrators only; the console's numbers equal the
shadow run's (through JSON); decisions write through the fact-register editor with their audit rows and answer 404 /
409 / 422 problems; a manual shadow run is queued and audited, also through ``POST /api/jobs``; numeric fields state
their unit; the UI audit's fixture is what the route returns. SQLite twin store, in-memory Redis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.routers.learn import LearnConsole
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import facts, learn
from services import store as store_service
from services.jobs import JobStore
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_bmo import IndexedRedis
from tests.test_api_live import session_cookie
from tests.test_services_learn_console import REGISTER, SPEC, close
from utils.burden_engine.learn.evidence import EvidenceStore
from utils.burden_engine.learn.run import shadow_run

URL = "/api/admin/learn"
DECIDE = "/api/admin/learn/decisions"
RUN = "/api/admin/learn/run"
KIND = "learn.shadow"


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("learn")
    learn.shadow(folder)
    return folder, shadow_run(EvidenceStore(), dict(SPEC), dict(REGISTER))


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    facts.approved_overlay.clear()
    yield store
    store_service.reset()
    facts.approved_overlay.clear()
    store.dispose()


@pytest.fixture()
def folder(outputs, monkeypatch):
    monkeypatch.setattr(learn, "STORE_DIR", outputs[0])
    return outputs[0]


@pytest.fixture()
def client(twin):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        app.state.live.store = JobStore(IndexedRedis())
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def call(client: TestClient, method: str, user: str | None, url: str, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.request(method, url, headers=headers, **kw)


def _audit(twin: TwinStore) -> list[Any]:
    rows, _ = twin.audit.page(limit=50, offset=0)
    return rows


# ------------------------------------------------------------------------------------------------------- roles

ROLE_MATRIX = {None: 401, "op": 403, "sup": 403}


@pytest.mark.parametrize("actor", list(ROLE_MATRIX))
@pytest.mark.parametrize(("method", "url"), [("GET", URL), ("POST", DECIDE), ("POST", RUN)])
def test_the_console_answers_administrators_only(client, actor, method, url) -> None:
    r = call(client, method, actor, url, json={"proposal": "x", "items": ["VAL-02"], "approve": False})
    assert r.status_code == ROLE_MATRIX[actor], r.text
    assert r.headers["content-type"].startswith("application/problem+json")


# ------------------------------------------------------------------------------------------------------ parity


def test_the_console_equals_the_shadow_run(client, folder, outputs) -> None:
    """Acceptance: the numbers equal learn.run output, through the API's JSON."""
    ref = outputs[1]
    r = call(client, "GET", "adm", URL)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run"]["n_days"] == 191 and body["gate"]["status"] == ref["gate"]["status"]
    assert close(body["cusum"]["s_plus"], ref["drift"]["s_plus"])
    for s in body["skill"]:
        assert close(s["coke_day"], ref["skill"][s["label"]]["coke_day"])
    for f, rf in zip(body["families"], ref["families"], strict=True):
        assert (
            close(f["strata_kg_thm"], rf["strata"]) and close(f["skill"], rf["skill"]) and f["verdict"] == rf["verdict"]
        )
    pellet = next(p for p in body["trajectory"]["params"] if p["name"] == "lam_pellet")
    assert all(close(a, b) for a, b in zip(pellet["mean"], ref["history"]["lam_pellet"], strict=True))
    assert body["proposal"]["n_auto"] == ref["proposal"]["n_auto"] and body["decisions_reason"] is None
    assert body["validation"]["strict"]["periods"][0] == {"period": "All days", "rmse_fixed": 8.23, "rmse_strata": 8.0}


def test_without_a_run_the_console_shows_the_asset_only(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(learn, "STORE_DIR", tmp_path)
    body = call(client, "GET", "adm", URL).json()
    assert body["run"] is None and body["proposal"] is None and body["skill"] == [] and body["runs"] == []
    assert body["validation"]["expanding"]["rmse_fixed"] == 6.39 and body["job"]["schedule"] == "03:00"
    r = call(client, "POST", "adm", DECIDE, json={"proposal": "x", "items": ["VAL-02"], "approve": True})
    assert r.status_code == 404 and r.json()["title"] == "No run"


# -------------------------------------------------------------------------------------------------- decisions


def test_an_approval_is_a_pending_facts_version_with_its_audit_row(client, twin, folder) -> None:
    proposal = call(client, "GET", "adm", URL).json()["proposal"]
    body = {"proposal": proposal["hash"], "items": ["VAL-02"], "approve": True, "note": " monthly "}
    r = call(client, "POST", "adm", DECIDE, json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["decision"] == "approved" and out["facts_version"]["status"] == "pending"
    assert out["facts_version"]["facts"] == ["VAL-02"] and out["facts_version"]["created_by"] == "adm"
    propose, decision = _audit(twin)  # newest first
    assert (decision.action, decision.object_id, decision.id) == (
        "learn.approve",
        proposal["object_id"],
        out["audit_id"],
    )
    assert decision.details["note"] == "monthly" and decision.client_ip and propose.action == "facts.propose"
    # the editor lists the version; the console shows the decision; the same decision again is a conflict
    assert call(client, "GET", "adm", "/api/admin/facts/versions").json()[0]["version"] == 1
    item = next(i for i in call(client, "GET", "adm", URL).json()["proposal"]["items"] if i["key"] == "VAL-02")
    assert item["state"] == "proposed" and item["decision"]["facts_version"] == 1
    r = call(client, "POST", "adm", DECIDE, json=body)
    assert r.status_code == 409 and "VAL-02 was approved by adm" in r.json()["detail"]


def test_decisions_answer_problems(client, twin, folder) -> None:
    h = call(client, "GET", "adm", URL).json()["proposal"]["hash"]
    cases = [
        ({"proposal": "0" * 16, "items": ["VAL-02"], "approve": True}, 409, "reload the console"),
        ({"proposal": h, "items": ["VAL-99"], "approve": False}, 422, "No item VAL-99"),
        ({"proposal": h, "items": ["GAS-23"], "approve": True}, 422, "Nothing an overlay carries"),
    ]
    for body, status, text in cases:
        r = call(client, "POST", "adm", DECIDE, json=body)
        assert r.status_code == status and text in r.json()["detail"], r.text
        assert r.headers["content-type"].startswith("application/problem+json")
    assert call(client, "POST", "adm", DECIDE, json={"proposal": h, "items": [], "approve": True}).status_code == 422
    assert _audit(twin) == []
    r = call(client, "POST", "adm", DECIDE, json={"proposal": h, "items": ["GAS-23"], "approve": False})
    assert r.status_code == 200 and r.json()["facts_version"] is None
    (entry,) = _audit(twin)
    assert entry.action == "learn.reject" and entry.after == {"items": {"GAS-23": "rejected"}}


def test_without_the_twin_store_decisions_are_refused_and_not_read(client, folder) -> None:
    store_service.use(None)
    body = call(client, "GET", "adm", URL).json()
    assert body["decisions_reason"].startswith("No twin store") and body["proposal"]["items"]
    h = body["proposal"]["hash"]
    r = call(client, "POST", "adm", DECIDE, json={"proposal": h, "items": ["VAL-02"], "approve": True})
    assert r.status_code == 503


# ------------------------------------------------------------------------------------------------------ the run


def test_a_shadow_run_is_queued_and_audited(client, twin) -> None:
    r = call(client, "POST", "adm", RUN)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert job["kind"] == KIND and job["status"] == "queued" and not r.json()["duplicate"]
    (entry,) = _audit(twin)
    assert (entry.action, entry.object_id, entry.id) == ("job.start", job["id"], r.json()["audit_id"])
    assert entry.details["origin"] == "validation console"
    r = call(client, "POST", "adm", "/api/jobs", json={"kind": KIND, "params": {}})
    assert r.status_code == 202 and r.json()["duplicate"] and _audit(twin)[0].details["origin"] == "jobs"
    kinds = {k["name"]: k for k in call(client, "GET", "adm", "/api/jobs/kinds").json()}
    assert kinds[KIND]["audited"]


# ------------------------------------------------------------------------------------------------------- units

UNITS = ("count", "dimensionless", "kg/tHM", "s", "days", "fraction", "standardised", "unit")
SCHEMAS = [
    "LearnConsole", "LearnAsset", "LearnSkill",
    "FamilyFact", "BlendFamily", "ScaleTrajectory", "LearnTrajectory", "LearnGate", "LearnCusum", "RegisterChange",
    "ItemDecision", "ProposalItem",
    "LearnProposal", "LearnRunInfo", "LearnRunDay", "LearnJob", "RegisterInForce", "LearnDecisionResult",
    "LearnRunStarted",
]  # fmt: skip


@pytest.mark.parametrize("schema", SCHEMAS)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    numeric = [
        (name, prop)
        for name, prop in spec.get("properties", {}).items()
        if (
            {prop.get("type"), prop.get("items", {}).get("type")}
            | {p.get("type") for p in prop.get("anyOf", [])}
            | {p.get("items", {}).get("type") for p in prop.get("anyOf", [])}
            | {i.get("type") for i in prop.get("items", {}).get("anyOf", [])}
        )
        & {"number", "integer"}
        and not name.endswith("id")
    ]
    missing = [
        f"{name}: {prop.get('description')!r}"
        for name, prop in numeric
        if (prop.get("description") or "").rsplit(",", 1)[-1].strip().split(" ")[0] not in UNITS
    ]
    assert not missing, missing


# ------------------------------------------------------------------------------------------------------- UI audit

UI_AUDIT_CONSOLE = Path(__file__).resolve().parents[1] / "frontend" / "scripts" / "fixtures" / "learn-console.json"


def test_the_ui_audit_console_is_what_the_route_returns() -> None:
    """The UI audit's mock of GET /api/admin/learn validates against its model and shows every item state."""
    body = LearnConsole.model_validate(json.loads(UI_AUDIT_CONSOLE.read_text(encoding="utf-8")))
    assert body.proposal is not None and body.trajectory is not None and body.gate is not None
    assert {i.state for i in body.proposal.items} >= {"open", "proposed", "rejected"}
    assert len(body.trajectory.dates) == len(body.trajectory.observed) == len(body.trajectory.params[0].mean)
