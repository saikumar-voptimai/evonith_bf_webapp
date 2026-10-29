"""Fact register editor API (SV-4.2, #305): administrators only, browse the register, propose and review versions
(second administrator for a status change), compare versions and export an approved one as the schema-valid register
file. RFC 7807 problems for refused writes. SQLite twin store."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from api.routers.facts import FactRegisterView, FactsDiff, FactsVersionView
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore
from services import facts
from services import store as store_service
from tests.test_api_auth import PASSWORDS, _declared_roles, build
from tests.test_api_live import session_cookie
from utils.burden_engine.facts import SCHEMA_PATH, FactRegister, load_register

URL = "/api/admin/facts"


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
def client(twin):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
        yield c


def call(client: TestClient, user: str | None, method: str, url: str, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.request(method, url, headers=headers, **kw)


def problem(r) -> str:
    assert r.headers["content-type"].startswith("application/problem+json"), r.text
    return r.json()["detail"]


@pytest.mark.parametrize("actor,status", [(None, 401), ("op", 403), ("sup", 403), ("adm", 200)])
def test_the_register_answers_administrators_only(client, actor, status) -> None:
    for method, url, body in [
        ("GET", URL, None),
        ("GET", f"{URL}/versions", None),
        ("GET", f"{URL}/diff", None),
        ("POST", f"{URL}/versions", {"parent_version": 0, "changes": {"COKE-01": {"value": -9}}, "note": "refit"}),
    ]:
        r = call(client, actor, method, url, json=body)
        expected = 201 if status == 200 and method == "POST" else status
        assert r.status_code == expected, (method, url, r.text)


def test_every_fact_route_requires_the_admin_role() -> None:
    routes = [r for r in build().routes if isinstance(r, APIRoute) and r.path.startswith(URL)]
    assert sorted((r.path, *r.methods) for r in routes) == [
        (URL, "GET"),
        (f"{URL}/diff", "GET"),
        (f"{URL}/export", "GET"),
        (f"{URL}/versions", "GET"),
        (f"{URL}/versions", "POST"),
        (f"{URL}/versions/{{version}}/review", "POST"),
    ]
    assert all(set(_declared_roles(r.dependant)) == {"admin"} for r in routes)


def test_browse_propose_review_diff_and_export(client, twin) -> None:
    r = call(client, "adm", "GET", URL)
    view = FactRegisterView.model_validate(r.json())
    assert r.status_code == 200 and view.label == "1.8" and len(view.facts) == 116 and view.in_force_version == 0
    assert view.groups == ["COKE", "GAS", "FLOW", "DATA", "BAND", "ASM", "MOD", "VAL"]
    assert {f.id for f in view.facts if f.retired} >= {"GAS-19", "COKE-17"}
    band = next(f for f in view.facts if f.id == "BAND-02")
    assert band.band == [1440.0, 1520.0] and band.band_kind == "plausibility" and band.unit

    body = {
        "parent_version": 0,
        "changes": {"BAND-02": {"band": [1445, 1515]}, "COKE-01": {"value": -9.0, "status": "Disputed"}},
        "note": "October refit",
    }
    r = call(client, "adm", "POST", f"{URL}/versions", json=body)
    assert r.status_code == 201, r.text
    v1 = FactsVersionView.model_validate(r.json())
    assert v1.version == 1 and v1.status == "pending" and v1.second_review and v1.facts == ["BAND-02", "COKE-01"]
    r = call(client, "adm", "POST", f"{URL}/versions/1/review", json={"approve": True})
    assert r.status_code == 409 and "second administrator" in problem(r)
    r = call(client, "adm", "GET", f"{URL}/export", params={"version": 1})
    assert r.status_code == 409 and "only an approved version" in problem(r)

    # a second administrator (alice) proposes values only; adm reviews both
    facts.propose(twin, Actor("alice", "admin"), parent_version=0, changes={"COKE-02": {"value": 1.0}}, note="x")
    r = call(client, "adm", "POST", f"{URL}/versions/2/review", json={"approve": False, "note": "Not tested"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    facts.review(twin, Actor("alice", "admin"), 1, approve=True)
    assert load_register().label == "1.8 + overlay 1"

    r = call(client, "adm", "GET", f"{URL}/versions")
    assert [(v["version"], v["status"], v["in_force"]) for v in r.json()] == [
        (2, "rejected", False),
        (1, "approved", True),
    ]
    r = call(client, "adm", "GET", URL)
    view = FactRegisterView.model_validate(r.json())
    coke = next(f for f in view.facts if f.id == "COKE-01")
    assert view.in_force_version == 1 and view.label == "1.8 + overlay 1"
    assert coke.changed == ["value", "status"] and coke.file == {"value": -8.7, "status": "Model"}
    assert call(client, "adm", "GET", URL, params={"version": 0}).json()["label"] == "1.8"

    r = call(client, "adm", "GET", f"{URL}/diff")
    diff = FactsDiff.model_validate(r.json())
    assert diff.from_version == 0 and diff.to_version == 1
    assert [(c.id, c.field) for c in diff.changes] == [("COKE-01", "value"), ("COKE-01", "status"), ("BAND-02", "band")]

    r = call(client, "adm", "GET", f"{URL}/export")
    assert r.status_code == 200 and 'filename="facts_register_v1.9.json"' in r.headers["content-disposition"]
    doc = json.loads(r.text)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(doc)) == []
    assert FactRegister(doc).version == "1.9" and r.text.endswith("}\n")


def test_refused_writes_answer_problems(client) -> None:
    def post(changes: dict, parent: int = 0, note: str = "refit"):
        return call(client, "adm", "POST", f"{URL}/versions", json={"parent_version": parent, "changes": changes,
                                                                     "note": note})  # fmt: skip

    r = post({"COKE-01": {"status": "Obsolete"}})
    assert r.status_code == 422 and "unknown status" in problem(r)
    r = post({"FLOW-02": {"status": "Retired"}})
    assert r.status_code == 422 and "retiring it needs a code change first" in problem(r)
    r = post({"COKE-01": {"band": [-8, -9]}})
    assert r.status_code == 422 and "lies above" in problem(r)
    r = post({"COKE-01": {"se": 1.0}})  # not an editable field: the request model refuses it
    assert r.status_code == 422
    r = post({"COKE-01": {"value": -9}}, parent=3)
    assert r.status_code == 409 and "version 3" in problem(r)
    r = call(client, "adm", "POST", f"{URL}/versions/7/review", json={"approve": True})
    assert r.status_code == 404 and "version 7" in problem(r)
    r = call(client, "adm", "GET", URL, params={"version": 7})
    assert r.status_code == 404
    r = call(client, "adm", "GET", f"{URL}/export")
    assert r.status_code == 409 and "No approved version" in problem(r)


def test_without_a_store_the_register_is_the_file_and_writes_answer_503(client) -> None:
    store_service.use(None)
    r = call(client, "adm", "GET", URL)
    assert r.status_code == 200 and r.json()["version_status"] == "file" and "No twin store" in r.json()["reason"]
    r = call(client, "adm", "POST", f"{URL}/versions", json={"parent_version": 0, "changes": {"COKE-01": {"value": 1}},
                                                              "note": "refit"})  # fmt: skip
    assert r.status_code == 503 and "BF_TWIN_DB_URL" in problem(r)


# ------------------------------------------------------------------------------------------------------- units

SCHEMAS = [
    "FactRegisterView",
    "FactView",
    "FactsVersionView",
    "FactChange",
    "FactsProposal",
    "FactDiffRow",
    "FactsDiff",
]


def _numeric(prop: dict[str, Any]) -> bool:
    types = {prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}
    items = [prop.get("items") or {}] + [p.get("items") or {} for p in prop.get("anyOf", [])]
    return bool(types & {"number", "integer"}) or any(i.get("type") == "number" for i in items)


def _unit(description: str) -> bool:
    last = description.rsplit(",", 1)[-1].strip()
    return last == "count" or last == "in the fact's unit"


@pytest.mark.parametrize("schema", SCHEMAS)
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = [
        f"{name}: {prop.get('description')!r}"
        for name, prop in spec.get("properties", {}).items()
        if _numeric(prop) and not _unit(prop.get("description") or "")
    ]
    assert not missing, missing


def test_query_parameters_state_their_unit() -> None:
    paths = build().openapi()["paths"]
    for path, operations in paths.items():
        if not path.startswith(URL):
            continue
        for operation in operations.values():
            for p in operation.get("parameters", []):
                types = {p["schema"].get("type")} | {a.get("type") for a in p["schema"].get("anyOf", [])}
                if types & {"integer", "number"}:
                    assert p.get("description", "").endswith(", count"), (path, p)
