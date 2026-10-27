"""Admin API (SV-4.1, #304): the audit log route answers administrators only (role matrix), pages and filters the log
with the values before and after, and states the unit of every numeric field. SQLite twin store."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.routers.admin import AuditPage
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore
from services import audit
from services import store as store_service
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie

T0 = datetime(2026, 10, 27, 0, 30, tzinfo=UTC)  # 06:00 IST
H = timedelta(hours=1)
URL = "/api/admin/audit"
#: The page the UI audit answers for GET /api/admin/audit (frontend/scripts/ui-audit.mjs).
UI_AUDIT_PAGE = Path(__file__).resolve().parents[1] / "frontend" / "scripts" / "fixtures" / "audit-page.json"


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


def get(client: TestClient, user: str | None, url: str, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.get(url, headers=headers, **kw)


# ------------------------------------------------------------------------------------------------------- roles

ROLE_MATRIX = {None: 401, "op": 403, "sup": 403, "adm": 200}  # the audit log is for administrators only


@pytest.mark.parametrize("actor", list(ROLE_MATRIX))
def test_the_audit_log_answers_administrators_only(client, actor) -> None:
    r = get(client, actor, URL)
    assert r.status_code == ROLE_MATRIX[actor], r.text
    if r.status_code == 403:
        assert r.headers["content-type"].startswith("application/problem+json")
        assert r.json()["detail"].startswith("This action needs the admin role")


def test_every_admin_route_requires_the_admin_role() -> None:
    from fastapi.routing import APIRoute

    from tests.test_api_auth import _declared_roles

    routes = [r for r in build().routes if isinstance(r, APIRoute) and r.path.startswith("/api/admin/")]
    assert [r.path for r in routes] == [URL]
    assert all(set(_declared_roles(r.dependant)) == {"admin"} for r in routes)


def test_without_a_store_the_audit_log_answers_503(client) -> None:
    store_service.use(None)
    r = get(client, "adm", URL)
    assert r.status_code == 503 and "BF_TWIN_DB_URL" in r.json()["detail"]


# ------------------------------------------------------------------------------------------------------- pages


def seed(twin: TwinStore) -> None:
    alice, bob = Actor("alice", "admin", "10.0.0.5"), Actor("bob", "admin")
    twin.audit.record(alice, "user.create", "user", "u-17", {"role": "user"}, at=T0, after={"role": "user"})
    twin.audit.record(
        bob,
        "hopper.assign",
        "hopper",
        "hopper_04",
        at=T0 + H,
        before={"material": "ore_6", "share_pct": 12.5},
        after={"material": "ore_11", "share_pct": 12.5},
    )
    twin.audit.record(alice, "plan.decide", "optimiser_run", "run-1", {"decision": "accepted"}, at=T0 + 2 * H)


def test_entries_newest_first_with_their_changes(twin, client) -> None:
    seed(twin)
    body = get(client, "adm", URL).json()
    assert body["total"] == 3 and (body["limit"], body["offset"]) == (50, 0)
    assert [e["action"] for e in body["entries"]] == ["plan.decide", "hopper.assign", "user.create"]
    decision, hopper, user = body["entries"]
    assert decision["before"] is None and decision["after"] is None and decision["changes"] == []
    assert decision["details"] == {"decision": "accepted"}
    assert hopper["changes"] == [{"path": "material", "kind": "changed", "before": "ore_6", "after": "ore_11"}]
    assert user["changes"] == [{"path": "role", "kind": "added", "before": None, "after": "user"}]
    assert user["client_ip"] == "10.0.0.5" and user["role"] == "admin"
    assert datetime.fromisoformat(user["at"]) == T0
    assert body["facets"] == {
        "actors": ["alice", "bob"],
        "actions": ["hopper.assign", "plan.decide", "user.create"],
        "object_types": ["hopper", "optimiser_run", "user"],
    }


def test_filters_and_paging(twin, client) -> None:
    seed(twin)
    with audit.admin_write(twin, Actor("carol", "admin"), "hopper.assign", "hopper", "hopper_04", at=T0 + 3 * H) as w:
        w.before, w.after = {"material": "ore_11"}, {"material": "ore_6"}

    def actions(**params: Any) -> list[str]:
        r = get(client, "adm", URL, params=params)
        assert r.status_code == 200, r.text
        return [e["action"] for e in r.json()["entries"]]

    assert actions(actor="alice") == ["plan.decide", "user.create"]
    assert actions(actor=" bob ") == ["hopper.assign"]  # surrounding spaces are ignored
    assert actions(object_type="hopper", object_id="hopper_04") == ["hopper.assign", "hopper.assign"]
    assert actions(action="plan.decide") == ["plan.decide"]
    window = {"since": (T0 + H).isoformat(), "until": "2026-10-27T08:30:00+05:30"}  # [07:00, 08:30) IST
    assert actions(**window) == ["plan.decide", "hopper.assign"]
    assert actions(since="2026-10-27T02:00:00") == ["hopper.assign", "plan.decide"]  # naive is UTC, not IST

    first = get(client, "adm", URL, params={"limit": 3}).json()
    second = get(client, "adm", URL, params={"limit": 3, "offset": 3}).json()
    assert first["total"] == second["total"] == 4
    assert [e["actor"] for e in first["entries"]] == ["carol", "alice", "bob"]
    assert [e["actor"] for e in second["entries"]] == ["alice"]
    assert first["entries"][0]["changes"][0] == {
        "path": "material",
        "kind": "changed",
        "before": "ore_11",
        "after": "ore_6",
    }


@pytest.mark.parametrize(
    "params",
    [
        {"since": "2026-10-27T02:00:00Z", "until": "2026-10-27T01:00:00Z"},
        {"since": "2026-10-27T01:00:00Z", "until": "2026-10-27T01:00:00Z"},
        {"limit": 0},
        {"limit": 201},
        {"offset": -1},
        {"actor": "x" * 129},
        {"since": "yesterday"},
    ],
)
def test_bad_requests_are_422(client, params) -> None:
    r = get(client, "adm", URL, params=params)
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/problem+json")


# ------------------------------------------------------------------------------------------------------- units

UNITS = ("count",)


def _numeric(prop: dict[str, Any]) -> bool:
    types = {prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}
    return bool(types & {"number", "integer"})


@pytest.mark.parametrize("schema", ["AuditPage", "AuditEntry", "AuditChange", "AuditFacets"])
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = [
        f"{name}: {prop.get('description')!r}"
        for name, prop in spec.get("properties", {}).items()
        if _numeric(prop)
        and name != "id"
        and (prop.get("description") or "").rsplit(",", 1)[-1].strip().split(" ")[0] not in UNITS
    ]
    assert not missing, missing


def test_query_parameters_state_their_unit() -> None:
    params = build().openapi()["paths"][URL]["get"]["parameters"]
    numeric = {p["name"]: p.get("description", "") for p in params if p["schema"].get("type") == "integer"}
    assert numeric == {"limit": "Page size, count", "offset": "Entries to skip, count"}


# ------------------------------------------------------------------------------------------------------- UI audit


def test_the_ui_audit_page_is_what_the_route_returns() -> None:
    """The UI audit's mock of the route validates against its model, each entry's changes are those
    ``services.audit.changes`` gives, the facets list the entries' values sorted, and the page has every kind of
    record the viewer formats (changed, created, deleted, details only)."""
    data = json.loads(UI_AUDIT_PAGE.read_text(encoding="utf-8"))
    page = AuditPage.model_validate(data)
    assert page.total == len(page.entries) and (page.limit, page.offset) == (50, 0)
    assert [e.id for e in page.entries] == sorted((e.id for e in page.entries), reverse=True)
    assert [e.at for e in page.entries] == sorted((e.at for e in page.entries), reverse=True)  # newest first
    for entry in data["entries"]:
        assert entry["changes"] == audit.changes(entry["before"], entry["after"]), entry["id"]
    assert page.facets.model_dump() == {
        "actors": sorted({e.actor for e in page.entries}),
        "actions": sorted({e.action for e in page.entries}),
        "object_types": sorted({e.object_type for e in page.entries}),
    }
    records = {
        "details": any(e.before is None and e.after is None for e in page.entries),
        "created": any(e.before is None and e.after is not None for e in page.entries),
        "deleted": any(e.before is not None and e.after is None for e in page.entries),
        "changed": any(e.before is not None and e.after is not None and e.changes for e in page.entries),
    }
    assert all(records.values()), records
