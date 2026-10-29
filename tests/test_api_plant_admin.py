"""User, hopper and burden-distribution admin API (SV-4.7, #310): administrators only, the writes of
``services.plant_admin`` (the Streamlit tools' writes, ``tests/test_services_plant_admin.py``) over HTTP, each with its
audit row; RFC 7807 problems for refused writes and for a database without the plant tables. Needs PostgreSQL."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from api.routers.plant_admin import BurdenAdminView, HopperAdminView, UsersView
from data.db import password_digest
from data.twin.repositories import TwinStore
from services import store as store_service
from tests.pg_server import fresh_database, needs_postgres, postgres_server
from tests.test_api_auth import PASSWORDS, _declared_roles, build
from tests.test_api_live import session_cookie
from tests.test_services_plant_admin import ASSIGN, T_ASSIGN, T_IMPORT, audit_rows, dump, seed
from tests.test_twin_store import alembic_config

IST = timezone(timedelta(hours=5, minutes=30))
URL = "/api/admin"
UI_AUDIT_FIXTURE = Path(__file__).resolve().parents[1] / "frontend" / "scripts" / "fixtures" / "plant-admin.json"


def test_every_plant_admin_route_requires_the_admin_role() -> None:
    paths = {f"{URL}/users", f"{URL}/hoppers", f"{URL}/hoppers/history", f"{URL}/burden", f"{URL}/burden/history"}
    routes = [r for r in build().routes if isinstance(r, APIRoute) and r.path in paths]
    assert sorted((r.path, *r.methods) for r in routes) == [
        (f"{URL}/burden", "GET"),
        (f"{URL}/burden", "POST"),
        (f"{URL}/burden/history", "DELETE"),
        (f"{URL}/hoppers", "GET"),
        (f"{URL}/hoppers", "POST"),
        (f"{URL}/hoppers/history", "DELETE"),
        (f"{URL}/users", "GET"),
        (f"{URL}/users", "POST"),
    ]
    assert all(set(_declared_roles(r.dependant)) == {"admin"} for r in routes)


def test_the_ui_audit_fixture_matches_the_response_models() -> None:
    """``frontend/scripts/ui-audit.mjs`` answers the three views with this file (made from the seeded test database)."""
    data = json.loads(UI_AUDIT_FIXTURE.read_text(encoding="utf-8"))
    users = UsersView.model_validate(data["users"])
    hoppers = HopperAdminView.model_validate(data["hoppers"])
    burden = BurdenAdminView.model_validate(data["burden"])
    assert len(users.users) == 3 and len(hoppers.history) == 3 and len(burden.history) == 3
    assert burden.current is not None and burden.history[0].date_time > burden.at  # one snapshot applies later
    assert {f.name for f in burden.fields} == set(burden.current.values)


@pytest.fixture(scope="module")
def pg_server() -> Iterator[str]:
    with postgres_server() as server:
        yield server


@pytest.fixture()
def database(pg_server) -> Iterator[str]:
    with fresh_database(pg_server) as url:
        seed(url)
        yield url


def _client(url: str) -> Iterator[TestClient]:
    store = TwinStore.from_url(url)
    assert store is not None
    store_service.use(store)
    try:
        app = build()
        with TestClient(app, base_url="https://testserver") as c:
            c.cookies_by_user = {u: session_cookie(app, u) for u in PASSWORDS}  # type: ignore[attr-defined]
            yield c
    finally:
        store_service.reset()
        store.dispose()


@pytest.fixture()
def client(database) -> Iterator[TestClient]:
    yield from _client(database)


def call(client: TestClient, user: str | None, method: str, url: str, **kw: Any):
    headers = {"Cookie": client.cookies_by_user[user]} if user else {}  # type: ignore[attr-defined]
    return client.request(method, url, headers=headers, **kw)


def problem(r, status: int) -> str:
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json"), r.text
    return r.json()["detail"]


@needs_postgres
@pytest.mark.parametrize("actor,status", [(None, 401), ("op", 403), ("sup", 403)])
def test_non_administrators_are_refused_and_nothing_is_written(client, database, actor, status) -> None:
    before = dump(database)
    for method, url, kw in [
        ("GET", f"{URL}/users", {}),
        ("POST", f"{URL}/users", {"json": {"username": "qa", "password": "pw", "role": "user"}}),
        ("GET", f"{URL}/hoppers", {}),
        ("POST", f"{URL}/hoppers", {"json": {"effective_from": T_ASSIGN.isoformat(), "materials": ASSIGN}}),
        ("DELETE", f"{URL}/hoppers/history", {"params": {"ids": [1]}}),
        ("GET", f"{URL}/burden", {}),
        ("POST", f"{URL}/burden", {"json": {"effective_from": T_ASSIGN.isoformat(), "values": {}}}),
        ("DELETE", f"{URL}/burden/history", {"params": {"ids": [1]}}),
    ]:
        assert call(client, actor, method, url, **kw).status_code == status, (method, url)
    assert dump(database) == before and audit_rows(database) == []


@needs_postgres
def test_register_assign_change_and_delete_over_http(client, database) -> None:
    users = UsersView.model_validate(call(client, "adm", "GET", f"{URL}/users").json())
    assert [u.username for u in users.users] == ["adm", "admin"] and users.roles == ["user", "supervisor", "admin"]

    r = call(client, "adm", "POST", f"{URL}/users", json={"username": " qa ", "password": " s3cret ", "role": "user"})
    assert r.status_code == 201, r.text
    assert (r.json()["user"]["username"], r.json()["user"]["roles"]) == ("qa", ["user"])
    assert "s3cret" not in r.text and password_digest("s3cret") not in r.text
    assert "already exists" in problem(
        call(client, "adm", "POST", f"{URL}/users", json={"username": "qa", "password": "x"}), 409
    )
    assert "fill in all fields" in problem(
        call(client, "adm", "POST", f"{URL}/users", json={"username": " ", "password": "x"}), 422
    )

    hoppers = HopperAdminView.model_validate(call(client, "adm", "GET", f"{URL}/hoppers").json())
    assert hoppers.current is not None and hoppers.current.date_time == T_IMPORT
    form = {**hoppers.current.materials, **ASSIGN}
    effective = T_ASSIGN.astimezone(IST).isoformat()  # 06:00 IST, as the SPA sends it
    r = call(client, "adm", "POST", f"{URL}/hoppers", json={"effective_from": effective, "materials": form})
    assert r.status_code == 200 and r.json()["written"] and r.json()["changed"] == list(ASSIGN), r.text
    in_force = HopperAdminView.model_validate(call(client, "adm", "GET", f"{URL}/hoppers").json()).current
    assert in_force is not None  # the form re-read after the write (an alias comes back as its client name)
    again = call(
        client, "adm", "POST", f"{URL}/hoppers", json={"effective_from": effective, "materials": in_force.materials}
    )
    assert again.json() == {"written": False, "id": None, "changed": [], "audit_id": None}
    assert "not offered" in problem(
        call(
            client,
            "adm",
            "POST",
            f"{URL}/hoppers",
            json={"effective_from": effective, "materials": {"hopper_01": "Gold"}},
        ),
        422,
    )
    naive = T_ASSIGN.replace(tzinfo=None).isoformat()
    assert (
        call(client, "adm", "POST", f"{URL}/hoppers", json={"effective_from": naive, "materials": {}}).status_code
        == 422
    )
    hoppers = HopperAdminView.model_validate(call(client, "adm", "GET", f"{URL}/hoppers").json())
    assert hoppers.current is not None and hoppers.current.date_time == T_ASSIGN
    assert hoppers.current.user_name == "adm" and hoppers.current.materials["hopper_05"] == "UNASSIGNED"
    assert [h.date_time for h in hoppers.history] == [T_ASSIGN, T_IMPORT]

    burden = BurdenAdminView.model_validate(call(client, "adm", "GET", f"{URL}/burden").json())
    assert burden.current is not None and burden.current.values["coke_charge_pattern"] == "C 3 2 2"
    r = call(
        client,
        "adm",
        "POST",
        f"{URL}/burden",
        json={"effective_from": effective, "values": {"coke_p03_angles": 31.5, "coke_discharge_time": "92"}},
    )
    assert r.status_code == 200 and r.json()["changed"] == ["coke_p03_angles"], r.text  # "92" = 92.0 in force
    assert "is not a number" in problem(
        call(
            client,
            "adm",
            "POST",
            f"{URL}/burden",
            json={"effective_from": effective, "values": {"coke_p01_rings": "x"}},
        ),
        422,
    )
    burden = BurdenAdminView.model_validate(call(client, "adm", "GET", f"{URL}/burden").json())
    assert burden.current is not None and burden.current.values["coke_p03_angles"] == 31.5
    assert burden.current.values["coke_charge_pattern"] == "C 3 2 2"  # copied from the snapshot in force
    assert len(burden.history) == 2

    imported = burden.history[-1].id
    assert "No burden history row(s) 999" in problem(
        call(client, "adm", "DELETE", f"{URL}/burden/history", params={"ids": [imported, 999]}), 404
    )
    r = call(client, "adm", "DELETE", f"{URL}/burden/history", params={"ids": [imported]})
    assert r.status_code == 200 and r.json()["deleted"] == [imported], r.text
    assert call(client, "adm", "DELETE", f"{URL}/hoppers/history").status_code == 422  # no id

    log = audit_rows(database)
    assert [(a["action"], a["actor"], a["client_ip"]) for a in log] == [
        ("user.create", "adm", "testclient"),
        ("hopper.assign", "adm", "testclient"),
        ("burden.update", "adm", "testclient"),
        ("burden.delete", "adm", "testclient"),
    ]
    rows = dump(database)
    assert rows["hoppers"][-1]["ip_address"] == "testclient" and rows["hoppers"][-1]["user_modified"] == "adm"


@needs_postgres
def test_a_database_without_the_plant_tables_is_503(pg_server) -> None:
    with fresh_database(pg_server) as url:
        command.upgrade(alembic_config(url), "head")  # the twin schema only
        for c in _client(url):
            for path in ("users", "hoppers", "burden"):
                assert "identity, plant_master and ops_config" in problem(call(c, "adm", "GET", f"{URL}/{path}"), 503)
