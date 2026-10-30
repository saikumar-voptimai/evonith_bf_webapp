"""twin-api foundation (#272): problem responses, request ids, version, SPA fallback, schema drift."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy.exc import ProgrammingError

from api.errors import ApiError
from api.main import create_app
from api.settings import ApiSettings

REPO = Path(__file__).resolve().parents[1]


class Body(BaseModel):
    n: int


SCHEMA = REPO / "frontend" / "src" / "lib" / "api" / "openapi.json"


@pytest.fixture()
def app():
    application = create_app(ApiSettings(static_dir=None))

    @application.post("/api/_test/echo")
    def echo(body: Body) -> Body:
        return body

    @application.get("/api/_test/boom")
    def boom() -> dict:
        raise RuntimeError("secret connection string")

    @application.get("/api/_test/conflict")
    def conflict() -> dict:
        raise ApiError(409, "Another run is in progress.")

    @application.get("/api/_test/no-table")
    def no_table() -> dict:
        orig = type("UndefinedTable", (Exception,), {"pgcode": "42P01"})("relation twin.shift_report does not exist")
        raise ProgrammingError("SELECT ... FROM twin.shift_report", {}, orig)

    @application.get("/api/_test/bad-sql")
    def bad_sql() -> dict:
        raise ProgrammingError("SELEC 1", {}, type("SyntaxError_", (Exception,), {"pgcode": "42601"})("syntax error"))

    return application


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def test_health_and_generated_request_id(client) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "twin-api"}
    assert len(r.headers["x-request-id"]) == 32


def test_client_request_id_is_echoed_and_unsafe_ids_are_replaced(client) -> None:
    assert client.get("/api/health", headers={"X-Request-ID": "shift-B.42"}).headers["x-request-id"] == "shift-B.42"
    replaced = client.get("/api/health", headers={"X-Request-ID": "bad id\n"}).headers["x-request-id"]
    assert replaced != "bad id\n" and len(replaced) == 32


def test_version(client) -> None:
    body = client.get("/api/version").json()
    assert body["version"] and body["environment"] in {"dev", "prod"} and body["git_sha"]


def test_unknown_route_is_a_problem(client) -> None:
    r = client.get("/api/nope", headers={"X-Request-ID": "req-1"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json() == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "instance": "/api/nope",
        "request_id": "req-1",
    }


def test_validation_error_lists_fields(client) -> None:
    r = client.post("/api/_test/echo", json={"n": "x"})
    body = r.json()
    assert r.status_code == 422 and body["title"] == "Unprocessable Entity"
    assert body["errors"][0]["loc"] == ["body", "n"]


def test_api_error_carries_status_and_detail(client) -> None:
    r = client.get("/api/_test/conflict")
    assert r.status_code == 409
    assert r.json()["detail"] == "Another run is in progress."


def test_unhandled_error_hides_the_exception_and_logs_it(client, caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="api.access.error"):
        r = client.get("/api/_test/boom", headers={"X-Request-ID": "req-500"})
    assert r.status_code == 500
    assert "secret" not in r.text
    assert r.json()["request_id"] == "req-500"
    assert any("secret connection string" in (rec.exc_text or "") for rec in caplog.records)


def test_missing_twin_tables_are_a_503_that_names_the_migration(client) -> None:
    r = client.get("/api/_test/no-table")
    assert r.status_code == 503 and r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert body["title"] == "Twin tables not installed" and "alembic -n twin upgrade head" in body["detail"]
    # Any other SQL programming error stays an internal error without the statement text.
    r = client.get("/api/_test/bad-sql")
    assert r.status_code == 500 and "SELEC" not in r.text


def test_spa_fallback_serves_files_and_200_html(tmp_path) -> None:
    (tmp_path / "200.html").write_text("<html>spa</html>")
    (tmp_path / "_app").mkdir()
    (tmp_path / "_app" / "a.js").write_text("console.log(1)")
    client = TestClient(create_app(ApiSettings(static_dir=tmp_path)))
    assert client.get("/").text == "<html>spa</html>"
    assert client.get("/trace").text == "<html>spa</html>"
    assert client.get("/_app/a.js").text == "console.log(1)"
    assert client.get("/../../etc/passwd").text == "<html>spa</html>"
    assert client.get("/api/missing").status_code == 404
    assert client.get("/api/health").json()["status"] == "ok"


def test_committed_openapi_schema_matches_the_app() -> None:
    """Regenerate with: python scripts/export_openapi.py && pnpm --dir frontend api:types"""
    generated = create_app(ApiSettings(static_dir=None)).openapi()
    committed = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(generated, sort_keys=True)), "OpenAPI schema drift: regenerate the client"


def test_health_latency_p95_under_20_ms(client) -> None:
    samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        client.get("/api/health")
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    assert samples[int(0.95 * len(samples))] < 20
