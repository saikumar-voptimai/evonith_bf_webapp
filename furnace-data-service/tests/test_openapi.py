"""Tests for API v1 OpenAPI schema and legacy route flags."""

from fastapi.testclient import TestClient


def test_openapi_includes_api_v1_health(client):
    schema = client.get("/openapi.json").json()

    assert "/api/v1/health" in schema["paths"]
    assert schema["info"]["title"] == "Evonith BF Backend API"
    assert schema["info"]["version"] == "0.1.0"


def test_legacy_routes_enabled_by_default(app_factory):
    app = app_factory(legacy_routes=True)
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_legacy_routes_can_be_disabled(app_factory):
    app = app_factory(legacy_routes=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        legacy_response = client.get("/health")
        versioned_response = client.get("/api/v1/health")

    assert legacy_response.status_code == 404
    assert legacy_response.json()["error"]["code"] == "NOT_FOUND"
    assert versioned_response.status_code == 200
