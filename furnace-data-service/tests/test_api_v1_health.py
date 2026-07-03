"""Tests for API v1 health and readiness endpoints."""


def test_api_v1_health_returns_ok(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["data"]["status"] == "ok"
    assert body["data"]["service"] == "evonith-backend-api"
    assert body["data"]["api_version"] == "v1"
    assert body["meta"]["api_version"] == "v1"


def test_api_v1_readiness_returns_ready(client):
    response = client.get("/api/v1/readiness")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["status"] == "ready"
    assert body["data"]["checks"]["runtime_dir"] == "ok"
    assert body["data"]["checks"]["config"] == "ok"


def test_api_v1_runtime_status_returns_runtime_checks(client):
    response = client.get("/api/v1/status/runtime")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["status"] in {"ok", "degraded"}
    assert "runtime_dir" in body["data"]["checks"]
    assert "cache" in body["data"]["directories"]
