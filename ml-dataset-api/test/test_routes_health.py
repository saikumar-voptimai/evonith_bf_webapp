"""Tests for GET /health."""


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health_version_string(client):
    resp = client.get("/health")
    assert isinstance(resp.json()["version"], str)
    assert len(resp.json()["version"]) > 0
