"""Tests for request id middleware."""


def test_response_includes_request_id_header(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_incoming_request_id_is_echoed(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "phase2-test-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "phase2-test-id"
    assert response.json()["request_id"] == "phase2-test-id"
