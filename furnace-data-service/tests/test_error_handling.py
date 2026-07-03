"""Tests for structured API error responses."""


def test_unknown_route_returns_structured_error(client):
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["message"]


def test_validation_error_returns_structured_error(client):
    response = client.post(
        "/api/v1/datasets/fetch",
        json={
            "start_date": "2026-01-01",
            "end_date": "2026-01-07",
            "rm_choice": "invalid_choice",
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "errors" in body["error"]["details"]
