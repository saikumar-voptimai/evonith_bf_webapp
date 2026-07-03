"""Tests for environment-driven CORS behavior."""

from fastapi.testclient import TestClient


def test_cors_allows_configured_streamlit_origin(app_factory):
    app = app_factory(cors_origins=["http://localhost:8501"])
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Request-ID",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert "X-Request-ID" in response.headers.get("access-control-allow-headers", "")

    with TestClient(app) as client:
        get_response = client.get("/api/v1/health", headers={"Origin": "http://localhost:8501"})

    assert "X-Request-ID" in get_response.headers.get("access-control-expose-headers", "")


def test_cors_does_not_allow_random_origin(app_factory):
    app = app_factory(cors_origins=["http://localhost:8501"])
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.headers.get("access-control-allow-origin") != "http://evil.example"
