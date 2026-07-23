"""Tests for environment-driven CORS behavior."""

import pytest
from pydantic import ValidationError

from fastapi.testclient import TestClient


def test_cors_rejects_wildcard_origin_for_credentialed_requests():
    from apps.backend_api.app.core.config import BackendSettings

    with pytest.raises(ValidationError, match="must not contain"):
        BackendSettings(cors_origins=["*"])

def test_cors_allows_configured_streamlit_origin(app_factory):
    app = app_factory(cors_origins=["http://localhost:8501"])
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Request-ID, Idempotency-Key",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
    allowed_headers = response.headers.get("access-control-allow-headers", "")
    assert "X-Request-ID" in allowed_headers
    assert "Idempotency-Key" in allowed_headers

    with TestClient(app) as client:
        get_response = client.get("/api/v1/health", headers={"Origin": "http://localhost:8501"})

    exposed_headers = get_response.headers.get("access-control-expose-headers", "")
    assert "X-Request-ID" in exposed_headers
    assert "Content-Disposition" in exposed_headers
    assert "X-Dataset-Version" in exposed_headers


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
