"""Tests for frontend backend status helpers."""

from __future__ import annotations

import httpx

from apps.frontend_streamlit.services.api_client import ApiClient
from apps.frontend_streamlit.services.backend_status import check_backend_health, get_backend_status_summary


def test_backend_status_returns_unavailable_instead_of_crashing():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    status = check_backend_health(ApiClient(transport=httpx.MockTransport(handler), max_retries=0))

    assert status.is_available is False
    assert status.status == "unavailable"
    assert "Backend API" in status.message


def test_backend_status_summary_checks_health_and_readiness():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(
                200,
                json={
                    "request_id": "health-id",
                    "data": {"status": "ok"},
                    "meta": {"api_version": "v1"},
                },
                headers={"X-Request-ID": "health-id"},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "request_id": "ready-id",
                "data": {"status": "ready"},
                "meta": {"api_version": "v1"},
            },
            headers={"X-Request-ID": "ready-id"},
            request=request,
        )

    status = get_backend_status_summary(ApiClient(transport=httpx.MockTransport(handler), max_retries=0))

    assert status.is_available is True
    assert status.is_ready is True
    assert status.status == "ready"
    assert status.request_id == "ready-id"
