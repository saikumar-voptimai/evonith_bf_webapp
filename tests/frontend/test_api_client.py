"""Tests for the frontend backend API client."""

from __future__ import annotations

import httpx
import pytest

from services.api_client import ApiClient, is_wrapped_api_response, unwrap_api_response
from services.api_errors import (
    BackendApiDecodeError,
    BackendApiHTTPError,
    BackendApiTimeoutError,
    BackendApiValidationError,
    BackendUnavailableError,
)


def _json_response(request: httpx.Request, payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        headers={"X-Request-ID": "backend-request-id"},
        request=request,
    )


def test_get_builds_correct_url_and_sends_request_id():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["request_id"] = request.headers["X-Request-ID"]
        return _json_response(request, {"ok": True})

    client = ApiClient(
        base_url="http://backend.local/api/v1/",
        transport=httpx.MockTransport(handler),
    )

    assert client.get("health") == {"ok": True}
    assert seen["url"] == "http://backend.local/api/v1/health"
    assert seen["request_id"]
    assert client.last_response_request_id == "backend-request-id"


def test_custom_request_id_is_sent_and_backend_request_id_is_captured():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Request-ID"] == "frontend-request-id"
        return _json_response(request, {"ok": True})

    client = ApiClient(transport=httpx.MockTransport(handler))

    client.request("GET", "/health", headers={"X-Request-ID": "frontend-request-id"})

    assert client.last_request_id == "frontend-request-id"
    assert client.last_response_request_id == "backend-request-id"


def test_post_helper_forwards_custom_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer token"
        return _json_response(request, {"ok": True})

    client = ApiClient(transport=httpx.MockTransport(handler))

    assert client.post(
        "/auth/logout",
        json={},
        headers={"Authorization": "Bearer token"},
    ) == {"ok": True}


def test_structured_backend_error_is_parsed():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request,
            {
                "request_id": "structured-id",
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Service unavailable",
                    "details": {"check": "offline"},
                },
            },
            status_code=503,
        )

    client = ApiClient(transport=httpx.MockTransport(handler))

    with pytest.raises(BackendApiHTTPError) as exc_info:
        client.get("/readiness")

    exc = exc_info.value
    assert exc.status_code == 503
    assert exc.request_id == "structured-id"
    assert exc.error_code == "SERVICE_UNAVAILABLE"
    assert exc.details == {"check": "offline"}


def test_validation_error_uses_validation_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request,
            {
                "request_id": "validation-id",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {"errors": []},
                },
            },
            status_code=422,
        )

    client = ApiClient(transport=httpx.MockTransport(handler))

    with pytest.raises(BackendApiValidationError):
        client.post("/datasets/fetch", json={"rm_choice": "bad"})


def test_connection_failure_becomes_unavailable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = ApiClient(transport=httpx.MockTransport(handler), max_retries=0)

    with pytest.raises(BackendUnavailableError):
        client.get("/health")


def test_timeout_becomes_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = ApiClient(transport=httpx.MockTransport(handler), max_retries=0)

    with pytest.raises(BackendApiTimeoutError):
        client.get("/health")


def test_invalid_json_becomes_decode_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", request=request)

    client = ApiClient(transport=httpx.MockTransport(handler))

    with pytest.raises(BackendApiDecodeError):
        client.get("/health")


def test_get_retries_when_enabled():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ConnectError("temporary failure", request=request)
        return _json_response(request, {"ok": True})

    client = ApiClient(transport=httpx.MockTransport(handler), max_retries=1)

    assert client.get("/health") == {"ok": True}
    assert attempts["count"] == 2


def test_post_is_not_retried_by_default():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("temporary failure", request=request)

    client = ApiClient(transport=httpx.MockTransport(handler), max_retries=3)

    with pytest.raises(BackendUnavailableError):
        client.post("/datasets/fetch", json={})

    assert attempts["count"] == 1


def test_wrapped_response_detection_and_unwrap():
    wrapped = {"request_id": "id", "data": {"status": "ok"}, "meta": {"api_version": "v1"}}
    plain = {"status": "ok"}

    assert is_wrapped_api_response(wrapped) is True
    assert unwrap_api_response(wrapped) == {"status": "ok"}
    assert is_wrapped_api_response(plain) is False
    assert unwrap_api_response(plain) == plain
