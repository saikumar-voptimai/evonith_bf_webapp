"""Reusable frontend API client for the versioned backend."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

try:
    from config.frontend_settings import load_frontend_settings
    from services.api_errors import (
        BackendApiDecodeError,
        BackendApiHTTPError,
        BackendApiTimeoutError,
        BackendApiValidationError,
        BackendUnavailableError,
    )
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.config.frontend_settings import load_frontend_settings
    from src.services.api_errors import (
        BackendApiDecodeError,
        BackendApiHTTPError,
        BackendApiTimeoutError,
        BackendApiValidationError,
        BackendUnavailableError,
    )


SAFE_RETRY_METHODS = {"GET"}


def is_wrapped_api_response(payload: Any) -> bool:
    """Return true when *payload* looks like the Phase 2 response wrapper."""
    return (
        isinstance(payload, dict)
        and "request_id" in payload
        and "data" in payload
        and isinstance(payload.get("meta"), dict)
    )


def unwrap_api_response(payload: Any) -> Any:
    """Return the data object from a wrapped API response, otherwise payload."""
    if is_wrapped_api_response(payload):
        return payload["data"]
    return payload


class ApiClient:
    """Small synchronous client for Streamlit/frontend backend calls."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        connect_timeout: float | None = None,
        max_retries: int | None = None,
        verify_ssl: bool | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = load_frontend_settings()
        self.base_url = (base_url or settings.backend_api_base_url).rstrip("/")
        self.timeout_seconds = timeout if timeout is not None else settings.backend_api_timeout_seconds
        self.connect_timeout_seconds = (
            connect_timeout
            if connect_timeout is not None
            else settings.backend_api_connect_timeout_seconds
        )
        self.max_retries = max(0, max_retries if max_retries is not None else settings.backend_api_max_retries)
        self.verify_ssl = verify_ssl if verify_ssl is not None else settings.backend_api_verify_ssl
        self.health_path = settings.backend_api_health_path
        self.readiness_path = settings.backend_api_readiness_path
        self.transport = transport
        self.last_request_id: str | None = None
        self.last_response_request_id: str | None = None

    def _path(self, path: str) -> str:
        clean_path = str(path or "").strip()
        return "/" + clean_path.lstrip("/")

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self.timeout_seconds,
            connect=self.connect_timeout_seconds,
        )

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self._timeout(),
            verify=self.verify_ssl,
            transport=self.transport,
            trust_env=False,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | list[Any] | None = None,
        headers: dict[str, str] | None = None,
        expect_json: bool = True,
    ) -> Any:
        """Send a backend request and return decoded JSON by default."""
        method = method.upper()
        request_id = (headers or {}).get("X-Request-ID") or str(uuid.uuid4())
        request_headers = {"X-Request-ID": request_id}
        if headers:
            request_headers.update(headers)
        self.last_request_id = request_id
        self.last_response_request_id = None

        attempts = self.max_retries + 1 if method in SAFE_RETRY_METHODS else 1
        last_error: Exception | None = None

        for _attempt in range(attempts):
            try:
                with self._client() as client:
                    response = client.request(
                        method,
                        self._path(path),
                        params=params,
                        json=json,
                        headers=request_headers,
                    )
                self.last_response_request_id = (
                    response.headers.get("X-Request-ID") or self.last_request_id
                )
                return self._handle_response(response, expect_json=expect_json)
            except httpx.TimeoutException as exc:
                last_error = exc
                if method not in SAFE_RETRY_METHODS:
                    break
            except httpx.TransportError as exc:
                last_error = exc
                if method not in SAFE_RETRY_METHODS:
                    break

        if isinstance(last_error, httpx.TimeoutException):
            raise BackendApiTimeoutError(
                "Backend API request timed out",
                request_id=self.last_response_request_id or self.last_request_id,
            ) from last_error

        raise BackendUnavailableError(
            "Backend API is unavailable",
            request_id=self.last_response_request_id or self.last_request_id,
            details={"error": str(last_error)} if last_error else {},
        ) from last_error

    def _handle_response(self, response: httpx.Response, *, expect_json: bool) -> Any:
        if response.status_code >= 400:
            self._raise_http_error(response)

        if not expect_json:
            return response.content
        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise BackendApiDecodeError(
                "Backend API returned invalid JSON",
                status_code=response.status_code,
                request_id=self.last_response_request_id,
            ) from exc

    def _raise_http_error(self, response: httpx.Response) -> None:
        request_id = response.headers.get("X-Request-ID")
        error_code = None
        message = response.reason_phrase or "Backend API error"
        details: dict[str, Any] = {}

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            payload = None

        if isinstance(payload, dict):
            request_id = payload.get("request_id") or request_id
            error = payload.get("error")
            if isinstance(error, dict):
                error_code = error.get("code")
                message = error.get("message") or message
                if isinstance(error.get("details"), dict):
                    details = error["details"]
            elif "detail" in payload:
                details = {"detail": payload["detail"]}
                if isinstance(payload["detail"], str):
                    message = payload["detail"]
        elif response.text:
            details = {"body": response.text[:500]}

        error_cls = BackendApiValidationError if response.status_code == 422 or error_code == "VALIDATION_ERROR" else BackendApiHTTPError
        raise error_cls(
            message,
            status_code=response.status_code,
            request_id=request_id,
            error_code=error_code,
            details=details,
        )

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self.request("GET", path, params=params, headers=headers)

    def post(
        self,
        path: str,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self.request("POST", path, json=json, params=params, headers=headers)

    def put(
        self,
        path: str,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self.request("PUT", path, json=json, params=params, headers=headers)

    def patch(
        self,
        path: str,
        json: dict[str, Any] | list[Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self.request("PATCH", path, json=json, params=params, headers=headers)

    def delete(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self.request("DELETE", path, params=params, headers=headers)

    def download(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        return self.request(
            "GET",
            path,
            params=params,
            headers=headers,
            expect_json=False,
        )

    def upload(
        self,
        path: str,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Upload one file using multipart/form-data without unsafe retries."""
        request_id = (headers or {}).get("X-Request-ID") or str(uuid.uuid4())
        request_headers = {"X-Request-ID": request_id}
        if headers:
            request_headers.update(headers)
        self.last_request_id = request_id
        self.last_response_request_id = None

        try:
            with self._client() as client:
                response = client.post(
                    self._path(path),
                    files={"file": (filename, content, content_type)},
                    headers=request_headers,
                )
            self.last_response_request_id = (
                response.headers.get("X-Request-ID") or self.last_request_id
            )
            return self._handle_response(response, expect_json=True)
        except httpx.TimeoutException as exc:
            raise BackendApiTimeoutError(
                "Backend API request timed out",
                request_id=self.last_response_request_id or self.last_request_id,
            ) from exc
        except httpx.TransportError as exc:
            raise BackendUnavailableError(
                "Backend API is unavailable",
                request_id=self.last_response_request_id or self.last_request_id,
                details={"error": str(exc)},
            ) from exc

    def health(self) -> dict[str, Any]:
        return self.get(self.health_path)

    def readiness(self) -> dict[str, Any]:
        return self.get(self.readiness_path)


def get_api_client() -> ApiClient:
    """Return a configured API client instance."""
    return ApiClient()
