"""Frontend backend health/readiness status helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    from config.frontend_settings import load_frontend_settings
    from services.api_client import ApiClient, unwrap_api_response
    from services.api_errors import FrontendApiError
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.config.frontend_settings import load_frontend_settings
    from src.services.api_client import ApiClient, unwrap_api_response
    from src.services.api_errors import FrontendApiError


@dataclass(frozen=True)
class BackendStatus:
    is_available: bool
    is_ready: bool | None
    status: str
    message: str
    request_id: str | None = None
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _status_client() -> ApiClient:
    settings = load_frontend_settings()
    return ApiClient(
        timeout=min(settings.backend_api_timeout_seconds, 2.0),
        connect_timeout=min(settings.backend_api_connect_timeout_seconds, 0.5),
        max_retries=0,
    )


def check_backend_health(client: ApiClient | None = None) -> BackendStatus:
    """Check backend health without raising UI-breaking exceptions."""
    api = client or _status_client()
    started = time.perf_counter()
    try:
        payload = api.health()
        data = unwrap_api_response(payload)
        latency_ms = (time.perf_counter() - started) * 1000
        status = str(data.get("status", "unknown")) if isinstance(data, dict) else "unknown"
        return BackendStatus(
            is_available=status == "ok",
            is_ready=None,
            status=status,
            message="Backend API available" if status == "ok" else "Backend API status unknown",
            request_id=api.last_response_request_id,
            latency_ms=latency_ms,
            details=data if isinstance(data, dict) else {"payload": data},
        )
    except FrontendApiError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return BackendStatus(
            is_available=False,
            is_ready=None,
            status="unavailable",
            message=exc.message,
            request_id=exc.request_id,
            latency_ms=latency_ms,
            details=exc.details,
        )


def check_backend_readiness(client: ApiClient | None = None) -> BackendStatus:
    """Check backend readiness without raising UI-breaking exceptions."""
    api = client or _status_client()
    started = time.perf_counter()
    try:
        payload = api.readiness()
        data = unwrap_api_response(payload)
        latency_ms = (time.perf_counter() - started) * 1000
        status = str(data.get("status", "unknown")) if isinstance(data, dict) else "unknown"
        return BackendStatus(
            is_available=True,
            is_ready=status == "ready",
            status=status,
            message="Backend API ready" if status == "ready" else "Backend API not ready",
            request_id=api.last_response_request_id,
            latency_ms=latency_ms,
            details=data if isinstance(data, dict) else {"payload": data},
        )
    except FrontendApiError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return BackendStatus(
            is_available=False,
            is_ready=False,
            status="unavailable",
            message=exc.message,
            request_id=exc.request_id,
            latency_ms=latency_ms,
            details=exc.details,
        )


def get_backend_status_summary(client: ApiClient | None = None) -> BackendStatus:
    """Return a combined health/readiness summary for the UI badge."""
    api = client or _status_client()
    health = check_backend_health(api)
    if not health.is_available:
        return health

    readiness = check_backend_readiness(api)
    if readiness.is_ready:
        return readiness
    if readiness.is_available:
        return readiness
    return BackendStatus(
        is_available=True,
        is_ready=False,
        status="not_ready",
        message="Backend API available but readiness check failed",
        request_id=readiness.request_id or health.request_id,
        latency_ms=(health.latency_ms or 0) + (readiness.latency_ms or 0),
        details={"health": health.details, "readiness": readiness.details},
    )
