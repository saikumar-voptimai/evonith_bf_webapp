"""HTTP middleware for the backend API."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from apps.backend_api.app.services.redaction_service import redact_dict

access_log = logging.getLogger("evonith.access")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a request id to request state and response headers."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Record safe access logs, metrics, and best-effort audit events."""

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = int(getattr(response, "status_code", 500))
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            error_code = getattr(request.state, "error_code", None)
            route = self._route_label(request)
            query = dict(request.query_params) if self._include_query(request) else None
            user = getattr(request.state, "current_user", None) or {}
            extra: dict[str, Any] = {
                "event": "http_access",
                "request_id": getattr(request.state, "request_id", None),
                "user_id": user.get("id"),
                "method": request.method,
                "path": route,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 3),
                "error_code": error_code,
            }
            if query:
                extra["query"] = redact_dict(query)
            self._record_metrics(request, method=request.method, route=route, status_code=status_code, duration_ms=duration_ms, error_code=error_code)
            self._record_audit(request, status_code=status_code, duration_ms=duration_ms, error_code=error_code)
            if getattr(request.app.state.backend_settings, "access_log_enabled", True):
                access_log.info("request completed", extra=extra)

    @staticmethod
    def _route_label(request: Request) -> str:
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        return str(path or request.url.path)

    @staticmethod
    def _include_query(request: Request) -> bool:
        settings = getattr(request.app.state, "backend_settings", None)
        return bool(getattr(settings, "access_log_include_query_params", False))

    @staticmethod
    def _record_metrics(
        request: Request,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float,
        error_code: str | None,
    ) -> None:
        service = getattr(request.app.state, "metrics_service", None)
        if service is None:
            return
        try:
            service.record_request(
                method=method,
                route=route,
                status_code=status_code,
                duration_ms=duration_ms,
                error_code=error_code,
            )
        except Exception:
            access_log.debug("metrics record failed", exc_info=True)

    @staticmethod
    def _record_audit(
        request: Request,
        *,
        status_code: int,
        duration_ms: float,
        error_code: str | None,
    ) -> None:
        service = getattr(request.app.state, "audit_service", None)
        if service is None:
            return
        try:
            service.record_request(
                request,
                status_code=status_code,
                duration_ms=duration_ms,
                error_code=error_code,
            )
        except Exception:
            access_log.debug("audit record failed", exc_info=True)
