"""Operational metrics endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.v1.schemas.common import ApiResponse
from app.core.auth_dependencies import get_optional_current_user, require_admin_user
from app.core.config import BackendSettings
from app.core.errors import ApiError
from app.core.responses import get_request_id
from app.services.metrics_service import MetricsService

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def _metrics(request: Request) -> MetricsService:
    service = getattr(request.app.state, "metrics_service", None)
    if service is None:
        service = MetricsService()
        request.app.state.metrics_service = service
    return service


def _wrap(request: Request, data: Any) -> ApiResponse:
    return ApiResponse(request_id=get_request_id(request), data=data)


def _require_admin(user: dict[str, Any] | None) -> None:
    if not user:
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", 401)
    if str(user.get("role") or "").lower() != "admin":
        raise ApiError("FORBIDDEN", "Admin role is required.", 403)


@router.get("", response_model=ApiResponse)
def metrics(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    settings = _settings(request)
    if not settings.metrics_enabled:
        raise ApiError("METRICS_DISABLED", "Metrics are disabled.", 404)
    if settings.metrics_require_auth:
        _require_admin(current_user)
    return _wrap(request, _metrics(request).snapshot())


@router.post("/reset", response_model=ApiResponse)
def reset_metrics(
    request: Request,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    settings = _settings(request)
    if not settings.metrics_enabled:
        raise ApiError("METRICS_DISABLED", "Metrics are disabled.", 404)
    if not settings.metrics_reset_enabled:
        raise ApiError("METRICS_RESET_DISABLED", "Metrics reset is disabled.", 403)
    _metrics(request).reset()
    return _wrap(request, {"reset": True})
