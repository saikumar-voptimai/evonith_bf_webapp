"""Operational status endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from apps.backend_api.app.api.v1.schemas.common import ApiResponse
from apps.backend_api.app.core.auth_dependencies import get_optional_current_user, require_admin_user
from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.dependency_status_service import DependencyStatusService
from apps.backend_api.app.services.runtime_status_service import RuntimeStatusService

router = APIRouter(prefix="/status", tags=["status"])


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def _wrap(request: Request, data: Any) -> ApiResponse:
    return ApiResponse(request_id=get_request_id(request), data=data)


@router.get("", response_model=ApiResponse)
def status_summary(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    settings = _settings(request)
    runtime = RuntimeStatusService(settings).status(create_missing=True, include_sizes=False)
    data: dict[str, Any] = {
        "status": "ok" if runtime["status"] in {"ok", "warning"} else "degraded",
        "health": {
            "service": "evonith-backend-api",
            "api_version": "v1",
            "environment": settings.backend_env,
        },
        "runtime": {
            "status": runtime["status"],
            "checks": runtime["checks"],
            "warnings": runtime["warnings"],
        },
    }
    if current_user and str(current_user.get("role") or "").lower() == "admin":
        data["dependencies"] = DependencyStatusService(settings).status()
    return _wrap(request, data)


@router.get("/runtime/details", response_model=ApiResponse)
def runtime_details(
    request: Request,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    return _wrap(request, RuntimeStatusService(_settings(request)).status())


@router.get("/config", response_model=ApiResponse)
def status_config(
    request: Request,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    return _wrap(request, _settings(request).safe_runtime_profile_summary())


@router.get("/dependencies", response_model=ApiResponse)
def dependency_status(
    request: Request,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    service = getattr(request.app.state, "dependency_status_service", None)
    if service is None:
        service = DependencyStatusService(_settings(request))
        request.app.state.dependency_status_service = service
    return _wrap(request, service.status())
