"""Unified job visibility endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from app.api.v1.schemas.common import ApiResponse
from app.core.auth_dependencies import require_admin_user
from app.core.errors import ApiError
from app.core.responses import get_request_id
from app.services.unified_job_service import UnifiedJobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _wrap(request: Request, data: Any) -> ApiResponse:
    return ApiResponse(request_id=get_request_id(request), data=data)


def _service(request: Request) -> UnifiedJobService:
    service = getattr(request.app.state, "unified_job_service", None)
    if service is None:
        service = UnifiedJobService()
        request.app.state.unified_job_service = service
    return service


@router.get("", response_model=ApiResponse)
def list_jobs(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    if not request.app.state.backend_settings.unified_jobs_enabled:
        raise ApiError("JOBS_DISABLED", "Unified jobs are disabled.", 404)
    return _wrap(request, _service(request).list_jobs(limit=limit, offset=offset))


@router.get("/{job_id}", response_model=ApiResponse)
def get_job(
    request: Request,
    job_id: str,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    if not request.app.state.backend_settings.unified_jobs_enabled:
        raise ApiError("JOBS_DISABLED", "Unified jobs are disabled.", 404)
    return _wrap(request, _service(request).get_job(job_id))

