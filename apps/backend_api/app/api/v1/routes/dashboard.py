"""API v1 dashboard routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from apps.backend_api.app.api.v1.schemas.common import ApiMeta
from apps.backend_api.app.api.v1.schemas.dashboard import (
    DashboardBucket,
    DashboardKpisApiResponse,
    DashboardWindow,
)
from apps.backend_api.app.core.auth_dependencies import require_authenticated_user
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def get_dashboard_service(request: Request) -> DashboardService:
    """Return app dashboard service or a lazy default."""
    service = getattr(request.app.state, "dashboard_service", None)
    if service is not None:
        return service
    service = DashboardService()
    request.app.state.dashboard_service = service
    return service


def _wrap(
    request: Request,
    data: dict[str, Any],
    warnings: list[str] | None = None,
) -> DashboardKpisApiResponse:
    return DashboardKpisApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(warnings=warnings or []),
    )


@router.get(
    "/kpis",
    response_model=DashboardKpisApiResponse,
    operation_id="getDashboardKpis",
)
def get_dashboard_kpis(
    request: Request,
    window: DashboardWindow = Query("1h"),
    bucket: DashboardBucket = Query("15m"),
    _current_user: dict = Depends(require_authenticated_user),
    dashboard_service: DashboardService = Depends(get_dashboard_service),
):
    data, warnings = dashboard_service.get_kpis(window=window, bucket=bucket)
    return _wrap(request, data, warnings)
