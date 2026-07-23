"""API v1 V-Board routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from apps.backend_api.app.api.v1.schemas.common import ApiErrorResponse, ApiMeta
from apps.backend_api.app.api.v1.schemas.vboard import (
    VBoardCatalogResponse,
    VBoardContoursRequest,
    VBoardContoursResponse,
    VBoardHeatloadTimeseriesRequest,
    VBoardHeatloadTimeseriesResponse,
)
from apps.backend_api.app.core.auth_dependencies import require_permission
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.vboard_service import VBoardService


_API_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ApiErrorResponse, "description": "The request could not be processed."},
    401: {"model": ApiErrorResponse, "description": "Authentication is required."},
    403: {"model": ApiErrorResponse, "description": "The caller is not permitted."},
    413: {"model": ApiErrorResponse, "description": "The V-Board request exceeds configured limits."},
    422: {"model": ApiErrorResponse, "description": "The request payload is invalid."},
    500: {"model": ApiErrorResponse, "description": "The V-Board result could not be processed."},
    503: {"model": ApiErrorResponse, "description": "A required V-Board source is unavailable."},
}


router = APIRouter(prefix="/vboard", tags=["vboard"], responses=_API_ERROR_RESPONSES)


def get_vboard_service(request: Request) -> VBoardService:
    """Return the app V-Board service or a lazy default."""

    service = getattr(request.app.state, "vboard_service", None)
    if service is not None:
        return service
    service = VBoardService(settings=getattr(request.app.state, "backend_settings", None))
    request.app.state.vboard_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None):
    return {
        "request_id": get_request_id(request),
        "data": data,
        "meta": ApiMeta(warnings=warnings or []),
    }


@router.get(
    "/catalog",
    response_model=VBoardCatalogResponse,
    operation_id="get_vboard_catalog",
)
def get_vboard_catalog(
    request: Request,
    _: dict[str, Any] = Depends(require_permission("vboard:read")),
    service: VBoardService = Depends(get_vboard_service),
):
    return _wrap(request, service.get_catalog())


@router.post(
    "/contours",
    response_model=VBoardContoursResponse,
    operation_id="query_vboard_contours",
)
def query_vboard_contours(
    request: Request,
    payload: VBoardContoursRequest,
    _: dict[str, Any] = Depends(require_permission("vboard:read")),
    service: VBoardService = Depends(get_vboard_service),
):
    data = service.query_contours(payload)
    warnings = [
        *data["temperature"].get("warnings", []),
        *data["heatload"].get("warnings", []),
    ]
    return _wrap(request, data, warnings=list(dict.fromkeys(warnings)))


@router.post(
    "/heatload-timeseries",
    response_model=VBoardHeatloadTimeseriesResponse,
    operation_id="query_vboard_heatload_timeseries",
)
def query_vboard_heatload_timeseries(
    request: Request,
    payload: VBoardHeatloadTimeseriesRequest,
    _: dict[str, Any] = Depends(require_permission("vboard:read")),
    service: VBoardService = Depends(get_vboard_service),
):
    data = service.query_heatload_timeseries(payload)
    return _wrap(request, data, warnings=data.get("warnings", []))
