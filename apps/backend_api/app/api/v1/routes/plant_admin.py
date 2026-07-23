"""API v1 plant administration routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from apps.backend_api.app.api.v1.schemas.common import ApiMeta
from apps.backend_api.app.api.v1.schemas.plant_admin import (
    BurdenDistributionContextApiResponse,
    BurdenDistributionUpdateRequest,
    BurdenDistributionHistoryApiResponse,
    BurdenHistoryDeleteApiResponse,
    HistoryDeleteRequest,
    HopperHistoryDeleteApiResponse,
    HopperMappingContextApiResponse,
    HopperMappingHistoryApiResponse,
    HopperMappingUpdateRequest,
)
from apps.backend_api.app.core.auth_dependencies import require_permission
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.plant_admin_service import PlantAdminService

router = APIRouter(prefix="/admin", tags=["plant-admin"])


def get_plant_admin_service(request: Request) -> PlantAdminService:
    """Return app plant-admin service or a lazy default."""
    service = getattr(request.app.state, "plant_admin_service", None)
    if service is not None:
        return service
    service = PlantAdminService()
    request.app.state.plant_admin_service = service
    return service


def _client_ip(request: Request) -> str | None:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or None
    return getattr(request.client, "host", None)


def _wrap(response_model, request: Request, data: dict[str, Any]):
    return response_model(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(),
    )


@router.get(
    "/hopper-mappings/context",
    response_model=HopperMappingContextApiResponse,
    operation_id="getHopperMappingContext",
)
def get_hopper_mapping_context(
    request: Request,
    at: datetime | None = Query(None),
    _current_user: dict = Depends(require_permission("hopper:write")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    return _wrap(
        HopperMappingContextApiResponse,
        request,
        plant_admin_service.hopper_context(at=at),
    )


@router.get(
    "/hopper-mappings/history",
    response_model=HopperMappingHistoryApiResponse,
    operation_id="listHopperMappingHistory",
)
def list_hopper_mapping_history(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _current_user: dict = Depends(require_permission("hopper:write")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    return _wrap(
        HopperMappingHistoryApiResponse,
        request,
        plant_admin_service.hopper_history(limit=limit, offset=offset),
    )


@router.put(
    "/hopper-mappings",
    response_model=HopperMappingContextApiResponse,
    operation_id="updateHopperMapping",
)
def update_hopper_mapping(
    request: Request,
    payload: HopperMappingUpdateRequest,
    current_user: dict = Depends(require_permission("hopper:write")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    data = plant_admin_service.update_hopper_mapping(
        effective_at=payload.effective_at,
        expected_snapshot_id=payload.expected_snapshot_id,
        assignments=payload.assignments,
        current_user=current_user,
        ip_address=_client_ip(request),
    )
    return _wrap(HopperMappingContextApiResponse, request, data)


@router.delete(
    "/hopper-mappings/history",
    response_model=HopperHistoryDeleteApiResponse,
    operation_id="deleteHopperMappingHistory",
)
def delete_hopper_mapping_history(
    request: Request,
    payload: HistoryDeleteRequest,
    _current_user: dict = Depends(require_permission("hopper:history:delete")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    return _wrap(
        HopperHistoryDeleteApiResponse,
        request,
        plant_admin_service.delete_hopper_history(record_ids=payload.record_ids),
    )


@router.get(
    "/burden-distribution/context",
    response_model=BurdenDistributionContextApiResponse,
    operation_id="getBurdenDistributionContext",
)
def get_burden_distribution_context(
    request: Request,
    at: datetime | None = Query(None),
    _current_user: dict = Depends(require_permission("burden:write")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    return _wrap(
        BurdenDistributionContextApiResponse,
        request,
        plant_admin_service.burden_context(at=at),
    )


@router.get(
    "/burden-distribution/history",
    response_model=BurdenDistributionHistoryApiResponse,
    operation_id="listBurdenDistributionHistory",
)
def list_burden_distribution_history(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _current_user: dict = Depends(require_permission("burden:write")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    return _wrap(
        BurdenDistributionHistoryApiResponse,
        request,
        plant_admin_service.burden_history(limit=limit, offset=offset),
    )


@router.put(
    "/burden-distribution",
    response_model=BurdenDistributionContextApiResponse,
    operation_id="updateBurdenDistribution",
)
def update_burden_distribution(
    request: Request,
    payload: BurdenDistributionUpdateRequest,
    current_user: dict = Depends(require_permission("burden:write")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    data = plant_admin_service.update_burden_distribution(
        effective_at=payload.effective_at,
        expected_snapshot_id=payload.expected_snapshot_id,
        values=payload.values,
        current_user=current_user,
        ip_address=_client_ip(request),
    )
    return _wrap(BurdenDistributionContextApiResponse, request, data)


@router.delete(
    "/burden-distribution/history",
    response_model=BurdenHistoryDeleteApiResponse,
    operation_id="deleteBurdenDistributionHistory",
)
def delete_burden_distribution_history(
    request: Request,
    payload: HistoryDeleteRequest,
    _current_user: dict = Depends(require_permission("burden:history:delete")),
    plant_admin_service: PlantAdminService = Depends(get_plant_admin_service),
):
    return _wrap(
        BurdenHistoryDeleteApiResponse,
        request,
        plant_admin_service.delete_burden_history(record_ids=payload.record_ids),
    )
