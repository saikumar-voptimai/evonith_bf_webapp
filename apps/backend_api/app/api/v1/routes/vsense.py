"""API v1 V-Sense advisory optimization routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Request, status

from apps.backend_api.app.api.v1.schemas.common import ApiErrorResponse, ApiMeta
from apps.backend_api.app.api.v1.schemas.vsense import (
    VSenseCatalogResponse,
    VSenseContextCreateRequest,
    VSenseContextResponse,
    VSenseControlProfileResponse,
    VSenseControlProfileUpdateRequest,
    VSenseRunAcceptedResponse,
    VSenseRunCreateRequest,
    VSenseRunEventsResponse,
    VSenseRunStatusResponse,
)
from apps.backend_api.app.core.auth_dependencies import require_permission
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.repositories.vsense_repository import VSenseRepository
from apps.backend_api.app.services.vsense_run_service import VSenseRunService
from apps.backend_api.app.services.vsense_service import VSenseService


_API_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ApiErrorResponse, "description": "The V-Sense request is invalid."},
    401: {"model": ApiErrorResponse, "description": "Authentication is required."},
    403: {"model": ApiErrorResponse, "description": "The caller is not permitted."},
    404: {"model": ApiErrorResponse, "description": "The V-Sense resource was not found."},
    409: {"model": ApiErrorResponse, "description": "The V-Sense request conflicts with current state."},
    410: {"model": ApiErrorResponse, "description": "The V-Sense context or run has expired."},
    422: {"model": ApiErrorResponse, "description": "The request payload is invalid."},
    500: {"model": ApiErrorResponse, "description": "The V-Sense result could not be processed."},
    503: {"model": ApiErrorResponse, "description": "A required V-Sense source is unavailable."},
}


router = APIRouter(prefix="/vsense", tags=["vsense"], responses=_API_ERROR_RESPONSES)


def get_vsense_service(request: Request) -> VSenseService:
    service = getattr(request.app.state, "vsense_service", None)
    if service is not None:
        return service
    repository = getattr(request.app.state, "vsense_repository", None) or VSenseRepository()
    service = VSenseService(
        settings=getattr(request.app.state, "backend_settings", None),
        repository=repository,
        audit_service=getattr(request.app.state, "audit_service", None),
    )
    request.app.state.vsense_repository = repository
    request.app.state.vsense_service = service
    return service


def get_vsense_run_service(request: Request) -> VSenseRunService:
    service = getattr(request.app.state, "vsense_run_service", None)
    if service is not None:
        return service
    vsense_service = get_vsense_service(request)
    service = VSenseRunService(
        repository=vsense_service.repository,
        context_service=vsense_service.context_service,
        settings=getattr(request.app.state, "backend_settings", None),
        audit_service=getattr(request.app.state, "audit_service", None),
    )
    request.app.state.vsense_run_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "request_id": get_request_id(request),
        "data": data,
        "meta": ApiMeta(warnings=warnings or []),
    }


@router.get(
    "/catalog",
    response_model=VSenseCatalogResponse,
    operation_id="get_vsense_catalog",
)
def get_vsense_catalog(
    request: Request,
    _: dict[str, Any] = Depends(require_permission("vsense:read")),
    service: VSenseService = Depends(get_vsense_service),
) -> dict[str, Any]:
    return _wrap(request, service.get_catalog())


@router.post(
    "/contexts",
    response_model=VSenseContextResponse,
    operation_id="create_vsense_context",
)
def create_vsense_context(
    request: Request,
    payload: VSenseContextCreateRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255),
    current_user: dict[str, Any] = Depends(require_permission("vsense:read")),
    service: VSenseService = Depends(get_vsense_service),
) -> dict[str, Any]:
    data = service.create_context(
        payload.model_dump(mode="json"),
        current_user=current_user,
        idempotency_key=idempotency_key,
    )
    return _wrap(request, data, warnings=data.get("warnings", []))


@router.get(
    "/control-profiles/{optimization_type_id}",
    response_model=VSenseControlProfileResponse,
    operation_id="get_vsense_control_profile",
)
def get_vsense_control_profile(
    request: Request,
    optimization_type_id: str,
    _: dict[str, Any] = Depends(require_permission("vsense:read")),
    service: VSenseService = Depends(get_vsense_service),
) -> dict[str, Any]:
    return _wrap(request, service.get_control_profile(optimization_type_id))


@router.put(
    "/control-profiles/{optimization_type_id}",
    response_model=VSenseControlProfileResponse,
    operation_id="update_vsense_control_profile",
)
def update_vsense_control_profile(
    request: Request,
    optimization_type_id: str,
    payload: VSenseControlProfileUpdateRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255),
    current_user: dict[str, Any] = Depends(require_permission("vsense:bounds:write")),
    service: VSenseService = Depends(get_vsense_service),
) -> dict[str, Any]:
    return _wrap(
        request,
        service.update_control_profile(
            optimization_type_id,
            payload.model_dump(mode="json"),
            current_user=current_user,
            idempotency_key=idempotency_key,
            request_id=get_request_id(request),
        ),
    )


@router.post(
    "/runs",
    response_model=VSenseRunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="create_vsense_run",
)
def create_vsense_run(
    request: Request,
    payload: VSenseRunCreateRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255),
    current_user: dict[str, Any] = Depends(require_permission("vsense:run")),
    service: VSenseRunService = Depends(get_vsense_run_service),
) -> dict[str, Any]:
    return _wrap(
        request,
        service.create_run(
            payload.model_dump(mode="json"),
            current_user=current_user,
            idempotency_key=idempotency_key,
            request_id=get_request_id(request),
        ),
    )


@router.get(
    "/runs/{run_id}",
    response_model=VSenseRunStatusResponse,
    operation_id="get_vsense_run",
)
def get_vsense_run(
    request: Request,
    run_id: str,
    current_user: dict[str, Any] = Depends(require_permission("vsense:read")),
    service: VSenseRunService = Depends(get_vsense_run_service),
) -> dict[str, Any]:
    return _wrap(request, service.get_run(run_id, current_user=current_user))


@router.get(
    "/runs/{run_id}/events",
    response_model=VSenseRunEventsResponse,
    operation_id="get_vsense_run_events",
)
def get_vsense_run_events(
    request: Request,
    run_id: str,
    after: int = 0,
    current_user: dict[str, Any] = Depends(require_permission("vsense:read")),
    service: VSenseRunService = Depends(get_vsense_run_service),
) -> dict[str, Any]:
    return _wrap(
        request,
        service.get_events(run_id, current_user=current_user, after=after),
    )


@router.post(
    "/runs/{run_id}/cancel",
    response_model=VSenseRunStatusResponse,
    operation_id="cancel_vsense_run",
)
def cancel_vsense_run(
    request: Request,
    run_id: str,
    current_user: dict[str, Any] = Depends(require_permission("vsense:run")),
    service: VSenseRunService = Depends(get_vsense_run_service),
) -> dict[str, Any]:
    return _wrap(
        request,
        service.cancel_run(
            run_id,
            current_user=current_user,
            request_id=get_request_id(request),
        ),
    )
