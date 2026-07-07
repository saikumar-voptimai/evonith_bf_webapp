"""Admin-protected operational endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from apps.backend_api.app.api.v1.schemas.common import ApiResponse
from apps.backend_api.app.api.v1.schemas.ops import CleanupRequest
from apps.backend_api.app.core.auth_dependencies import require_admin_user
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.audit_service import AuditService
from apps.backend_api.app.services.error_registry_service import ErrorRegistryService
from apps.backend_api.app.services.runtime_cleanup_service import RuntimeCleanupService

router = APIRouter(prefix="/ops", tags=["ops"])


def _wrap(request: Request, data: Any) -> ApiResponse:
    return ApiResponse(request_id=get_request_id(request), data=data)


def _audit(request: Request) -> AuditService:
    service = getattr(request.app.state, "audit_service", None)
    if service is None:
        service = AuditService(settings=request.app.state.backend_settings)
        request.app.state.audit_service = service
    return service


@router.post("/cleanup/dry-run", response_model=ApiResponse)
def cleanup_dry_run(
    request: Request,
    payload: CleanupRequest | None = None,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    service = RuntimeCleanupService(request.app.state.backend_settings)
    result = service.dry_run((payload or CleanupRequest(dry_run=True)).model_dump(exclude_none=True))
    _audit(request).record_event(
        {
            "request_id": get_request_id(request),
            "actor_user_id": current_user.get("id"),
            "actor_username": current_user.get("username"),
            "event_type": "cleanup.executed",
            "resource_type": "runtime",
            "action": "dry_run",
            "result": "success",
            "status_code": 200,
            "metadata": {"dry_run": True, "would_delete": result.get("would_delete")},
        }
    )
    return _wrap(request, result)


@router.post("/cleanup/run", response_model=ApiResponse)
def cleanup_run(
    request: Request,
    payload: CleanupRequest | None = None,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    service = RuntimeCleanupService(request.app.state.backend_settings)
    options = (payload or CleanupRequest()).model_dump(exclude_none=True)
    options.setdefault("dry_run", False)
    result = service.run(options)
    _audit(request).record_event(
        {
            "request_id": get_request_id(request),
            "actor_user_id": current_user.get("id"),
            "actor_username": current_user.get("username"),
            "event_type": "cleanup.executed",
            "resource_type": "runtime",
            "action": "run",
            "result": "success",
            "status_code": 200,
            "metadata": {"dry_run": result.get("dry_run"), "deleted": result.get("deleted")},
        }
    )
    return _wrap(request, result)


@router.get("/audit/events", response_model=ApiResponse)
def list_audit_events(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    event_type: str | None = None,
    actor_user_id: str | None = None,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    if not request.app.state.backend_settings.audit_admin_read_enabled:
        from apps.backend_api.app.core.errors import ApiError

        raise ApiError("AUDIT_READ_DISABLED", "Audit reads are disabled.", 403)
    return _wrap(
        request,
        _audit(request).list_events(
            limit=limit,
            offset=offset,
            event_type=event_type,
            actor_user_id=actor_user_id,
        ),
    )


@router.post("/audit/retention", response_model=ApiResponse)
def cleanup_audit_retention(
    request: Request,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    return _wrap(request, _audit(request).cleanup_retention())


@router.get("/error-codes", response_model=ApiResponse)
def list_error_codes(
    request: Request,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    return _wrap(request, ErrorRegistryService().list_codes())


@router.get("/error-codes/{code}", response_model=ApiResponse)
def get_error_code(
    request: Request,
    code: str,
    current_user: dict[str, Any] = Depends(require_admin_user),
):
    _ = current_user
    return _wrap(request, ErrorRegistryService().get_code(code))

