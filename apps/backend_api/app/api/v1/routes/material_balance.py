"""API v1 Material Balance routes."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiErrorResponse, ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.material_balance import (
    MaterialBalanceAshAnalysesResponse,
    MaterialBalanceAshAnalysesUpdateRequest,
    MaterialBalanceCacheRefreshResponse,
    MaterialBalanceConfigResponse,
    MaterialBalanceDprMappingResponse,
    MaterialBalanceDprMappingUpdateRequest,
    MaterialBalanceRefreshCacheRequest,
    MaterialBalanceRunRequest,
    MaterialBalanceRunResponse,
    MaterialBalanceValidateResponse,
)
from apps.backend_api.app.core.auth_dependencies import get_optional_current_user
from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.compute_job_service import compute_job_service
from apps.backend_api.app.services.material_balance_service import MaterialBalanceService

_API_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ApiErrorResponse},
    401: {"model": ApiErrorResponse},
    403: {"model": ApiErrorResponse},
    404: {"model": ApiErrorResponse},
    409: {"model": ApiErrorResponse},
    410: {"model": ApiErrorResponse},
    422: {"model": ApiErrorResponse},
    500: {"model": ApiErrorResponse},
    503: {"model": ApiErrorResponse},
}

router = APIRouter(prefix="/material-balance", tags=["material-balance"], responses=_API_ERROR_RESPONSES)


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def _permission(permission: str) -> Callable[..., dict[str, Any] | None]:
    def dependency(
        request: Request,
        current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    ) -> dict[str, Any] | None:
        if not _settings(request).compute_require_auth:
            if current_user is not None:
                request.state.current_user = current_user
            return current_user
        if current_user is None:
            raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)
        permissions = {str(item) for item in current_user.get("permissions") or []}
        if permission not in permissions:
            raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)
        request.state.current_user = current_user
        return current_user

    return dependency


def get_material_balance_service(request: Request) -> MaterialBalanceService:
    service = getattr(request.app.state, "material_balance_service", None)
    if service is None:
        service = MaterialBalanceService(
            settings=_settings(request),
            audit_service=getattr(request.app.state, "audit_service", None),
        )
        request.app.state.material_balance_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> dict[str, Any]:
    return {"request_id": get_request_id(request), "data": data, "meta": ApiMeta(warnings=warnings or [])}


@router.get("/config", response_model=MaterialBalanceConfigResponse, operation_id="get_material_balance_config")
def get_config(
    request: Request,
    _: dict[str, Any] | None = Depends(_permission("material_balance:read")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    data = service.config()
    return _wrap(request, data)


@router.post("/run", response_model=MaterialBalanceRunResponse, operation_id="run_material_balance")
def run_material_balance(
    request: Request,
    payload: MaterialBalanceRunRequest,
    current_user: dict[str, Any] | None = Depends(_permission("material_balance:run")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    data = service.run(
        payload.model_dump(mode="json"),
        route_prefix="/material-balance",
        current_user=current_user,
        request_id=get_request_id(request),
    )
    return _wrap(request, data)


@router.get("/ash-analyses", response_model=MaterialBalanceAshAnalysesResponse, operation_id="get_material_balance_ash_analyses")
def get_ash_analyses(
    request: Request,
    _: dict[str, Any] | None = Depends(_permission("material_balance:read")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    return _wrap(request, service.get_ash_analyses())


@router.put("/ash-analyses", response_model=MaterialBalanceAshAnalysesResponse, operation_id="update_material_balance_ash_analyses")
def update_ash_analyses(
    request: Request,
    payload: MaterialBalanceAshAnalysesUpdateRequest,
    current_user: dict[str, Any] | None = Depends(_permission("material_balance:config:write")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    return _wrap(
        request,
        service.update_ash_analyses(
            payload.model_dump(mode="json"),
            current_user=current_user,
            request_id=get_request_id(request),
            client_metadata={"user_agent": request.headers.get("user-agent")},
        ),
    )


@router.get("/dpr-mapping", response_model=MaterialBalanceDprMappingResponse, operation_id="get_material_balance_dpr_mapping")
def get_dpr_mapping(
    request: Request,
    sample_day: str | None = Query(default=None),
    _: dict[str, Any] | None = Depends(_permission("material_balance:read")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    from datetime import date

    parsed_day = date.fromisoformat(sample_day) if sample_day else None
    return _wrap(request, service.get_dpr_mapping(sample_day=parsed_day))


@router.put("/dpr-mapping", response_model=MaterialBalanceDprMappingResponse, operation_id="update_material_balance_dpr_mapping")
def update_dpr_mapping(
    request: Request,
    payload: MaterialBalanceDprMappingUpdateRequest,
    current_user: dict[str, Any] | None = Depends(_permission("material_balance:config:write")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    return _wrap(
        request,
        service.update_dpr_mapping(
            payload.model_dump(mode="json"),
            current_user=current_user,
            request_id=get_request_id(request),
            client_metadata={"user_agent": request.headers.get("user-agent")},
        ),
    )


@router.post("/cache/refresh", response_model=MaterialBalanceCacheRefreshResponse, operation_id="refresh_material_balance_cache")
def refresh_cache(
    request: Request,
    payload: MaterialBalanceRefreshCacheRequest,
    _: dict[str, Any] | None = Depends(_permission("material_balance:run")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    return _wrap(request, service.refresh_cache(payload.model_dump(mode="json")))


@router.post("/validate", response_model=MaterialBalanceValidateResponse, deprecated=True, operation_id="validate_material_balance_deprecated")
def validate(
    request: Request,
    payload: MaterialBalanceRunRequest,
    _: dict[str, Any] | None = Depends(_permission("material_balance:run")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    return _wrap(request, service.validate(payload.model_dump(mode="json")))


@router.post("/jobs", response_model=ApiResponse, deprecated=True, operation_id="start_material_balance_inline_job_deprecated")
def start_job(
    request: Request,
    payload: MaterialBalanceRunRequest,
    current_user: dict[str, Any] | None = Depends(_permission("material_balance:run")),
    service: MaterialBalanceService = Depends(get_material_balance_service),
) -> dict[str, Any]:
    job = compute_job_service.run_inline(
        workflow="material_balance",
        fn=lambda: service.run(payload.model_dump(mode="json"), route_prefix="/material-balance", current_user=current_user, request_id=get_request_id(request)),
        message="Material Balance inline compatibility job completed",
    )
    return _wrap(request, compute_job_service.response(job))


@router.get("/jobs/{job_id}", response_model=ApiResponse, deprecated=True, operation_id="get_material_balance_inline_job_deprecated")
def get_job(
    request: Request,
    job_id: str,
    _: dict[str, Any] | None = Depends(_permission("material_balance:read")),
) -> dict[str, Any]:
    job = compute_job_service.get_job(job_id)
    if job is None or job.workflow != "material_balance":
        raise ApiError("COMPUTE_JOB_NOT_FOUND", "Compute job not found.", status_code=404)
    return _wrap(request, compute_job_service.status(job))


@router.get("/artifacts/{artifact_id}/download", operation_id="download_material_balance_artifact")
def download_artifact(
    request: Request,
    artifact_id: str,
    current_user: dict[str, Any] | None = Depends(_permission("material_balance:export")),
):
    service = ComputeArtifactService(_settings(request))
    try:
        metadata = service.get_metadata(artifact_id)
        path = service.get_path(artifact_id)
    except ValueError as exc:
        raise ApiError("MATERIAL_BALANCE_ARTIFACT_NOT_FOUND", "Invalid artifact id.", status_code=400) from exc
    except FileNotFoundError as exc:
        raise ApiError("MATERIAL_BALANCE_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404) from exc
    if metadata.workflow != "material_balance":
        raise ApiError("MATERIAL_BALANCE_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404)
    if service.is_expired(metadata):
        raise ApiError("MATERIAL_BALANCE_ARTIFACT_EXPIRED", "Artifact has expired.", status_code=410)
    owner = metadata.owner_user_id
    permissions = {str(item) for item in ((current_user or {}).get("permissions") or [])}
    if _settings(request).compute_require_auth and owner and owner != str((current_user or {}).get("id") or "") and "material_balance:artifacts:read:any" not in permissions:
        raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename, headers={"X-Request-ID": get_request_id(request)})