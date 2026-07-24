"""API v1 Blend Optimizer routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request, status
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.blend_optimizer import (
    BlendManualEvaluationRequest,
    BlendOptimizerContextCreateRequest,
    BlendOptimizerPreferencesPatchRequest,
    BlendOptimizerRequest,
    BlendOptimizerRunCreateRequest,
)
from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.core.auth_dependencies import get_optional_current_user
from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.blend_optimizer_api_service import BlendOptimizerApiService
from apps.backend_api.app.services.blend_optimizer_service import BlendOptimizerService
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.compute_job_service import compute_job_service

router = APIRouter(prefix="/blend-optimizer", tags=["blend-optimizer"])


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def _user_id(current_user: dict[str, Any] | None) -> str:
    if not current_user:
        return "anonymous"
    return str(current_user.get("id") or current_user.get("username") or "anonymous")


def _has_permission(current_user: dict[str, Any] | None, permission: str) -> bool:
    return permission in {str(item) for item in (current_user or {}).get("permissions") or []}


def _permission(request: Request, current_user: dict[str, Any] | None, permission: str) -> dict[str, Any] | None:
    if not _settings(request).compute_require_auth:
        return current_user
    if current_user is None:
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)
    if not _has_permission(current_user, permission):
        raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)
    return current_user


def _require_compute_user(settings: BackendSettings, user: dict[str, Any] | None) -> None:
    if settings.compute_require_auth and not user:
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)


def get_blend_optimizer_service(request: Request) -> BlendOptimizerService:
    service = getattr(request.app.state, "blend_optimizer_service", None)
    if service is None:
        service = BlendOptimizerService(settings=_settings(request))
        request.app.state.blend_optimizer_service = service
    return service


def get_blend_optimizer_api_service(request: Request) -> BlendOptimizerApiService:
    service = getattr(request.app.state, "blend_optimizer_api_service", None)
    if service is None:
        service = BlendOptimizerApiService(settings=_settings(request))
        service.ensure_storage()
        request.app.state.blend_optimizer_api_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> ApiResponse:
    return ApiResponse(request_id=get_request_id(request), data=data, meta=ApiMeta(warnings=warnings or []))


@router.get("/catalog", response_model=ApiResponse, operation_id="get_blend_optimizer_catalog")
def get_catalog(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    _permission(request, current_user, "blend_optimizer:read")
    return _wrap(request, service.catalog())


@router.post("/contexts", response_model=ApiResponse, operation_id="create_blend_optimizer_context")
def create_context(
    request: Request,
    payload: BlendOptimizerContextCreateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:read")
    return _wrap(request, service.create_context(payload.model_dump(), user, idempotency_key=idempotency_key))


@router.get("/contexts/{context_id}", response_model=ApiResponse, operation_id="get_blend_optimizer_context_by_id")
def get_context_by_id(
    request: Request,
    context_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:read")
    return _wrap(request, service.get_context(context_id, user))


@router.get("/contexts/{context_id}/diagnostics", response_model=ApiResponse, operation_id="get_blend_optimizer_context_diagnostics")
def get_context_diagnostics(
    request: Request,
    context_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:diagnostics")
    return _wrap(request, service.context_diagnostics(context_id, user))


@router.get("/preferences", response_model=ApiResponse, operation_id="get_blend_optimizer_preferences")
def get_preferences(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:read")
    return _wrap(request, service.get_preferences(user))


@router.patch("/preferences", response_model=ApiResponse, operation_id="update_blend_optimizer_preferences")
def update_preferences(
    request: Request,
    payload: BlendOptimizerPreferencesPatchRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:preferences:write")
    return _wrap(request, service.update_preferences(payload.model_dump(), user))


@router.post("/runs", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, operation_id="create_blend_optimizer_run")
def create_run(
    request: Request,
    payload: BlendOptimizerRunCreateRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:run")
    run = service.create_run(payload.model_dump(), user, idempotency_key=idempotency_key)
    background_tasks.add_task(service.process_run, run["run_id"], user)
    return _wrap(request, run)


@router.get("/runs/{run_id}", response_model=ApiResponse, operation_id="get_blend_optimizer_run")
def get_run(
    request: Request,
    run_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:read")
    return _wrap(request, service.get_run(run_id, user))


@router.get("/runs/{run_id}/events", response_model=ApiResponse, operation_id="get_blend_optimizer_run_events")
def get_run_events(
    request: Request,
    run_id: str,
    after: int | None = Query(default=None, ge=0),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:read")
    return _wrap(request, service.run_events(run_id, user, after=after))


@router.post("/runs/{run_id}/cancel", response_model=ApiResponse, operation_id="cancel_blend_optimizer_run")
def cancel_run(
    request: Request,
    run_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:run")
    return _wrap(request, service.cancel_run(run_id, user))


@router.post("/runs/{run_id}/manual-evaluations", response_model=ApiResponse, operation_id="create_blend_optimizer_manual_evaluation")
def manual_evaluation(
    request: Request,
    run_id: str,
    payload: BlendManualEvaluationRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerApiService = Depends(get_blend_optimizer_api_service),
):
    user = _permission(request, current_user, "blend_optimizer:run")
    return _wrap(request, service.evaluate_manual_blend(run_id, payload.model_dump(), user, idempotency_key=idempotency_key))


@router.get("/artifacts/{artifact_id}/download", operation_id="download_blend_optimizer_artifact")
def download_artifact(
    request: Request,
    artifact_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    user = _permission(request, current_user, "blend_optimizer:export")
    service = ComputeArtifactService(_settings(request))
    try:
        metadata = service.get_metadata(artifact_id)
        path = service.get_path(artifact_id)
    except ValueError as exc:
        raise ApiError("BMO_ARTIFACT_NOT_FOUND", "Invalid artifact id.", status_code=400) from exc
    except FileNotFoundError as exc:
        raise ApiError("BMO_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404) from exc
    if metadata.workflow != "blend_optimizer":
        raise ApiError("BMO_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404)
    if service.is_expired(metadata):
        raise ApiError("BMO_ARTIFACT_EXPIRED", "Artifact has expired.", status_code=410)
    if metadata.owner_user_id and metadata.owner_user_id != _user_id(user) and not _has_permission(user, "blend_optimizer:artifacts:read:any"):
        raise ApiError("FORBIDDEN", "Artifact access denied.", status_code=403)
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename, headers={"X-Request-ID": get_request_id(request)})


@router.get("/context", response_model=ApiResponse, deprecated=True, operation_id="get_legacy_blend_optimizer_context")
def context(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.context())


@router.get("/models", response_model=ApiResponse, deprecated=True, operation_id="list_legacy_blend_optimizer_models")
def models(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.list_models())


@router.post("/predict", response_model=ApiResponse, deprecated=True, operation_id="legacy_blend_optimizer_predict")
def predict(
    request: Request,
    payload: dict[str, Any],
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.predict(payload))


@router.post("/optimize", response_model=ApiResponse, deprecated=True, operation_id="legacy_blend_optimizer_optimize")
def optimize(
    request: Request,
    payload: BlendOptimizerRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.optimize(payload.model_dump(), route_prefix="/blend-optimizer"))


@router.post("/jobs", response_model=ApiResponse, deprecated=True, operation_id="legacy_start_blend_optimizer_job")
def start_job(
    request: Request,
    payload: BlendOptimizerRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    job = compute_job_service.run_inline(
        workflow="blend_optimizer",
        fn=lambda: service.optimize(payload.model_dump(), route_prefix="/blend-optimizer"),
        message="Deprecated Blend Optimizer job queued",
    )
    return _wrap(request, compute_job_service.response(job))


@router.get("/jobs/{job_id}", response_model=ApiResponse, deprecated=True, operation_id="legacy_get_blend_optimizer_job")
def get_job(
    request: Request,
    job_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    _require_compute_user(_settings(request), current_user)
    job = compute_job_service.get_job(job_id)
    if job is None or job.workflow != "blend_optimizer":
        raise ApiError("COMPUTE_JOB_NOT_FOUND", "Compute job not found.", status_code=404)
    return _wrap(request, compute_job_service.status(job))
