"""API v1 Blend Optimizer routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from app.api.v1.schemas.blend_optimizer import BlendOptimizerRequest, BlendPredictionRequest
from app.api.v1.schemas.common import ApiMeta, ApiResponse
from app.core.auth_dependencies import get_optional_current_user
from app.core.config import BackendSettings
from app.core.errors import ApiError
from app.core.responses import get_request_id
from app.services.blend_optimizer_service import BlendOptimizerService
from app.services.compute_artifact_service import ComputeArtifactService
from app.services.compute_job_service import compute_job_service

router = APIRouter(prefix="/blend-optimizer", tags=["blend-optimizer"])


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def _require_compute_user(settings: BackendSettings, user: dict[str, Any] | None) -> None:
    if settings.compute_require_auth and not user:
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)


def get_blend_optimizer_service(request: Request) -> BlendOptimizerService:
    service = getattr(request.app.state, "blend_optimizer_service", None)
    if service is None:
        service = BlendOptimizerService(settings=_settings(request))
        request.app.state.blend_optimizer_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> ApiResponse:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(warnings=warnings or []),
    )


@router.get("/context", response_model=ApiResponse)
def context(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.context())


@router.get("/models", response_model=ApiResponse)
def models(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.list_models())


@router.post("/predict", response_model=ApiResponse)
def predict(
    request: Request,
    payload: BlendPredictionRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.predict(payload.model_dump()))


@router.post("/optimize", response_model=ApiResponse)
def optimize(
    request: Request,
    payload: BlendOptimizerRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: BlendOptimizerService = Depends(get_blend_optimizer_service),
):
    _require_compute_user(_settings(request), current_user)
    return _wrap(request, service.optimize(payload.model_dump(), route_prefix="/blend-optimizer"))


@router.post("/jobs", response_model=ApiResponse)
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
        message="Blend Optimizer job queued",
    )
    return _wrap(request, compute_job_service.response(job))


@router.get("/jobs/{job_id}", response_model=ApiResponse)
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


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(
    request: Request,
    artifact_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    _require_compute_user(_settings(request), current_user)
    service = ComputeArtifactService(_settings(request))
    try:
        metadata = service.get_metadata(artifact_id)
        path = service.get_path(artifact_id)
    except ValueError as exc:
        raise ApiError("COMPUTE_ARTIFACT_NOT_FOUND", "Invalid artifact id.", status_code=400) from exc
    except FileNotFoundError as exc:
        raise ApiError("COMPUTE_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404) from exc
    if metadata.workflow != "blend_optimizer":
        raise ApiError("COMPUTE_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404)
    return FileResponse(
        path=path,
        media_type=metadata.content_type,
        filename=metadata.filename,
        headers={"X-Request-ID": get_request_id(request)},
    )
