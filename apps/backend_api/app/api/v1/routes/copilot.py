"""API v1 AI Copilot routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.copilot import (
    CopilotAnalyzeRequest,
    CopilotAnomalyRequest,
    CopilotDataWindowRequest,
)
from apps.backend_api.app.core.auth_dependencies import get_optional_current_user
from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.compute_job_service import compute_job_service
from apps.backend_api.app.services.copilot_service import CopilotService

router = APIRouter(prefix="/copilot", tags=["copilot"])


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def _require_copilot_user(settings: BackendSettings, user: dict[str, Any] | None) -> None:
    if settings.copilot_require_auth and not user:
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)


def get_copilot_service(request: Request) -> CopilotService:
    service = getattr(request.app.state, "copilot_service", None)
    if service is None:
        service = CopilotService(settings=_settings(request))
        request.app.state.copilot_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> ApiResponse:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(warnings=warnings or []),
    )


@router.get("/config", response_model=ApiResponse)
def config(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: CopilotService = Depends(get_copilot_service),
):
    _require_copilot_user(_settings(request), current_user)
    return _wrap(request, service.get_config())


@router.post("/recent-data", response_model=ApiResponse)
def recent_data(
    request: Request,
    payload: CopilotDataWindowRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: CopilotService = Depends(get_copilot_service),
):
    _require_copilot_user(_settings(request), current_user)
    return _wrap(request, service.get_recent_data(payload.model_dump()))


@router.post("/anomaly", response_model=ApiResponse)
def anomaly(
    request: Request,
    payload: CopilotAnomalyRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: CopilotService = Depends(get_copilot_service),
):
    _require_copilot_user(_settings(request), current_user)
    return _wrap(request, service.analyze_anomaly(payload.model_dump()))


@router.post("/analyze", response_model=ApiResponse)
def analyze(
    request: Request,
    payload: CopilotAnalyzeRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: CopilotService = Depends(get_copilot_service),
):
    _require_copilot_user(_settings(request), current_user)
    return _wrap(
        request,
        service.analyze_question(payload.model_dump(), request_id=get_request_id(request)),
    )


@router.post("/jobs", response_model=ApiResponse)
def start_job(
    request: Request,
    payload: CopilotAnalyzeRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: CopilotService = Depends(get_copilot_service),
):
    _require_copilot_user(_settings(request), current_user)
    job = service.start_analysis_job(payload.model_dump(), request_id=get_request_id(request))
    return _wrap(request, compute_job_service.response(job))


@router.get("/jobs/{job_id}", response_model=ApiResponse)
def get_job(
    request: Request,
    job_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: CopilotService = Depends(get_copilot_service),
):
    _require_copilot_user(_settings(request), current_user)
    return _wrap(request, service.get_job_status(job_id))


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(
    request: Request,
    artifact_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    _require_copilot_user(_settings(request), current_user)
    service = ComputeArtifactService(_settings(request))
    try:
        metadata = service.get_metadata(artifact_id)
        path = service.get_path(artifact_id)
    except ValueError as exc:
        raise ApiError("COPILOT_ARTIFACT_NOT_FOUND", "Invalid artifact id.", status_code=400) from exc
    except FileNotFoundError as exc:
        raise ApiError("COPILOT_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404) from exc
    if metadata.workflow != "copilot":
        raise ApiError("COPILOT_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404)
    return FileResponse(
        path=path,
        media_type=metadata.content_type,
        filename=metadata.filename,
        headers={"X-Request-ID": get_request_id(request)},
    )
