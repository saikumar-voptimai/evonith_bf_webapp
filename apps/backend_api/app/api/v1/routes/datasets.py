"""API v1 routes for the canonical static ML dataset."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiErrorResponse, ApiResponse
from apps.backend_api.app.api.v1.schemas.datasets import (
    DatasetJobCreated,
    DatasetJobEventsResponse,
    DatasetJobRequest,
    DatasetJobStatus,
    DatasetInfo,
    DatasetPreviewResponse,
    DatasetRefreshRequest,
    DatasetJobResponse,
    StaticDatasetJobStatus,
    StaticDatasetMetadata,
    StaticDatasetValidation,
    ScatterAnalysisRequest,
    ScatterAnalysisResponse,
    TimeSeriesRequest,
    TimeSeriesResponse,
)
from apps.backend_api.app.core.auth_dependencies import (
    require_authenticated_user,
    require_permission,
)
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.models.schemas import FetchDatasetRequest, TaskCreatedResponse, TaskStatusResponse, UpdateStaticRequest
from apps.backend_api.app.routes import dataset as legacy_dataset
from apps.backend_api.app.services import dataset_service
from apps.backend_api.app.services.artifact_service import (
    ArtifactExpiredError,
    ArtifactNotFoundError,
    artifact_is_accessible_by,
    get_artifact_metadata,
    get_artifact_path,
)


_API_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ApiErrorResponse, "description": "The request could not be processed."},
    401: {"model": ApiErrorResponse, "description": "Authentication is required."},
    403: {"model": ApiErrorResponse, "description": "The caller is not permitted to perform this action."},
    404: {"model": ApiErrorResponse, "description": "The requested resource was not found."},
    409: {"model": ApiErrorResponse, "description": "The request conflicts with existing state."},
    410: {"model": ApiErrorResponse, "description": "The requested resource has expired."},
    413: {"model": ApiErrorResponse, "description": "The request exceeds a configured size limit."},
    422: {"model": ApiErrorResponse, "description": "The request payload is invalid."},
    500: {"model": ApiErrorResponse, "description": "An unexpected server error occurred."},
    503: {"model": ApiErrorResponse, "description": "The dataset source is temporarily unavailable."},
}


router = APIRouter(prefix="/datasets", tags=["datasets"], responses=_API_ERROR_RESPONSES)

_CSV_DOWNLOAD_RESPONSE = {
    200: {
        "description": "Owner-scoped CSV download.",
        "content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}},
    }
}

def _wrap(request: Request, data: Any) -> ApiResponse[Any]:
    return ApiResponse(request_id=get_request_id(request), data=data)


def _assert_artifact_access(metadata: Any, current_user: dict[str, Any]) -> None:
    user_id = str(current_user.get("id") or "").strip()
    permissions = {str(value) for value in current_user.get("permissions") or []}
    if user_id and (
        artifact_is_accessible_by(
            metadata,
            user_id=user_id,
            permissions=permissions,
        )
        or "datasets:override" in permissions
        or str(current_user.get("role") or "").lower() == "admin"
    ):
        return
    raise ApiError("FORBIDDEN", "You do not have access to this dataset artifact.", 403)


def _stream_artifact(artifact_id: str, current_user: dict[str, Any]) -> FileResponse:
    try:
        metadata = get_artifact_metadata(artifact_id)
        _assert_artifact_access(metadata, current_user)
        path = get_artifact_path(artifact_id)
    except ArtifactExpiredError as exc:
        raise ApiError("ARTIFACT_EXPIRED", "Dataset artifact has expired.", 410) from exc
    except (ArtifactNotFoundError, ValueError) as exc:
        raise ApiError("ARTIFACT_NOT_FOUND", "Dataset artifact was not found.", 404) from exc
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename)


def _stream_job_artifact(job_id: str, current_user: dict[str, Any]) -> FileResponse:
    status = dataset_service.get_static_job(job_id, current_user)
    if status.status == "expired":
        raise ApiError("ARTIFACT_EXPIRED", "Dataset job download has expired.", 410)
    if not status.artifact_id:
        raise ApiError("ARTIFACT_NOT_FOUND", "Dataset job has no download artifact.", 404)
    return _stream_artifact(status.artifact_id, current_user)


@router.get(
    "/static_ml_dataset",
    response_model=ApiResponse[StaticDatasetMetadata],
    operation_id="getStaticMlDataset",
)
def get_static_ml_dataset(
    request: Request,
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    _ = current_user
    return _wrap(request, dataset_service.get_static_metadata())


@router.post(
    "/static_ml_dataset/analyses/scatter",
    response_model=ApiResponse[ScatterAnalysisResponse],
    operation_id="analyzeStaticMlDatasetScatter",
)
def analyze_static_ml_dataset_scatter(
    request: Request,
    payload: ScatterAnalysisRequest,
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    _ = current_user
    return _wrap(request, dataset_service.get_scatter_analysis(payload))


@router.post(
    "/static_ml_dataset/timeseries",
    response_model=ApiResponse[TimeSeriesResponse],
    operation_id="getStaticMlDatasetTimeseries",
)
def get_static_ml_dataset_timeseries(
    request: Request,
    payload: TimeSeriesRequest,
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    _ = current_user
    return _wrap(request, dataset_service.get_timeseries(payload))


@router.post(
    "/static_ml_dataset/jobs",
    response_model=ApiResponse[DatasetJobCreated],
    operation_id="createStaticMlDatasetJob",
)
def create_static_ml_dataset_job(
    request: Request,
    payload: DatasetJobRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255),
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return _wrap(
        request,
        dataset_service.submit_static_dataset_job(
            payload,
            current_user=current_user,
            idempotency_key=idempotency_key,
        ),
    )


@router.get(
    "/static_ml_dataset/jobs/{job_id}",
    response_model=ApiResponse[StaticDatasetJobStatus],
    operation_id="getStaticMlDatasetJob",
)
def get_static_ml_dataset_job(
    request: Request,
    job_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return _wrap(request, dataset_service.get_static_job(job_id, current_user))


@router.get(
    "/static_ml_dataset/jobs/{job_id}/events",
    response_model=ApiResponse[DatasetJobEventsResponse],
    operation_id="getStaticMlDatasetJobEvents",
)
def get_static_ml_dataset_job_events(
    request: Request,
    job_id: str,
    after: int = Query(0, ge=0),
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return _wrap(request, dataset_service.get_static_job_events(job_id, current_user, after=after))


@router.post(
    "/static_ml_dataset/jobs/{job_id}/cancel",
    response_model=ApiResponse[StaticDatasetJobStatus],
    operation_id="cancelStaticMlDatasetJob",
)
def cancel_static_ml_dataset_job(
    request: Request,
    job_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return _wrap(request, dataset_service.cancel_static_job(job_id, current_user))


@router.get(
    "/static_ml_dataset/jobs/{job_id}/download",
    operation_id="downloadStaticMlDatasetJob",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Owner-scoped static dataset job CSV artifact.",
            "content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
def download_static_ml_dataset_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return _stream_job_artifact(job_id, current_user)


@router.get(
    "/static_ml_dataset/download",
    operation_id="downloadCurrentStaticMlDataset",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Canonical static dataset CSV with stable public field IDs.",
            "content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
def download_current_static_ml_dataset(
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    _ = current_user
    path, version = dataset_service.current_dataset_download()
    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=f"static_ml_dataset_{version[:12]}.csv",
        headers={"X-Dataset-Version": version},
    )


@router.get(
    "/static_ml_dataset/validation",
    response_model=ApiResponse[StaticDatasetValidation],
    operation_id="getStaticMlDatasetValidation",
)
def get_static_ml_dataset_validation(
    request: Request,
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    _ = current_user
    return _wrap(request, dataset_service.get_static_validation())


# Compatibility endpoints below are intentionally retained while older callers
# migrate to the versioned static_ml_dataset contract above.


@router.get("", response_model=ApiResponse[list[DatasetInfo]], operation_id="listDatasets")
def list_datasets(
    request: Request,
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    _ = current_user
    return _wrap(request, [dataset.model_dump() for dataset in dataset_service.list_datasets()])


@router.get("/{dataset_id}/preview", response_model=ApiResponse[DatasetPreviewResponse], operation_id="previewDataset")
def preview_dataset(
    request: Request,
    dataset_id: str,
    limit: int = 500,
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    _ = current_user
    preview = dataset_service.preview_dataset(dataset_id, limit=limit)
    return _wrap(request, preview.model_dump())


@router.post("/refresh", response_model=ApiResponse[DatasetJobResponse], operation_id="refreshDataset")
def refresh_dataset(
    request: Request,
    refresh_request: DatasetRefreshRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key", min_length=1, max_length=255),
    current_user: dict[str, Any] = Depends(require_permission("datasets:refresh")),
):
    job = dataset_service.refresh_dataset(
        refresh_request,
        request_id=get_request_id(request),
        current_user=current_user,
        idempotency_key=idempotency_key,
    )
    return _wrap(request, job.model_dump())


@router.get("/jobs/{job_id}", response_model=ApiResponse[DatasetJobStatus], operation_id="getLegacyDatasetJob")
def get_dataset_job(
    request: Request,
    job_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    job = dataset_service.get_job(job_id, current_user)
    return _wrap(request, job.model_dump())


@router.get("/jobs/{job_id}/download", operation_id="downloadLegacyDatasetJob", response_class=FileResponse, responses=_CSV_DOWNLOAD_RESPONSE)
def download_dataset_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    job = dataset_service.get_job(job_id, current_user)
    if job.status == "expired":
        raise ApiError("ARTIFACT_EXPIRED", "Dataset job download has expired.", 410)
    if not job.artifact_id:
        raise ApiError("ARTIFACT_NOT_FOUND", "Dataset job has no artifact.", status_code=404)
    return _stream_artifact(job.artifact_id, current_user)


@router.get("/artifacts/{artifact_id}/download", operation_id="downloadLegacyDatasetArtifact", response_class=FileResponse, responses=_CSV_DOWNLOAD_RESPONSE)
def download_dataset_artifact(
    artifact_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return _stream_artifact(artifact_id, current_user)


@router.post("/fetch", response_model=TaskCreatedResponse, operation_id="legacyFetchDataset")
def fetch_dataset(
    req: FetchDatasetRequest,
    current_user: dict[str, Any] = Depends(require_permission("data:export")),
):
    return legacy_dataset.fetch_dataset(req, current_user=current_user)


@router.post("/update-static", response_model=TaskCreatedResponse, operation_id="legacyUpdateStaticDataset")
def update_static(
    req: UpdateStaticRequest,
    current_user: dict[str, Any] = Depends(require_permission("datasets:refresh")),
):
    return legacy_dataset.update_static(req, current_user=current_user)


@router.get("/status/{task_id}", response_model=TaskStatusResponse, operation_id="getLegacyDatasetTaskStatus")
def get_task_status(
    task_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return legacy_dataset.get_task_status(task_id, current_user=current_user)


@router.get("/download/{task_id}", operation_id="downloadLegacyDatasetTask", response_class=FileResponse, responses=_CSV_DOWNLOAD_RESPONSE)
def download_result(
    task_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    return legacy_dataset.download_result(task_id, current_user=current_user)


@router.get("/static", operation_id="downloadLegacyStaticDataset", response_class=FileResponse, responses=_CSV_DOWNLOAD_RESPONSE)
def download_static(
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    return download_current_static_ml_dataset(current_user=current_user)


@router.get("/cache-info", operation_id="getLegacyDatasetCacheInfo")
def cache_info(
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
) -> dict[str, Any]:
    return legacy_dataset.cache_info(current_user=current_user)
