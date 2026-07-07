"""API v1 dataset routes."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiResponse
from apps.backend_api.app.api.v1.schemas.datasets import DatasetRefreshRequest
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.models.schemas import FetchDatasetRequest, TaskCreatedResponse, TaskStatusResponse, UpdateStaticRequest
from apps.backend_api.app.routes import dataset as legacy_dataset
from apps.backend_api.app.services import dataset_service
from apps.backend_api.app.services.artifact_service import get_artifact_metadata, get_artifact_path

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _wrap(request: Request, data) -> ApiResponse:
    return ApiResponse(request_id=get_request_id(request), data=data)


@router.get("", response_model=ApiResponse)
def list_datasets(request: Request):
    return _wrap(request, [dataset.model_dump() for dataset in dataset_service.list_datasets()])


@router.get("/{dataset_id}/preview", response_model=ApiResponse)
def preview_dataset(request: Request, dataset_id: str, limit: int = 500):
    preview = dataset_service.preview_dataset(dataset_id, limit=limit)
    return _wrap(request, preview.model_dump())


@router.post("/refresh", response_model=ApiResponse)
def refresh_dataset(request: Request, refresh_request: DatasetRefreshRequest):
    job = dataset_service.refresh_dataset(refresh_request, request_id=get_request_id(request))
    return _wrap(request, job.model_dump())


@router.get("/jobs/{job_id}", response_model=ApiResponse)
def get_dataset_job(request: Request, job_id: str):
    job = dataset_service.get_job(job_id)
    return _wrap(request, job.model_dump())


@router.get("/jobs/{job_id}/download")
def download_dataset_job(job_id: str):
    job = dataset_service.get_job(job_id)
    if not job.artifact_id:
        raise ApiError("DATASET_ARTIFACT_NOT_FOUND", "Dataset job has no artifact", status_code=404)
    return download_dataset_artifact(job.artifact_id)


@router.get("/artifacts/{artifact_id}/download")
def download_dataset_artifact(artifact_id: str):
    try:
        metadata = get_artifact_metadata(artifact_id)
        path = get_artifact_path(artifact_id)
    except ValueError as exc:
        raise ApiError("DATASET_ARTIFACT_NOT_FOUND", "Invalid artifact id", status_code=400) from exc
    except FileNotFoundError as exc:
        raise ApiError("DATASET_ARTIFACT_NOT_FOUND", "Dataset artifact not found", status_code=404) from exc
    if not path.exists():
        raise ApiError("DATASET_ARTIFACT_NOT_FOUND", "Dataset artifact file not found", status_code=404)
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename)


@router.post("/fetch", response_model=TaskCreatedResponse)
def fetch_dataset(req: FetchDatasetRequest):
    return legacy_dataset.fetch_dataset(req)


@router.post("/update-static", response_model=TaskCreatedResponse)
def update_static(req: UpdateStaticRequest):
    return legacy_dataset.update_static(req)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    return legacy_dataset.get_task_status(task_id)


@router.get("/download/{task_id}")
def download_result(task_id: str):
    return legacy_dataset.download_result(task_id)


@router.get("/static")
def download_static():
    return legacy_dataset.download_static()


@router.get("/cache-info")
def cache_info() -> Dict[str, Any]:
    return legacy_dataset.cache_info()
