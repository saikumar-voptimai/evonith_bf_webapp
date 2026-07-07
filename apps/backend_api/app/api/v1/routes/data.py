"""API v1 data access routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.data import DataExportRequest, DataQueryRequest
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services import data_service
from apps.backend_api.app.services.artifact_service import get_artifact_metadata, get_artifact_path

router = APIRouter(prefix="/data", tags=["data"])


def _wrap(request: Request, data, warnings: list[str] | None = None) -> ApiResponse:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(warnings=warnings or []),
    )


@router.get("/sources", response_model=ApiResponse)
def list_sources(request: Request):
    return _wrap(request, [source.model_dump() for source in data_service.list_data_sources()])


@router.get("/offline/report-types", response_model=ApiResponse)
def offline_report_types(request: Request):
    return _wrap(request, data_service.list_offline_report_types())


@router.get("/offline/tables", response_model=ApiResponse)
def offline_tables(request: Request):
    return _wrap(request, data_service.list_offline_tables())


@router.post("/preview", response_model=ApiResponse)
def preview_data(request: Request, query: DataQueryRequest):
    preview = data_service.preview_data(query)
    return _wrap(request, preview.model_dump(), warnings=preview.warnings)


@router.post("/export", response_model=ApiResponse)
def export_data(request: Request, export_request: DataExportRequest):
    export = data_service.export_data(export_request)
    return _wrap(request, export.model_dump())


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str):
    try:
        metadata = get_artifact_metadata(artifact_id)
        path = get_artifact_path(artifact_id)
    except ValueError as exc:
        raise ApiError("DATA_EXPORT_FAILED", "Invalid artifact id", status_code=400) from exc
    except FileNotFoundError as exc:
        raise ApiError("DATA_EXPORT_FAILED", "Artifact not found", status_code=404) from exc
    if not path.exists():
        raise ApiError("DATA_EXPORT_FAILED", "Artifact file not found", status_code=404)
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename)
