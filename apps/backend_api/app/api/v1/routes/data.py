"""API v1 Data Explorer routes with typed, owner-aware contracts."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiErrorResponse, ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.data import (
    DataCatalogResponse,
    DataExportRequest,
    DataExportResponse,
    DataPreviewResponse,
    DataQueryRequest,
    DataSourceInfo,
    HotMetalSlagExportRequest,
    HotMetalSlagPreviewRequest,
    HotMetalSlagPreviewResponse,
)
from apps.backend_api.app.core.auth_dependencies import (
    require_authenticated_user,
    require_permission,
)
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services import data_service
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
    503: {"model": ApiErrorResponse, "description": "The data source is temporarily unavailable."},
}


router = APIRouter(prefix="/data", tags=["data"], responses=_API_ERROR_RESPONSES)


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> ApiResponse[Any]:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(warnings=warnings or []),
    )


def _settings(request: Request):
    return getattr(request.app.state, "backend_settings", None)


def _owner_id(current_user: dict[str, Any]) -> str:
    user_id = str(current_user.get("id") or "").strip()
    if not user_id:
        raise ApiError("FORBIDDEN", "Authenticated user identity is unavailable.", status_code=403)
    return user_id


def _require_export_artifact_permission(
    current_user: dict[str, Any] = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Require ordinary export access or the elevated cross-owner export grant."""
    permissions = {str(item) for item in current_user.get("permissions") or []}
    if not permissions.intersection({"data:export", "data:export:any"}):
        raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)
    return current_user


@router.get(
    "/catalog",
    response_model=ApiResponse[DataCatalogResponse],
    operation_id="dataCatalog",
)
def get_catalog(
    request: Request,
    _: dict[str, Any] = Depends(require_permission("data:read")),
):
    return _wrap(request, data_service.get_data_catalog(_settings(request)))


@router.get(
    "/sources",
    response_model=ApiResponse[list[DataSourceInfo]],
    operation_id="dataListSources",
)
def list_sources(
    request: Request,
    _: dict[str, Any] = Depends(require_permission("data:read")),
):
    return _wrap(request, data_service.list_data_sources())


@router.get(
    "/offline/report-types",
    response_model=ApiResponse[dict[str, str]],
    operation_id="dataListOfflineReports",
)
def offline_report_types(
    request: Request,
    _: dict[str, Any] = Depends(require_permission("data:read")),
):
    return _wrap(request, data_service.list_offline_report_types())


@router.get(
    "/offline/tables",
    response_model=ApiResponse[dict[str, Any]],
    operation_id="dataListOfflineTables",
)
def offline_tables(
    request: Request,
    _: dict[str, Any] = Depends(require_permission("data:read")),
):
    # Compatibility endpoint now returns only the catalog's public logical IDs.
    return _wrap(request, data_service.list_offline_tables())


@router.post(
    "/preview",
    response_model=ApiResponse[DataPreviewResponse],
    operation_id="dataPreview",
)
def preview_data(
    request: Request,
    query: DataQueryRequest,
    _: dict[str, Any] = Depends(require_permission("data:read")),
):
    preview = data_service.preview_data(query, settings=_settings(request))
    return _wrap(request, preview, warnings=preview.warnings)


@router.post(
    "/export",
    response_model=ApiResponse[DataExportResponse],
    operation_id="dataCreateExport",
)
def export_data(
    request: Request,
    export_request: DataExportRequest,
    idempotency_key: Annotated[
        str, Header(..., alias="Idempotency-Key", min_length=1, max_length=200)
    ],
    current_user: dict[str, Any] = Depends(require_permission("data:export")),
):
    export = data_service.export_data(
        export_request,
        owner_user_id=_owner_id(current_user),
        idempotency_key=idempotency_key,
        settings=_settings(request),
    )
    return _wrap(request, export.response, warnings=export.warnings)


@router.get(
    "/artifacts/{artifact_id}/download",
    response_class=FileResponse,
    operation_id="dataDownloadArtifact",
    responses={
        200: {
            "description": "Authenticated CSV artifact download.",
            "content": {
                "text/csv": {"schema": {"type": "string", "format": "binary"}}
            },
        },
        403: {"model": ApiErrorResponse, "description": "Artifact belongs to a different user."},
        404: {"model": ApiErrorResponse, "description": "Artifact was not found."},
        410: {"model": ApiErrorResponse, "description": "Artifact expired."},
    },
)
def download_artifact(
    artifact_id: str,
    current_user: dict[str, Any] = Depends(_require_export_artifact_permission),
):
    try:
        metadata = get_artifact_metadata(artifact_id)
    except ValueError as exc:
        raise ApiError("ARTIFACT_NOT_FOUND", "Artifact was not found.", status_code=404) from exc
    except ArtifactExpiredError as exc:
        raise ApiError("ARTIFACT_EXPIRED", "Artifact has expired.", status_code=410) from exc
    except ArtifactNotFoundError as exc:
        raise ApiError("ARTIFACT_NOT_FOUND", "Artifact was not found.", status_code=404) from exc

    if not artifact_is_accessible_by(
        metadata,
        user_id=_owner_id(current_user),
        permissions=current_user.get("permissions") or [],
    ):
        raise ApiError("FORBIDDEN", "You are not allowed to download this artifact.", status_code=403)
    try:
        path = get_artifact_path(artifact_id)
    except ArtifactExpiredError as exc:
        raise ApiError("ARTIFACT_EXPIRED", "Artifact has expired.", status_code=410) from exc
    except (ArtifactNotFoundError, ValueError) as exc:
        raise ApiError("ARTIFACT_NOT_FOUND", "Artifact was not found.", status_code=404) from exc
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename)


@router.post(
    "/hot-metal-slag/preview",
    response_model=ApiResponse[HotMetalSlagPreviewResponse],
    operation_id="dataPreviewHotMetalSlag",
)
def preview_hot_metal_slag(
    request: Request,
    payload: HotMetalSlagPreviewRequest,
    _: dict[str, Any] = Depends(require_permission("data:read")),
):
    preview = data_service.preview_hot_metal_slag(payload, settings=_settings(request))
    return _wrap(request, preview, warnings=preview.warnings)


@router.post(
    "/hot-metal-slag/export",
    response_model=ApiResponse[DataExportResponse],
    operation_id="dataExportHotMetalSlag",
)
def export_hot_metal_slag(
    request: Request,
    payload: HotMetalSlagExportRequest,
    idempotency_key: Annotated[
        str, Header(..., alias="Idempotency-Key", min_length=1, max_length=200)
    ],
    current_user: dict[str, Any] = Depends(require_permission("data:export")),
):
    export = data_service.export_hot_metal_slag(
        payload,
        owner_user_id=_owner_id(current_user),
        idempotency_key=idempotency_key,
        settings=_settings(request),
    )
    return _wrap(request, export.response, warnings=export.warnings)
