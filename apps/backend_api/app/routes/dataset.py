"""Authorized compatibility adapters for the canonical static dataset service.

The historical ``/dataset`` routes intentionally retain their response shapes,
but they no longer own a fetcher, cache manager, or in-memory task registry.
Every operation delegates to the durable static-dataset job service.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.datasets import (
    BuildRangeJobRequest,
    DatasetJobOptions,
    ExtendJobRequest,
    OverrideJobRequest,
)
from apps.backend_api.app.config import settings
from apps.backend_api.app.core.auth_dependencies import (
    require_authenticated_user,
    require_permission,
)
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.models.schemas import (
    FetchDatasetRequest,
    TaskCreatedResponse,
    TaskStatus,
    TaskStatusResponse,
    UpdateStaticRequest,
)
from apps.backend_api.app.services import dataset_service
from apps.backend_api.app.services.artifact_service import (
    ArtifactExpiredError,
    ArtifactNotFoundError,
    artifact_is_accessible_by,
    get_artifact_metadata,
    get_artifact_path,
)


router = APIRouter(prefix="/dataset", tags=["dataset"])


def _owner_id(current_user: dict[str, Any]) -> str:
    user_id = str(current_user.get("id") or "").strip()
    if not user_id:
        raise ApiError("FORBIDDEN", "Authenticated user identity is required.", 403)
    return user_id


def _utc_start(value) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _utc_end(value) -> datetime:
    return datetime.combine(value, time.max, tzinfo=timezone.utc)


def _legacy_options(*, validate: bool, produce_download: bool = True) -> DatasetJobOptions:
    return DatasetJobOptions(
        validate_dataset=validate,
        produce_download=produce_download,
    )


def _legacy_task_status(value: str) -> TaskStatus:
    if value in {status.value for status in TaskStatus}:
        return TaskStatus(value)
    # Legacy clients cannot represent canonical cancelled/expired states.
    return TaskStatus.failed


def _task_created(created) -> TaskCreatedResponse:
    return TaskCreatedResponse(
        task_id=created.job_id,
        status=_legacy_task_status(created.status),
        message="Task created",
    )


def _task_status(status) -> TaskStatusResponse:
    progress = None if status.progress is None else f"{float(status.progress):.0f}%"
    return TaskStatusResponse(
        task_id=status.job_id,
        status=_legacy_task_status(status.status),
        progress=progress,
        created_at=status.created_at,
        completed_at=status.completed_at,
        rows=None,
        columns=None,
        error=status.error_message or status.error_code,
    )


def _assert_artifact_access(metadata: Any, current_user: dict[str, Any]) -> None:
    permissions = {str(value) for value in current_user.get("permissions") or []}
    if (
        artifact_is_accessible_by(
            metadata,
            user_id=_owner_id(current_user),
            permissions=permissions,
        )
        or "datasets:override" in permissions
        or str(current_user.get("role") or "").lower() == "admin"
    ):
        return
    raise ApiError("FORBIDDEN", "You do not have access to this dataset artifact.", 403)


def _stream_job_artifact(job_id: str, current_user: dict[str, Any]) -> FileResponse:
    status = dataset_service.get_job(job_id, current_user)
    if status.status == "expired":
        raise ApiError("ARTIFACT_EXPIRED", "Dataset job download has expired.", 410)
    if not status.artifact_id:
        raise ApiError("ARTIFACT_NOT_FOUND", "Dataset job has no artifact.", 404)
    try:
        metadata = get_artifact_metadata(status.artifact_id)
        _assert_artifact_access(metadata, current_user)
        path = get_artifact_path(status.artifact_id)
    except ArtifactExpiredError as exc:
        raise ApiError("ARTIFACT_EXPIRED", "Dataset artifact has expired.", 410) from exc
    except (ArtifactNotFoundError, ValueError) as exc:
        raise ApiError("ARTIFACT_NOT_FOUND", "Dataset artifact was not found.", 404) from exc
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename)


def _legacy_idempotency_key() -> str:
    # Legacy request schemas have no client idempotency header. Preserve their
    # one-request/one-task behavior without weakening canonical job semantics.
    return f"legacy-{uuid4().hex}"


@router.post("/fetch", response_model=TaskCreatedResponse)
def fetch_dataset(
    req: FetchDatasetRequest,
    current_user: dict[str, Any] = Depends(require_permission("datasets:build")),
):
    """Build a downloadable candidate range through the durable job service."""
    if req.start_date > req.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")
    if req.rm_choice.value != "charge":
        raise ApiError(
            "INVALID_DATASET_OPTION",
            "The canonical static dataset supports RM Charge only.",
            422,
        )
    if req.callback_url:
        raise ApiError(
            "UNSUPPORTED_LEGACY_OPTION",
            "Callback URLs are not supported by the durable dataset job service.",
            422,
        )
    created = dataset_service.submit_static_dataset_job(
        BuildRangeJobRequest(
            operation="build_range",
            start=_utc_start(req.start_date),
            end=_utc_end(req.end_date),
            options=_legacy_options(validate=req.apply_cleaning),
        ),
        current_user=current_user,
        idempotency_key=_legacy_idempotency_key(),
    )
    return _task_created(created)


@router.post("/update-static", response_model=TaskCreatedResponse)
def update_static(
    req: UpdateStaticRequest,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    """Refresh/override through the canonical version-checked job contract."""
    if req.rm_choice.value != "charge":
        raise ApiError(
            "INVALID_DATASET_OPTION",
            "The canonical static dataset supports RM Charge only.",
            422,
        )
    if req.callback_url:
        raise ApiError(
            "UNSUPPORTED_LEGACY_OPTION",
            "Callback URLs are not supported by the durable dataset job service.",
            422,
        )
    metadata = dataset_service.get_static_metadata()
    end = datetime.now(timezone.utc)
    options = _legacy_options(validate=req.apply_cleaning)
    if req.reprocess_from is not None:
        payload = OverrideJobRequest(
            operation="override",
            start=_utc_start(req.reprocess_from),
            end=end,
            expected_dataset_version=metadata.version,
            options=options,
        )
    else:
        payload = ExtendJobRequest(
            operation="extend",
            end=end,
            expected_dataset_version=metadata.version,
            options=options,
        )
    created = dataset_service.submit_static_dataset_job(
        payload,
        current_user=current_user,
        idempotency_key=_legacy_idempotency_key(),
    )
    return _task_created(created)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(
    task_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    """Map durable job status into the historical task response shape."""
    return _task_status(dataset_service.get_job(task_id, current_user))


@router.get("/download/{task_id}", response_class=FileResponse)
def download_result(
    task_id: str,
    current_user: dict[str, Any] = Depends(require_authenticated_user),
):
    """Download the owner-scoped artifact produced by a durable job."""
    status = dataset_service.get_job(task_id, current_user)
    if status.status not in {"completed", "expired"}:
        raise HTTPException(
            status_code=409,
            detail=f"Task is {status.status}, not yet completed",
        )
    return _stream_job_artifact(task_id, current_user)


@router.get("/static", response_class=FileResponse)
def download_static(
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
):
    """Download canonical public-column CSV bytes for the current dataset."""
    _ = current_user
    path, version = dataset_service.current_dataset_download()
    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=f"static_ml_dataset_{version[:12]}.csv",
        headers={"X-Dataset-Version": version},
    )


@router.get("/cache-info")
def cache_info(
    current_user: dict[str, Any] = Depends(require_permission("data:read")),
) -> Dict[str, Any]:
    """Project canonical metadata into the historical cache-info response."""
    _ = current_user
    try:
        metadata = dataset_service.get_static_metadata()
    except ApiError as exc:
        if exc.code == "DATASET_NOT_AVAILABLE":
            return {
                "status": "no_cache",
                "detail": "No canonical static dataset is available.",
            }
        raise
    resolved_range = metadata.range
    if resolved_range is None:
        return {"status": "no_cache", "detail": "Canonical dataset range is unavailable."}
    raw_end = resolved_range.end.date()
    confirmed_end = raw_end - timedelta(days=max(0, int(settings.offline_lag_days)))
    return {
        "status": "ok",
        "data_start": resolved_range.start.date().isoformat(),
        "confirmed_end": confirmed_end.isoformat(),
        "raw_end": raw_end.isoformat(),
        "offline_lag_days": settings.offline_lag_days,
        "last_updated": metadata.last_built_at.isoformat() if metadata.last_built_at else None,
        "rows": metadata.row_count,
        "columns": metadata.column_count,
        "csv_file": "static_ml_dataset.csv",
        "rm_choice": "charge",
    }