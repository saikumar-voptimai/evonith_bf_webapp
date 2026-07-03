"""Service layer for API v1 dataset access."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd

from app.api.v1.schemas.datasets import (
    DatasetInfo,
    DatasetJobResponse,
    DatasetJobStatus,
    DatasetPreviewResponse,
    DatasetRefreshRequest,
)
from app.config import settings
from app.core.errors import ApiError
from app.services.artifact_service import create_csv_artifact
from app.services.job_service import JobState, job_service
from app.services.serialization import dataframe_to_preview
from furnace_data.dataset.static import StaticDatasetManager


STATIC_DATASET_ID = "static_ml_dataset"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def max_preview_rows() -> int:
    return max(1, _env_int("DATA_API_MAX_PREVIEW_ROWS", 500))


def artifact_ttl_hours() -> int:
    return max(1, _env_int("DATA_API_ARTIFACT_TTL_HOURS", 24))


def _make_manager() -> StaticDatasetManager:
    return StaticDatasetManager(
        static_dir=settings.static_dir,
        offline_lag_days=settings.offline_lag_days,
        max_versions=settings.static_max_versions,
        legacy_csv_path=settings.legacy_csv_path or None,
    )


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_static_dataset_dataframe() -> pd.DataFrame:
    manager = _make_manager()
    csv_path = manager.current_csv_path()
    if not csv_path:
        raise ApiError(
            code="DATASET_NOT_FOUND",
            message="Static ML dataset is not available",
            status_code=404,
        )
    df = pd.read_csv(csv_path, parse_dates=[0])
    if df.empty:
        return df
    return df.set_index(df.columns[0])


def list_datasets() -> list[DatasetInfo]:
    manager = _make_manager()
    meta = manager.get_meta()
    csv_path = manager.current_csv_path()
    columns: list[str] | None = None
    row_count = meta.rows if meta else None
    if csv_path and csv_path.exists():
        try:
            sample = pd.read_csv(csv_path, nrows=1)
            columns = list(sample.columns)
            if row_count is None:
                row_count = sum(1 for _line in csv_path.open("r", encoding="utf-8", errors="ignore")) - 1
        except Exception:
            columns = None
    return [
        DatasetInfo(
            id=STATIC_DATASET_ID,
            name="Static ML dataset",
            description="Runtime cached furnace ML dataset",
            available=bool(csv_path and csv_path.exists()),
            source="runtime" if meta else "legacy_fallback" if csv_path else None,
            row_count=max(row_count or 0, 0) if row_count is not None else None,
            last_updated=_parse_datetime(meta.last_updated) if meta else None,
            columns=columns,
        )
    ]


def preview_dataset(dataset_id: str, limit: int = 500) -> DatasetPreviewResponse:
    if dataset_id != STATIC_DATASET_ID:
        raise ApiError("DATASET_NOT_FOUND", f"Unknown dataset: {dataset_id}", status_code=404)
    capped_limit = min(max(limit, 0), max_preview_rows())
    df = load_static_dataset_dataframe()
    columns, rows, row_count, truncated = dataframe_to_preview(
        df,
        limit=capped_limit,
        include_index=True,
    )
    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        columns=columns,
        rows=rows,
        row_count=row_count,
        returned_rows=len(rows),
        truncated=truncated,
    )


def _job_download_url(job: JobState) -> str | None:
    if not job.artifact_id:
        return None
    return f"/api/v1/datasets/artifacts/{job.artifact_id}/download"


def job_to_status(job: JobState) -> DatasetJobStatus:
    return DatasetJobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        error_code=job.error_code,
        error_message=job.error_message,
        artifact_id=job.artifact_id,
        download_url=_job_download_url(job),
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )


def refresh_dataset(request: DatasetRefreshRequest, request_id: str | None = None) -> DatasetJobResponse:
    dataset_id = request.dataset_id or STATIC_DATASET_ID
    if dataset_id != STATIC_DATASET_ID:
        raise ApiError("DATASET_NOT_FOUND", f"Unknown dataset: {dataset_id}", status_code=404)

    job = job_service.create_job("Dataset refresh queued")

    def _run(job_state: JobState) -> None:
        manager = _make_manager()
        options: dict[str, Any] = request.options or {}
        rm_choice = options.get("rm_choice") or "RM Charge"
        apply_cleaning = bool(options.get("apply_cleaning", True))
        job_service.update_job(job_state.job_id, progress=0.2, message="Updating static dataset")
        df = manager.update_static(
            rm_choice=rm_choice,
            reprocess_from=request.start_time.date() if request.force and request.start_time else None,
            apply_cleaning=apply_cleaning,
        )
        if df.empty:
            raise ApiError("DATASET_REFRESH_FAILED", "Dataset refresh returned no rows", status_code=500)
        job_service.update_job(job_state.job_id, progress=0.8, message="Saving dataset artifact")
        manager.save(df, rm_choice)
        artifact = create_csv_artifact(df, "static_ml_dataset", ttl_hours=artifact_ttl_hours())
        job_service.update_job(
            job_state.job_id,
            status="completed",
            progress=1.0,
            message="Dataset refresh completed",
            artifact_id=artifact.artifact_id,
        )

    job_service.run_background(job, _run)
    return DatasetJobResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        request_id=request_id,
        created_at=job.created_at,
        updated_at=job.updated_at,
        artifact_id=job.artifact_id,
        download_url=_job_download_url(job),
    )


def get_job(job_id: str) -> DatasetJobStatus:
    job = job_service.get_job(job_id)
    if not job:
        raise ApiError("DATASET_JOB_NOT_FOUND", f"Dataset job not found: {job_id}", status_code=404)
    return job_to_status(job)
