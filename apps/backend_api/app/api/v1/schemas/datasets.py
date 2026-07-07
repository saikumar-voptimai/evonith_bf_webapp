"""Stable API v1 schemas for dataset access."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from apps.backend_api.app.api.v1.schemas.data import DataColumnInfo


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str | None = None
    available: bool
    source: str | None = None
    row_count: int | None = None
    last_updated: datetime | None = None
    columns: list[str] | None = None


class DatasetRefreshRequest(BaseModel):
    dataset_id: str | None = "static_ml_dataset"
    source: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    force: bool = False
    options: dict[str, Any] | None = None


class DatasetJobResponse(BaseModel):
    job_id: str
    status: str
    message: str | None = None
    request_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    artifact_id: str | None = None
    download_url: str | None = None


class DatasetJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifact_id: str | None = None
    download_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: list[DataColumnInfo]
    rows: list[dict[str, Any]]
    row_count: int | None
    returned_rows: int
    truncated: bool
