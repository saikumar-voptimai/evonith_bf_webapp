"""Shared schemas for backend-owned compute workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ComputeWarning(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ComputeArtifact(BaseModel):
    artifact_id: str
    filename: str
    content_type: str
    download_url: str
    expires_at: datetime | None = None


class ComputeJobResponse(BaseModel):
    job_id: str
    status: str
    message: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    artifact_id: str | None = None
    download_url: str | None = None


class ComputeJobStatus(BaseModel):
    job_id: str
    workflow: str
    status: str
    progress: float | None = None
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None
    artifact_id: str | None = None
    download_url: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class ChartSeries(BaseModel):
    name: str
    x: list[Any]
    y: list[Any]
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableData(BaseModel):
    columns: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    row_count: int
    returned_rows: int
    truncated: bool
