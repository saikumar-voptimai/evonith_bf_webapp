"""Stable API v1 schemas for data access."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DataSourceInfo(BaseModel):
    id: str
    name: str
    kind: str
    description: str | None = None
    supports_preview: bool = True
    supports_export: bool = True


class DataColumnInfo(BaseModel):
    name: str
    dtype: str | None = None
    unit: str | None = None
    description: str | None = None


class DataQueryRequest(BaseModel):
    source: str
    mode: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    report_type: str | None = None
    table_name: str | None = None
    columns: list[str] | None = None
    filters: dict[str, Any] | None = None
    limit: int | None = Field(500, ge=0)
    offset: int | None = Field(0, ge=0)
    timezone: str | None = "Asia/Kolkata"

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return value.strip().lower()


class DataPreviewResponse(BaseModel):
    columns: list[DataColumnInfo]
    rows: list[dict[str, Any]]
    row_count: int | None
    returned_rows: int
    truncated: bool
    source: str
    warnings: list[str] = Field(default_factory=list)


class DataExportRequest(BaseModel):
    query: DataQueryRequest
    format: str = "csv"

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        value = value.strip().lower()
        if value != "csv":
            raise ValueError("Only csv exports are supported in Phase 4")
        return value


class DataExportResponse(BaseModel):
    artifact_id: str
    filename: str
    content_type: str
    row_count: int | None
    download_url: str
    expires_at: datetime | None = None
