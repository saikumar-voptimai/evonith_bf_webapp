"""Unified job API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class UnifiedJob(BaseModel):
    job_id: str
    source: str
    workflow: str
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


class UnifiedJobList(BaseModel):
    items: list[UnifiedJob]
    total: int
    limit: int
    offset: int

