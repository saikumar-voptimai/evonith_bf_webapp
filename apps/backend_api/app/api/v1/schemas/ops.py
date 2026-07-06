"""Operational API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CleanupRequest(BaseModel):
    dry_run: bool | None = None
    include_logs: bool | None = None
    include_uploads: bool | None = None
    max_delete: int | None = Field(default=None, ge=1)


class CleanupResult(BaseModel):
    dry_run: bool
    would_delete: int
    deleted: int
    bytes_selected: int
    bytes_deleted: int
    max_delete: int
    truncated: bool
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)

