"""Operational status API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StatusSummary(BaseModel):
    status: str
    health: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] | None = None


class DependencyStatus(BaseModel):
    status: str
    timeout_seconds: int
    cache_seconds: int
    dependencies: list[dict[str, Any]] = Field(default_factory=list)

