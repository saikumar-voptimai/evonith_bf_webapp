"""Operational metrics API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetricsSnapshot(BaseModel):
    started_at: str
    generated_at: str
    requests_total: int
    errors_total: int
    duration_ms: dict[str, float] = Field(default_factory=dict)
    status_codes: dict[str, int] = Field(default_factory=dict)
    methods: dict[str, int] = Field(default_factory=dict)
    routes: dict[str, int] = Field(default_factory=dict)
    error_codes: dict[str, int] = Field(default_factory=dict)

