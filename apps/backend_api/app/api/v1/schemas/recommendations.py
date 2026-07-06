"""Recommendations API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.api.v1.schemas.compute_common import ComputeArtifact, ComputeWarning, TableData


class RecommendationRequest(BaseModel):
    source: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    input_data: dict[str, Any] | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    max_items: int | None = None
    include_explanations: bool = True
    async_job: bool = False
    timezone: str = "Asia/Kolkata"


class RecommendationItem(BaseModel):
    id: str
    title: str
    description: str
    priority: str | None = None
    category: str | None = None
    confidence: float | None = None
    expected_impact: dict[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ComputeWarning] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    items: list[RecommendationItem]
    summary: dict[str, Any]
    tables: dict[str, TableData] = Field(default_factory=dict)
    charts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ComputeWarning] = Field(default_factory=list)
    artifacts: list[ComputeArtifact] = Field(default_factory=list)
    computed_at: datetime
