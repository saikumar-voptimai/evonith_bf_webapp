"""Material Balance API schemas."""

from __future__ import annotations

from datetime import date as date_type, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.api.v1.schemas.compute_common import ComputeArtifact, ComputeWarning, TableData


class MaterialBalanceConfigResponse(BaseModel):
    mappings: dict[str, Any]
    available_sources: list[str]
    defaults: dict[str, Any]
    version: str | None = None
    warnings: list[ComputeWarning] = Field(default_factory=list)


class MaterialBalanceRunRequest(BaseModel):
    source: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    date: date_type | None = None
    input_data: dict[str, Any] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    export: bool = False
    async_job: bool = False
    timezone: str = "Asia/Kolkata"


class MaterialBalanceResult(BaseModel):
    summary: dict[str, Any]
    kpis: dict[str, Any]
    tables: dict[str, TableData]
    charts: dict[str, Any]
    warnings: list[ComputeWarning]
    artifacts: list[ComputeArtifact] = Field(default_factory=list)
    computed_at: datetime


class MaterialBalanceValidationResponse(BaseModel):
    valid: bool
    errors: list[ComputeWarning]
    warnings: list[ComputeWarning]
