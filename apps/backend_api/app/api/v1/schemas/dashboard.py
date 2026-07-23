"""Dashboard API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from apps.backend_api.app.api.v1.schemas.common import ApiMeta


DashboardWindow = Literal["1h"]
DashboardBucket = Literal["15m"]


class KpiMetric(BaseModel):
    value: float | None = None
    unit: str


class DashboardKpiMetrics(BaseModel):
    production_rate: KpiMetric
    fuel_rate: KpiMetric
    eta_co: KpiMetric
    blast_volume: KpiMetric


class DashboardKpisResponse(BaseModel):
    as_of: datetime
    window: DashboardWindow
    bucket: DashboardBucket
    sample_count: int = Field(..., ge=0)
    metrics: DashboardKpiMetrics


class DashboardKpisApiResponse(BaseModel):
    request_id: str
    data: DashboardKpisResponse
    meta: ApiMeta = Field(default_factory=ApiMeta)
