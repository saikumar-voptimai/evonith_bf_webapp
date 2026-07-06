"""Health and readiness response schemas for API v1."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.api.v1.schemas.common import ApiMeta


class HealthData(BaseModel):
    status: str
    service: str
    api_version: str
    environment: str


class HealthResponse(BaseModel):
    request_id: str
    data: HealthData
    meta: ApiMeta = Field(default_factory=ApiMeta)


class ReadinessData(BaseModel):
    status: str
    checks: dict[str, str]


class ReadinessResponse(BaseModel):
    request_id: str
    data: ReadinessData
    meta: ApiMeta = Field(default_factory=ApiMeta)


class RuntimeStatusData(BaseModel):
    status: str
    runtime_dir: str | None = None
    checks: dict[str, str]
    directories: dict[str, str]
    disk: dict[str, Any] = Field(default_factory=dict)


class RuntimeStatusResponse(BaseModel):
    request_id: str
    data: RuntimeStatusData
    meta: ApiMeta = Field(default_factory=ApiMeta)
