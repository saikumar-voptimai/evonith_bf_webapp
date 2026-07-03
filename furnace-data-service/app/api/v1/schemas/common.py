"""Common API v1 response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiMeta(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    api_version: str = "v1"


class ApiResponse(BaseModel):
    request_id: str
    data: Any
    meta: ApiMeta = Field(default_factory=ApiMeta)


class ApiErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApiErrorResponse(BaseModel):
    request_id: str
    error: ApiErrorBody
