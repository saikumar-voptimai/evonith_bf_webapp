"""Plant administration API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from apps.backend_api.app.api.v1.schemas.common import ApiMeta


class HopperOption(BaseModel):
    code: str
    display_name: str


class MaterialOption(BaseModel):
    code: str
    canonical_name: str
    display_name: str


class ConfigActor(BaseModel):
    user_id: str | None = None
    username: str | None = None


class HopperMappingContextResponse(BaseModel):
    at: datetime
    snapshot_id: int | None = None
    effective_at: datetime | None = None
    hoppers: list[HopperOption]
    materials: list[MaterialOption]
    assignments: dict[str, str | None]


class HopperMappingUpdateRequest(BaseModel):
    effective_at: datetime
    expected_snapshot_id: int | None = None
    assignments: dict[str, str | None]

    @field_validator("effective_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("effective_at must include a timezone offset")
        return value


class HopperMappingHistoryItem(BaseModel):
    snapshot_id: int
    effective_at: datetime
    assignments: dict[str, str | None]
    source_type: str | None = None
    actor: ConfigActor = Field(default_factory=ConfigActor)
    created_at: datetime | None = None


class HopperMappingHistoryResponse(BaseModel):
    items: list[HopperMappingHistoryItem]
    total: int
    limit: int
    offset: int


class HistoryDeleteRequest(BaseModel):
    record_ids: list[int] = Field(..., min_length=1)

    @field_validator("record_ids")
    @classmethod
    def require_positive_ids(cls, value: list[int]) -> list[int]:
        if any(int(item) <= 0 for item in value):
            raise ValueError("record_ids must be positive integers")
        return value


class HopperHistoryDeleteResponse(BaseModel):
    deleted_count: int
    current_context: HopperMappingContextResponse


class BurdenFieldDefinition(BaseModel):
    key: str
    label: str
    value_type: Literal["text", "number"]
    nullable: bool = True
    step: float | None = None


class BurdenDistributionContextResponse(BaseModel):
    at: datetime
    snapshot_id: int | None = None
    effective_at: datetime | None = None
    fields: list[BurdenFieldDefinition]
    values: dict[str, str | float | None]


class BurdenDistributionUpdateRequest(BaseModel):
    effective_at: datetime
    expected_snapshot_id: int | None = None
    values: dict[str, Any]

    @field_validator("effective_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("effective_at must include a timezone offset")
        return value


class BurdenDistributionHistoryItem(BaseModel):
    snapshot_id: int
    effective_at: datetime
    values: dict[str, str | float | None]
    source_type: str | None = None
    actor: ConfigActor = Field(default_factory=ConfigActor)
    created_at: datetime | None = None


class BurdenDistributionHistoryResponse(BaseModel):
    items: list[BurdenDistributionHistoryItem]
    total: int
    limit: int
    offset: int


class BurdenHistoryDeleteResponse(BaseModel):
    deleted_count: int
    current_context: BurdenDistributionContextResponse


class HopperMappingContextApiResponse(BaseModel):
    request_id: str
    data: HopperMappingContextResponse
    meta: ApiMeta = Field(default_factory=ApiMeta)


class HopperMappingHistoryApiResponse(BaseModel):
    request_id: str
    data: HopperMappingHistoryResponse
    meta: ApiMeta = Field(default_factory=ApiMeta)


class HopperHistoryDeleteApiResponse(BaseModel):
    request_id: str
    data: HopperHistoryDeleteResponse
    meta: ApiMeta = Field(default_factory=ApiMeta)


class BurdenDistributionContextApiResponse(BaseModel):
    request_id: str
    data: BurdenDistributionContextResponse
    meta: ApiMeta = Field(default_factory=ApiMeta)


class BurdenDistributionHistoryApiResponse(BaseModel):
    request_id: str
    data: BurdenDistributionHistoryResponse
    meta: ApiMeta = Field(default_factory=ApiMeta)


class BurdenHistoryDeleteApiResponse(BaseModel):
    request_id: str
    data: BurdenHistoryDeleteResponse
    meta: ApiMeta = Field(default_factory=ApiMeta)
