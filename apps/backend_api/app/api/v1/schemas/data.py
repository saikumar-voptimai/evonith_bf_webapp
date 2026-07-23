"""Frontend-neutral API v1 schemas for Data Explorer data access.

The query models deliberately expose logical identifiers only. Database table
names, Flux, SQL, filesystem paths and UI state are implementation details and
never form part of this contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)


class _StrictModel(BaseModel):
    """Base model that rejects undocumented, browser-specific fields."""

    model_config = ConfigDict(extra="forbid")


def _utc_rfc3339(value: datetime) -> str:
    """Serialize a timezone-aware value as canonical UTC RFC 3339."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class DataSourceInfo(_StrictModel):
    """Compatibility metadata for the small legacy sources endpoint."""

    id: str
    name: str
    kind: str
    description: str | None = None
    supports_preview: bool = True
    supports_export: bool = True


class DataColumnInfo(_StrictModel):
    """A canonical data column description used by previews and catalogues."""

    id: str
    label: str
    dtype: str
    unit: str | None = None
    description: str | None = None


class CatalogMeasurement(_StrictModel):
    id: str
    label: str
    fields: list[DataColumnInfo]


class TimePreset(_StrictModel):
    id: str
    label: str
    duration_seconds: int = Field(gt=0)
    sources: list[Literal["online", "offline"]] = Field(min_length=1)


class AggregationWindow(_StrictModel):
    id: str
    label: str
    duration_seconds: int | None = Field(default=None, ge=1)


class OfflineReportInfo(_StrictModel):
    id: str
    label: str
    fields: list[DataColumnInfo] = Field(default_factory=list)
    supports_field_selection: bool = True


class OfflineTableInfo(_StrictModel):
    """A public stable logical table identifier, never a physical table name."""

    id: str
    label: str
    fields: list[DataColumnInfo] = Field(default_factory=list)
    supports_field_selection: bool = True
    supports_aggregation: bool = False


class DataCatalogLimits(_StrictModel):
    max_preview_rows: int = Field(ge=1)
    max_selected_fields: int = Field(ge=1)
    max_scatter_points: int = Field(ge=1)
    max_timeseries_points_per_field: int = Field(ge=1)
    max_hm_slag_interval_minutes: int = Field(ge=1)


class DataCatalogResponse(_StrictModel):
    display_timezone: str
    online_measurements: list[CatalogMeasurement]
    time_presets: list[TimePreset]
    aggregation_windows: list[AggregationWindow]
    offline_reports: list[OfflineReportInfo]
    offline_tables: list[OfflineTableInfo]
    limits: DataCatalogLimits


class PresetTimeRange(_StrictModel):
    kind: Literal["preset"]
    preset_id: str = Field(min_length=1, max_length=80)


class AbsoluteTimeRange(_StrictModel):
    kind: Literal["absolute"]
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_timezone_aware_order(self) -> "AbsoluteTimeRange":
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("start must include a timezone offset")
        if self.end.tzinfo is None or self.end.utcoffset() is None:
            raise ValueError("end must include a timezone offset")
        if self.start > self.end:
            raise ValueError("start must be before or equal to end")
        return self


TimeRange: TypeAlias = Annotated[
    PresetTimeRange | AbsoluteTimeRange,
    Field(discriminator="kind"),
]


class AggregationRequest(_StrictModel):
    """Backend-supported aggregation. ``None`` means no aggregation."""

    mode: Literal["mean"]
    window_id: str = Field(min_length=1, max_length=80)


class OfflineReportSelection(_StrictModel):
    kind: Literal["report"]
    report_id: str = Field(min_length=1, max_length=100)


class OfflineTableSelection(_StrictModel):
    kind: Literal["table"]
    table_id: str = Field(min_length=1, max_length=100)


OfflineSelection: TypeAlias = Annotated[
    OfflineReportSelection | OfflineTableSelection,
    Field(discriminator="kind"),
]


class _QueryPaging(_StrictModel):
    limit: int = Field(default=500, ge=1, le=1_000_000)
    offset: int = Field(default=0, ge=0)

    @field_validator("limit", "offset", mode="before")
    @classmethod
    def reject_boolean_paging_values(cls, value: object) -> object:
        # Pydantic otherwise accepts ``true`` as integer one.
        if isinstance(value, bool):
            raise ValueError("must be an integer")
        return value


class OnlineDataQuery(_QueryPaging):
    source: Literal["online"]
    measurements: list[str] = Field(min_length=1, max_length=20)
    time_range: TimeRange
    aggregation: AggregationRequest | None = None
    fields: list[str] | None = Field(default=None, max_length=20)

    @field_validator("measurements", "fields")
    @classmethod
    def normalize_ids(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized = [str(value).strip() for value in values]
        if not all(normalized):
            raise ValueError("identifiers must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("identifiers must not contain duplicates")
        return normalized


class OfflineDataQuery(_QueryPaging):
    source: Literal["offline"]
    selection: OfflineSelection
    time_range: TimeRange
    aggregation: None = None
    fields: list[str] | None = Field(default=None, max_length=20)

    @field_validator("fields")
    @classmethod
    def normalize_fields(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized = [str(value).strip() for value in values]
        if not all(normalized):
            raise ValueError("identifiers must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("identifiers must not contain duplicates")
        return normalized


DataQueryRequest: TypeAlias = Annotated[
    OnlineDataQuery | OfflineDataQuery,
    Field(discriminator="source"),
]


class ResolvedTimeRange(_StrictModel):
    start: datetime
    end: datetime

    @field_serializer("start", "end", when_used="json")
    def serialize_utc(self, value: datetime) -> str:
        return _utc_rfc3339(value)


class DataPreviewResponse(_StrictModel):
    columns: list[DataColumnInfo]
    rows: list[dict[str, JsonValue]]
    total_rows: int = Field(ge=0)
    # ``row_count`` is retained for older clients while they migrate to the
    # unambiguous ``total_rows`` name.
    row_count: int = Field(ge=0)
    returned_rows: int = Field(ge=0)
    offset: int = Field(ge=0)
    truncated: bool
    source: Literal["online", "offline"]
    resolved_range: ResolvedTimeRange | None = None
    warnings: list[str] = Field(default_factory=list)


class DataExportRequest(_StrictModel):
    query: DataQueryRequest
    format: Literal["csv"] = "csv"


class DataExportResponse(_StrictModel):
    artifact_id: str
    filename: str
    content_type: Literal["text/csv"]
    row_count: int = Field(ge=0)
    download_path: str
    expires_at: datetime

    @field_serializer("expires_at", when_used="json")
    def serialize_expiry(self, value: datetime) -> str:
        return _utc_rfc3339(value)


class HotMetalSlagInterpolation(_StrictModel):
    numeric: Literal["time"] = "time"
    metadata: Literal["forward_backward_fill"] = "forward_backward_fill"


class HotMetalSlagPreviewRequest(_StrictModel):
    start: datetime
    end: datetime
    interval_minutes: int = Field(default=60, ge=1, le=600)
    interpolation: HotMetalSlagInterpolation = Field(
        default_factory=HotMetalSlagInterpolation
    )
    limit: int = Field(default=500, ge=1, le=1_000_000)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> "HotMetalSlagPreviewRequest":
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("start must include a timezone offset")
        if self.end.tzinfo is None or self.end.utcoffset() is None:
            raise ValueError("end must include a timezone offset")
        if self.start > self.end:
            raise ValueError("start must be before or equal to end")
        return self


class HotMetalSlagExportRequest(_StrictModel):
    query: HotMetalSlagPreviewRequest
    format: Literal["csv"] = "csv"


class HotMetalSlagPreviewResponse(_StrictModel):
    columns: list[DataColumnInfo]
    rows: list[dict[str, JsonValue]]
    returned_rows: int = Field(ge=0)
    total_rows: int = Field(ge=0)
    offset: int = Field(ge=0)
    truncated: bool
    resolved_range: ResolvedTimeRange
    interval_minutes: int = Field(ge=1)
    synthetic_row_count: int = Field(ge=0)
    interpolated_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
