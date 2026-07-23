"""Stable API v1 schemas for dataset access."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from apps.backend_api.app.api.v1.schemas.data import DataColumnInfo


STATIC_DATASET_ID = "static_ml_dataset"


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str | None = None
    available: bool
    source: str | None = None
    row_count: int | None = None
    last_updated: datetime | None = None
    columns: list[str] | None = None


class DatasetRefreshRequest(BaseModel):
    dataset_id: str | None = "static_ml_dataset"
    source: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    force: bool = False
    options: dict[str, Any] | None = None


class DatasetJobResponse(BaseModel):
    job_id: str
    status: str
    message: str | None = None
    request_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    artifact_id: str | None = None
    download_url: str | None = None


class DatasetJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifact_id: str | None = None
    download_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    columns: list[DataColumnInfo]
    rows: list[dict[str, Any]]
    row_count: int | None
    returned_rows: int
    truncated: bool


class _StrictModel(BaseModel):
    """Reject accidental browser-only or internal fields on public contracts."""

    model_config = ConfigDict(extra="forbid")


def _require_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone offset")
    return value


class StaticDatasetColumn(_StrictModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=256)
    dtype: Literal["number", "string", "boolean", "datetime"]
    unit: str | None = Field(default=None, max_length=64)
    plottable: bool = False
    filterable: bool = False


class StaticDatasetTimeColumn(_StrictModel):
    id: str
    timezone: Literal["UTC"] = "UTC"


class DatasetRange(_StrictModel):
    start: datetime
    end: datetime

    _aware_start = field_validator("start")(_require_aware_datetime)
    _aware_end = field_validator("end")(_require_aware_datetime)

    @model_validator(mode="after")
    def _ordered(self) -> "DatasetRange":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class StaticDatasetMetadata(_StrictModel):
    dataset_id: Literal["static_ml_dataset"] = STATIC_DATASET_ID
    version: str
    etag: str
    status: Literal["ready"] = "ready"
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[StaticDatasetColumn]
    time_column: StaticDatasetTimeColumn
    range: DatasetRange | None = None
    last_built_at: datetime | None = None
    validation_status: Literal["passed", "warning", "failed", "not_run"] = "not_run"
    download_available: bool = True


class NumericRangeFilter(_StrictModel):
    field: str = Field(min_length=1, max_length=128)
    mode: Literal["inside", "outside"]
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def _ordered(self) -> "NumericRangeFilter":
        if self.minimum > self.maximum:
            raise ValueError("minimum must be less than or equal to maximum")
        return self


class RegressionRequest(_StrictModel):
    enabled: bool = False
    degree: int = Field(default=1, ge=1, le=5)


class ScatterAnalysisRequest(_StrictModel):
    dataset_version: str = Field(min_length=1, max_length=128)
    x_field: str = Field(min_length=1, max_length=128)
    y_field: str = Field(min_length=1, max_length=128)
    filter: NumericRangeFilter | None = None
    regression: RegressionRequest = Field(default_factory=RegressionRequest)
    max_points: int = Field(default=5_000, ge=1, le=50_000)


class DroppedRows(_StrictModel):
    null: int = Field(ge=0)
    non_numeric: int = Field(ge=0)
    non_finite: int = Field(ge=0)


class ScatterRegression(_StrictModel):
    degree: int = Field(ge=1, le=5)
    coefficients: list[float]
    r_squared: float | None = None
    line_x: list[float]
    line_y: list[float]


class ScatterAnalysisResponse(_StrictModel):
    dataset_version: str
    x: list[float]
    y: list[float]
    total_matching_rows: int = Field(ge=0)
    returned_points: int = Field(ge=0)
    downsampled: bool
    regression: ScatterRegression | None = None
    dropped_rows: DroppedRows


class TimeSeriesRange(_StrictModel):
    start: datetime
    end: datetime

    _aware_start = field_validator("start")(_require_aware_datetime)
    _aware_end = field_validator("end")(_require_aware_datetime)

    @model_validator(mode="after")
    def _ordered(self) -> "TimeSeriesRange":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class TimeSeriesResample(_StrictModel):
    mode: Literal["mean"] = "mean"
    window: str = Field(min_length=2, max_length=16, pattern=r"^[1-9][0-9]*(min|h|d)$")


class TimeSeriesRequest(_StrictModel):
    dataset_version: str = Field(min_length=1, max_length=128)
    fields: list[str] = Field(min_length=1, max_length=20)
    time_range: TimeSeriesRange
    filter: NumericRangeFilter | None = None
    resample: TimeSeriesResample | None = None
    max_points_per_field: int = Field(default=5_000, ge=1, le=50_000)

    @field_validator("fields")
    @classmethod
    def _unique_fields(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("fields must not contain duplicates")
        return values


class TimeSeriesPoint(_StrictModel):
    timestamp: datetime
    value: float

    _aware_timestamp = field_validator("timestamp")(_require_aware_datetime)


class TimeSeries(_StrictModel):
    field: str
    label: str
    unit: str | None = None
    points: list[TimeSeriesPoint]


class TimeSeriesResponse(_StrictModel):
    dataset_version: str
    series: list[TimeSeries]
    resolved_range: DatasetRange
    downsampled: bool

class DatasetJobOptions(_StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    validate_dataset: bool = Field(default=True, alias="validate", serialization_alias="validate")
    produce_download: bool = False


class _DatasetJobRequestBase(_StrictModel):
    options: DatasetJobOptions = Field(default_factory=DatasetJobOptions)


class BuildRangeJobRequest(_DatasetJobRequestBase):
    operation: Literal["build_range"]
    start: datetime
    end: datetime

    _aware_start = field_validator("start")(_require_aware_datetime)
    _aware_end = field_validator("end")(_require_aware_datetime)

    @model_validator(mode="after")
    def _ordered(self) -> "BuildRangeJobRequest":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class ExtendJobRequest(_DatasetJobRequestBase):
    operation: Literal["extend"]
    end: datetime
    expected_dataset_version: str = Field(min_length=1, max_length=128)

    _aware_end = field_validator("end")(_require_aware_datetime)


class OverrideJobRequest(_DatasetJobRequestBase):
    operation: Literal["override"]
    start: datetime
    end: datetime
    expected_dataset_version: str = Field(min_length=1, max_length=128)

    _aware_start = field_validator("start")(_require_aware_datetime)
    _aware_end = field_validator("end")(_require_aware_datetime)

    @model_validator(mode="after")
    def _ordered(self) -> "OverrideJobRequest":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


DatasetJobRequest = Annotated[
    Union[BuildRangeJobRequest, ExtendJobRequest, OverrideJobRequest],
    Field(discriminator="operation"),
]


class DatasetJobCreated(_StrictModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled", "expired"]
    operation: Literal["build_range", "extend", "override"]
    idempotent_replay: bool = False
    created_at: datetime


class DatasetJobEvent(_StrictModel):
    sequence: int = Field(ge=1)
    stage: str
    percent: float = Field(ge=0, le=100)
    message: str
    created_at: datetime


class DatasetJobEventsResponse(_StrictModel):
    job_id: str
    events: list[DatasetJobEvent]
    last_sequence: int = Field(ge=0)


class DatasetJobResult(_StrictModel):
    dataset_version_before: str | None = None
    dataset_version_after: str | None = None
    row_count: int | None = Field(default=None, ge=0)
    validation_status: Literal["passed", "warning", "failed", "not_run"] | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    requested_range: DatasetRange | None = None


class StaticDatasetJobStatus(_StrictModel):
    job_id: str
    operation: Literal["build_range", "extend", "override"]
    status: Literal["pending", "running", "completed", "failed", "cancelled", "expired"]
    progress: float = Field(ge=0, le=100)
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifact_id: str | None = None
    cancel_requested: bool = False
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    result: DatasetJobResult | None = None


class ValidationSummary(_StrictModel):
    errors: int = Field(ge=0)
    warnings: int = Field(ge=0)


class ValidationCheck(_StrictModel):
    id: str
    status: Literal["passed", "warning", "failed"]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class StaticDatasetValidation(_StrictModel):
    dataset_version: str
    status: Literal["passed", "warning", "failed"]
    checked_at: datetime
    summary: ValidationSummary
    checks: list[ValidationCheck]
