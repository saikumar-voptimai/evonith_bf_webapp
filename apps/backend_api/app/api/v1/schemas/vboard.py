"""Typed API schemas for V-Board data visualisation."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from apps.backend_api.app.api.v1.schemas.common import ApiResponse


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VBoardPresetTimeRange(_StrictModel):
    kind: Literal["preset"]
    preset_id: str


class VBoardAbsoluteTimeRange(_StrictModel):
    kind: Literal["absolute"]
    start: datetime
    end: datetime


VBoardTimeRange = Annotated[
    VBoardPresetTimeRange | VBoardAbsoluteTimeRange,
    Field(discriminator="kind"),
]


class VBoardAutoResolution(_StrictModel):
    mode: Literal["auto"] = "auto"


class VBoardFixedResolution(_StrictModel):
    mode: Literal["fixed"]
    window_id: str


VBoardResolution = Annotated[
    VBoardAutoResolution | VBoardFixedResolution,
    Field(discriminator="mode"),
]


class VBoardPreset(_StrictModel):
    id: str
    label: str
    duration_seconds: int
    supported_for: list[str]


class VBoardRowDefinition(_StrictModel):
    id: str
    label: str
    order: int


class VBoardQuadrantDefinition(_StrictModel):
    id: str
    label: str
    order: int


class VBoardTemperatureLevelDefinition(_StrictModel):
    id: str
    elevation_m: float
    label: str
    sensor_count: int
    order: int


class VBoardTemperatureGroup(_StrictModel):
    id: str
    title: str
    level_ids: list[str]


class VBoardGeometryPoint(_StrictModel):
    x: float
    y: float


class VBoardGeometryRegion(_StrictModel):
    id: str
    label: str
    elevation_m: float


class VBoardLongitudinalGeometry(_StrictModel):
    profile_points: list[VBoardGeometryPoint]
    regions: list[VBoardGeometryRegion]
    x_range: tuple[float, float]
    y_range: tuple[float, float]


class VBoardDisplayMetadata(_StrictModel):
    temperature_unit: str
    heatload_unit: str | None = None
    heatload_label: str


class VBoardProcessingPolicy(_StrictModel):
    id: str
    description: str


class VBoardLimits(_StrictModel):
    max_absolute_range_days: int
    max_timeseries_points_per_quadrant: int
    max_source_rows: int


class VBoardResolutionWindow(_StrictModel):
    id: str
    label: str
    seconds: int


class VBoardCatalogData(_StrictModel):
    catalog_version: str
    display_timezone: str
    presets: list[VBoardPreset]
    rows: list[VBoardRowDefinition]
    quadrants: list[VBoardQuadrantDefinition]
    temperature_levels: list[VBoardTemperatureLevelDefinition]
    circumferential_temperature_groups: list[VBoardTemperatureGroup]
    longitudinal_geometry: VBoardLongitudinalGeometry
    display: VBoardDisplayMetadata
    processing_policy: VBoardProcessingPolicy
    limits: VBoardLimits
    resolution_windows: list[VBoardResolutionWindow] = Field(default_factory=list)


class VBoardResolvedRange(_StrictModel):
    start: datetime
    end: datetime
    requested_kind: Literal["preset", "absolute"]
    preset_id: str | None = None


VBoardSectionStatus = Literal["ok", "partial", "empty", "unavailable"]


class VBoardStatistic(_StrictModel):
    quadrant_id: str
    mean: float | None
    minimum: float | None
    maximum: float | None


class VBoardTemperatureStatistic(VBoardStatistic):
    valid_sensor_count: int


class VBoardTemperatureLevel(_StrictModel):
    level_id: str
    elevation_m: float
    quadrants: list[VBoardTemperatureStatistic]


class VBoardTemperatureSection(_StrictModel):
    status: VBoardSectionStatus
    unit: str
    levels: list[VBoardTemperatureLevel]
    missing_level_ids: list[str]
    warnings: list[str]


class VBoardHeatloadRow(_StrictModel):
    row_id: str
    quadrants: list[VBoardStatistic]


class VBoardHeatloadSection(_StrictModel):
    status: VBoardSectionStatus
    unit: str | None = None
    display_label: str
    rows: list[VBoardHeatloadRow]
    missing_row_ids: list[str]
    warnings: list[str]


class VBoardContoursRequest(_StrictModel):
    time_range: VBoardTimeRange


class VBoardContoursData(_StrictModel):
    generated_at: datetime
    resolved_range: VBoardResolvedRange
    catalog_version: str
    processing_policy_id: str
    temperature: VBoardTemperatureSection
    heatload: VBoardHeatloadSection


class VBoardSeriesPoint(_StrictModel):
    timestamp: datetime
    value: float | None


class VBoardSeries(_StrictModel):
    quadrant_id: str
    points: list[VBoardSeriesPoint]
    returned_points: int
    missing_points: int


class VBoardTimeseriesProcessing(_StrictModel):
    policy_id: str
    smoothing_kind: str
    smoothing_window_seconds: int
    normalization: str


class VBoardHeatloadTimeseriesRequest(_StrictModel):
    row_id: str
    time_range: VBoardTimeRange
    resolution: VBoardResolution = Field(default_factory=VBoardAutoResolution)


class VBoardHeatloadTimeseriesData(_StrictModel):
    generated_at: datetime
    resolved_range: VBoardResolvedRange
    row: VBoardRowDefinition
    unit: str | None = None
    display_label: str
    resolved_window_seconds: int
    processing: VBoardTimeseriesProcessing
    series: list[VBoardSeries]
    downsampled: bool
    warnings: list[str]


class VBoardCatalogResponse(ApiResponse[VBoardCatalogData]):
    pass


class VBoardContoursResponse(ApiResponse[VBoardContoursData]):
    pass


class VBoardHeatloadTimeseriesResponse(ApiResponse[VBoardHeatloadTimeseriesData]):
    pass
