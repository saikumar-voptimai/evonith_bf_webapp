"""Typed Material Balance API schemas."""

from __future__ import annotations

from datetime import date as date_type, datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from apps.backend_api.app.api.v1.schemas.common import ApiResponse
from apps.backend_api.app.api.v1.schemas.compute_common import ComputeArtifact, ComputeWarning, TableData


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DateRange(_StrictModel):
    minimum: date_type | None = None
    maximum: date_type | None = None


class MaterialBalanceDatasetInfo(_StrictModel):
    dataset_id: str
    version: str | None = None
    status: Literal["ready", "missing", "not_ready"]
    available_date_range: DateRange


class MaterialBalanceDefaults(_StrictModel):
    rm_lag_hours: int
    blast_lag_hours: int
    dust_catcher_t: float
    algorithm_version: str


class MaterialBalanceLimits(_StrictModel):
    rm_lag_hours_min: int
    rm_lag_hours_max: int
    blast_lag_hours_min: int
    blast_lag_hours_max: int
    dust_catcher_t_min: float
    dust_catcher_t_max: float


class ClosureThresholdRange(_StrictModel):
    minimum: float
    maximum: float


class ClosureThresholds(_StrictModel):
    good: ClosureThresholdRange
    warning: ClosureThresholdRange


class CatalogItem(_StrictModel):
    id: str
    label: str
    unit: str | None = None


class MaterialBalanceAlgorithmVersion(_StrictModel):
    id: str
    label: str
    tracked_element_ids: list[str]


class MaterialBalanceCapabilities(_StrictModel):
    runtime_configuration_writable: bool
    ash_analysis_editable: bool
    dpr_mapping_editable: bool
    export_available: bool
    async_jobs_required: bool


class MaterialBalanceConfigData(_StrictModel):
    catalog_version: str
    effective_config_version: str
    display_timezone: str
    dataset: MaterialBalanceDatasetInfo
    defaults: MaterialBalanceDefaults
    limits: MaterialBalanceLimits
    closure_thresholds: ClosureThresholds
    elements: list[CatalogItem]
    materials: list[CatalogItem]
    input_streams: list[CatalogItem]
    output_streams: list[CatalogItem]
    algorithm_versions: list[MaterialBalanceAlgorithmVersion]
    capabilities: MaterialBalanceCapabilities
    available_sources: list[str] = Field(default_factory=list)
    warnings: list[ComputeWarning] = Field(default_factory=list)


class MaterialBalanceRunOptions(_StrictModel):
    rm_lag_hours: int = 0
    blast_lag_hours: int = 0
    dust_catcher_t: float = 0.0
    algorithm_version: str = "legacy_v1"

    @field_validator("dust_catcher_t")
    @classmethod
    def finite_dust(cls, value: float) -> float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("dust_catcher_t must be finite")
        return value


class MaterialBalanceRunRequest(_StrictModel):
    source: Literal["static_dataset", "input_data"] = "static_dataset"
    day: date_type | None = Field(default=None, validation_alias=AliasChoices("day", "date"))
    expected_dataset_version: str | None = None
    expected_config_version: str | None = None
    options: MaterialBalanceRunOptions = Field(default_factory=MaterialBalanceRunOptions)
    export_format: Literal["closure_csv", "full_json"] | None = None
    export: bool = False
    input_data: dict[str, Any] | None = None
    # Deprecated compatibility fields. They are accepted but ignored for new runs.
    start_time: datetime | None = None
    end_time: datetime | None = None
    timezone: str | None = None
    async_job: bool = False


class MaterialBalanceRefreshCacheRequest(_StrictModel):
    day: date_type | None = None
    scopes: list[Literal["calculation_snapshot", "dpr"]]


class MaterialBalanceWindowData(_StrictModel):
    local_start: datetime
    local_end: datetime
    utc_start: datetime
    utc_end: datetime


class MaterialBalanceVersions(_StrictModel):
    dataset_version: str
    config_version: str
    catalog_version: str


class MaterialBalanceSummary(_StrictModel):
    overall_closure_pct: float | None = None
    closure_status: str
    total_input_element_t: float
    total_output_element_t: float
    delta_t: float
    hot_metal_mass_t: float | None = None
    slag_mass_t: float | None = None
    burden_mass_t: float | None = None
    dust_catcher_mass_t: float | None = None


class MaterialBalanceClosureRow(_StrictModel):
    element_id: str
    symbol: str
    label: str
    input_t: float | None = None
    output_t: float | None = None
    closure_pct: float | None = None
    delta_t: float | None = None
    status: str


class MaterialBalanceMassRow(_StrictModel):
    material_id: str
    label: str
    mass_t: float | None = None
    source: str
    source_field_id: str | None = None
    canonical_field_id: str | None = None
    quality: str


class ElementMass(_StrictModel):
    element_id: str
    symbol: str
    mass_t: float | None = None


class MaterialBalanceStream(_StrictModel):
    stream_id: str
    label: str
    total_t: float | None = None
    elements: list[ElementMass] = Field(default_factory=list)


class MaterialBalanceDiagramFlow(_StrictModel):
    flow_id: str
    label: str
    mass_t: float | None = None


class MaterialBalanceGasPhase(_StrictModel):
    wind_nm3_per_hour: float | None = None
    oxygen_flow_nm3_per_hour: float | None = None
    steam_kg_per_hour: float | None = None
    top_gas_nm3_per_day: float | None = None
    top_gas_method: str
    top_gas_fallback_applied: bool
    hot_blast_mass_t: float | None = None
    pci_plus_steam_mass_t: float | None = None


class MaterialBalanceAssumption(_StrictModel):
    id: str
    label: str
    text: str
    details: dict[str, Any] = Field(default_factory=dict)


class MaterialBalanceResult(_StrictModel):
    calculation_id: str
    computed_at: datetime
    day: date_type
    algorithm_version: str
    window_policy_version: str
    versions: MaterialBalanceVersions
    resolved_windows: dict[str, MaterialBalanceWindowData]
    summary: MaterialBalanceSummary
    closure_thresholds: ClosureThresholds
    closure: list[MaterialBalanceClosureRow]
    material_masses: list[MaterialBalanceMassRow]
    input_streams: list[MaterialBalanceStream]
    output_streams: list[MaterialBalanceStream]
    diagram_flows: dict[str, list[MaterialBalanceDiagramFlow]]
    gas_phase: MaterialBalanceGasPhase
    data_quality: dict[str, Any]
    warnings: list[ComputeWarning]
    assumptions: list[MaterialBalanceAssumption]
    artifacts: list[ComputeArtifact] = Field(default_factory=list)
    # Compatibility with earlier Phase 7 wrappers.
    summary_legacy: dict[str, Any] | None = None
    kpis: dict[str, Any] = Field(default_factory=dict)
    tables: dict[str, TableData] = Field(default_factory=dict)
    charts: dict[str, Any] = Field(default_factory=dict)


class MaterialBalanceValidationResponse(_StrictModel):
    valid: bool
    errors: list[ComputeWarning]
    warnings: list[ComputeWarning]


class AshSpeciesValue(_StrictModel):
    species_id: str
    label: str | None = None
    basis: Literal["ash", "net_fuel"] = "ash"
    value: float


class AshAnalysisMaterial(_StrictModel):
    material_id: Literal["coke", "nutcoke", "pci"]
    label: str
    species: list[AshSpeciesValue]


class MaterialBalanceAshAnalysesData(_StrictModel):
    config_version: str
    materials: list[AshAnalysisMaterial]
    writable: bool


class MaterialBalanceAshAnalysesUpdateRequest(_StrictModel):
    expected_config_version: str
    materials: list[AshAnalysisMaterial]


class DprSourceField(_StrictModel):
    source_field_id: str
    label: str
    data_type: Literal["number"] = "number"
    unit: str
    aggregation_policy: str


class DprCanonicalField(_StrictModel):
    canonical_field_id: str
    label: str
    unit: str
    aggregation_policy: str


class DprMappingItem(_StrictModel):
    canonical_field_id: str
    source_field_id: str | None = None


class MaterialBalanceDprMappingData(_StrictModel):
    config_version: str
    status: Literal["none", "partial", "complete"]
    canonical_fields: list[DprCanonicalField]
    mapping: list[DprMappingItem]
    approved_source_fields: list[DprSourceField]
    selected_day_availability: dict[str, Any] | None = None
    writable: bool


class MaterialBalanceDprMappingUpdateRequest(_StrictModel):
    expected_config_version: str
    mapping: list[DprMappingItem]


class MaterialBalanceCacheRefreshData(_StrictModel):
    invalidated_scopes: list[str]
    day: date_type | None = None


class MaterialBalanceConfigResponse(ApiResponse[MaterialBalanceConfigData]):
    pass


class MaterialBalanceRunResponse(ApiResponse[MaterialBalanceResult]):
    pass


class MaterialBalanceValidateResponse(ApiResponse[MaterialBalanceValidationResponse]):
    pass


class MaterialBalanceAshAnalysesResponse(ApiResponse[MaterialBalanceAshAnalysesData]):
    pass


class MaterialBalanceDprMappingResponse(ApiResponse[MaterialBalanceDprMappingData]):
    pass


class MaterialBalanceCacheRefreshResponse(ApiResponse[MaterialBalanceCacheRefreshData]):
    pass