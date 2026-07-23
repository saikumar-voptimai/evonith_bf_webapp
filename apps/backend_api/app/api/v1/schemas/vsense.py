"""Typed API schemas for V-Sense advisory optimization."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apps.backend_api.app.api.v1.schemas.common import ApiResponse


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VSenseTargetDefinition(_StrictModel):
    id: str
    label: str
    unit: str | None = None
    direction: Literal["maximize", "minimize"]
    precision: int


class VSenseOptimizationType(_StrictModel):
    id: str
    label: str
    target: VSenseTargetDefinition
    control_parameter_ids: list[str]
    input_groups: list[dict[str, Any]]
    impact_target_ids: list[str]
    default_algorithm_version: str


class VSenseParameterDefinition(_StrictModel):
    id: str
    label: str
    group_id: str
    role: Literal["control", "input", "target"]
    value_type: Literal["number"]
    unit: str | None = None
    precision: int
    nullable: bool
    override_allowed: bool
    approved_min: float | None = None
    approved_max: float | None = None


class VSenseAlgorithmVersion(_StrictModel):
    id: str
    label: str
    status: str


class VSenseIterationBudget(_StrictModel):
    id: str
    label: str
    max_iterations: int


class VSenseCapabilities(_StrictModel):
    llm_review_available: bool
    advanced_diagnostics_available: bool
    historical_context_available: bool
    run_cancellation_available: bool


class VSenseLimits(_StrictModel):
    context_ttl_seconds: int
    max_input_overrides: int
    max_concurrent_runs: int
    lambda_min: float
    lambda_max: float


class VSenseCatalogData(_StrictModel):
    catalog_version: str
    display_timezone: str
    advisory_only: bool
    optimization_types: list[VSenseOptimizationType]
    parameters: list[VSenseParameterDefinition]
    algorithm_versions: list[VSenseAlgorithmVersion]
    iteration_budgets: list[VSenseIterationBudget]
    capabilities: VSenseCapabilities
    limits: VSenseLimits


class VSenseContextCreateRequest(_StrictModel):
    optimization_type_id: str
    data_mode: Literal["live", "historical_only"] = "live"
    as_of: datetime | None = None


class VSenseDatasetSnapshot(_StrictModel):
    dataset_id: str
    version: str
    range_end: datetime
    staleness_seconds: int


class VSenseModelSnapshot(_StrictModel):
    optimization_type_id: str
    bundle_version: str
    status: str


class VSenseControlProfileSummary(_StrictModel):
    profile_id: str
    version: int
    parameters: list[dict[str, Any]] = Field(default_factory=list)


class VSenseContextValue(_StrictModel):
    parameter_id: str
    value: float | None
    source: str
    source_timestamp: datetime
    freshness_seconds: int
    quality: str
    observed_min: float | None = None
    observed_max: float | None = None
    approved_min: float | None = None
    approved_max: float | None = None


class VSenseContextInputGroup(_StrictModel):
    id: str
    label: str
    values: list[VSenseContextValue]


class VSenseContextTarget(_StrictModel):
    parameter_id: str
    value: float | None
    source: str
    source_timestamp: datetime
    quality: str


class VSenseContextData(_StrictModel):
    context_id: str
    created_at: datetime
    expires_at: datetime
    as_of: datetime
    display_timezone: str
    optimization_type_id: str
    catalog_version: str
    algorithm_version: str
    dataset: VSenseDatasetSnapshot
    models: list[VSenseModelSnapshot]
    control_profile: VSenseControlProfileSummary
    controls: list[VSenseContextValue]
    input_groups: list[VSenseContextInputGroup]
    target: VSenseContextTarget
    warnings: list[str] = Field(default_factory=list)
    idempotent_replay: bool = False


VSenseControlMode = Literal["optimize", "fixed"]


class VSenseControlPlanItem(_StrictModel):
    parameter_id: str
    mode: VSenseControlMode
    lower_bound: float
    upper_bound: float
    fixed_value: float | None = None

    @field_validator("lower_bound", "upper_bound", "fixed_value")
    @classmethod
    def finite_number(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("value must be finite")
        return value


class VSenseControlProfileData(_StrictModel):
    profile_id: str
    optimization_type_id: str
    version: int
    catalog_version: str
    parameters: list[VSenseControlPlanItem]
    updated_by_user_id: str | None = None
    updated_by_username: str | None = None
    created_at: datetime
    updated_at: datetime
    idempotent_replay: bool = False


class VSenseControlProfileUpdateRequest(_StrictModel):
    profile_id: str = "plant-default"
    expected_version: int
    parameters: list[VSenseControlPlanItem]


class VSenseInputOverride(_StrictModel):
    parameter_id: str
    value: float

    @field_validator("value")
    @classmethod
    def finite_override(cls, value: float) -> float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("value must be finite")
        return value


class VSenseRunOptions(_StrictModel):
    lambda_reg: float = 0.05
    iteration_budget_id: str = "standard"
    max_iterations: int | None = None
    seed: int | None = None
    request_llm_review: bool = False
    advanced_diagnostics: bool = False


class VSenseRunCreateRequest(_StrictModel):
    context_id: str
    optimization_type_id: str
    control_plan: list[VSenseControlPlanItem]
    input_overrides: list[VSenseInputOverride] = Field(default_factory=list)
    options: VSenseRunOptions = Field(default_factory=VSenseRunOptions)


class VSenseRunAcceptedData(_StrictModel):
    run_id: str
    status: str
    created_at: datetime
    status_url: str
    events_url: str
    cancellable: bool
    idempotent_replay: bool = False


class VSenseTargetResult(_StrictModel):
    parameter_id: str
    label: str
    unit: str | None = None
    direction: Literal["maximize", "minimize"]
    baseline: float | None
    recommended: float | None
    delta: float | None
    delta_pct: float | None = None


class VSenseControlResult(_StrictModel):
    parameter_id: str
    label: str
    unit: str | None = None
    mode: VSenseControlMode
    baseline: float | None
    recommended: float | None
    delta: float | None
    delta_pct: float | None = None
    lower_bound: float
    upper_bound: float
    at_bound: bool
    approved_min: float | None = None
    approved_max: float | None = None


class VSenseImpactResult(_StrictModel):
    parameter_id: str
    label: str
    unit: str | None = None
    baseline: float | None
    recommended: float | None
    delta: float | None
    delta_pct: float | None = None
    bundle_version: str


class VSenseDependentParameterResult(_StrictModel):
    parameter_id: str
    label: str
    unit: str | None = None
    baseline: float | None
    recommended: float | None
    delta: float | None


class VSenseFeasibilityResult(_StrictModel):
    feasible: bool
    violations: list[dict[str, Any]]
    operator_review_required: bool


class VSenseOptimizerDiagnostics(_StrictModel):
    algorithm_version: str
    seed: int
    lambda_reg: float
    optimizer: dict[str, Any]
    missing_feature_policy: str
    input_override_parameter_ids: list[str]


class VSenseVersionMetadata(_StrictModel):
    catalog_version: str
    algorithm_version: str
    dataset_version: str | None = None
    control_profile_version: int | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)


class VSenseReviewResult(_StrictModel):
    available: bool
    prompt_version: str | None = None
    markdown: str | None = None
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class VSenseRunResult(_StrictModel):
    advisory_only: bool
    requires_operator_review: bool
    status: str
    target: VSenseTargetResult
    controls: list[VSenseControlResult]
    impacts: list[VSenseImpactResult]
    dependent_parameters: list[VSenseDependentParameterResult]
    feasibility: VSenseFeasibilityResult
    diagnostics: VSenseOptimizerDiagnostics
    versions: VSenseVersionMetadata
    warnings: list[str] = Field(default_factory=list)
    review: VSenseReviewResult | None = None
    completed_at: datetime


class VSenseRunStatusData(_StrictModel):
    run_id: str
    context_id: str
    optimization_type_id: str
    status: str
    progress: float | None = None
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    cancellable: bool
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    result: VSenseRunResult | None = None


class VSenseRunEvent(_StrictModel):
    sequence: int
    stage: str
    progress: float
    message: str
    created_at: datetime


class VSenseRunEventsData(_StrictModel):
    run_id: str
    events: list[VSenseRunEvent]


class VSenseCatalogResponse(ApiResponse[VSenseCatalogData]):
    pass


class VSenseContextResponse(ApiResponse[VSenseContextData]):
    pass


class VSenseControlProfileResponse(ApiResponse[VSenseControlProfileData]):
    pass


class VSenseRunAcceptedResponse(ApiResponse[VSenseRunAcceptedData]):
    pass


class VSenseRunStatusResponse(ApiResponse[VSenseRunStatusData]):
    pass


class VSenseRunEventsResponse(ApiResponse[VSenseRunEventsData]):
    pass
