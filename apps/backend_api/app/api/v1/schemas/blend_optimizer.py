"""Blend Optimizer API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from apps.backend_api.app.api.v1.schemas.compute_common import ComputeArtifact, ComputeWarning, TableData


class BlendOptimizerContextResponse(BaseModel):
    materials: list[dict[str, Any]]
    constraints: dict[str, Any]
    defaults: dict[str, Any]
    models: list[dict[str, Any]]
    warnings: list[ComputeWarning] = Field(default_factory=list)


class BlendMaterialInput(BaseModel):
    material_id: str
    name: str | None = None
    available: bool = True
    min_percent: float | None = None
    max_percent: float | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None


class BlendOptimizerRequest(BaseModel):
    materials: list[BlendMaterialInput]
    constraints: dict[str, Any] = Field(default_factory=dict)
    objective: str = "min_cost"
    options: dict[str, Any] = Field(default_factory=dict)
    include_predictions: bool = True
    export: bool = False
    async_job: bool = False


class BlendPredictionRequest(BaseModel):
    model_name: str
    features: dict[str, Any] | list[dict[str, Any]]


class BlendCandidate(BaseModel):
    rank: int
    materials: dict[str, float]
    metrics: dict[str, Any]
    feasible: bool
    warnings: list[ComputeWarning] = Field(default_factory=list)


class BlendOptimizerResult(BaseModel):
    candidates: list[BlendCandidate]
    best_candidate: BlendCandidate | None = None
    summary: dict[str, Any]
    tables: dict[str, TableData] = Field(default_factory=dict)
    charts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ComputeWarning] = Field(default_factory=list)
    artifacts: list[ComputeArtifact] = Field(default_factory=list)
    computed_at: datetime

class BlendOptimizerCatalogResponse(BaseModel):
    catalog_version: str
    display_timezone: str
    advisory_only: bool = True
    operator_review_required: bool = True
    optimization_modes: list[dict[str, Any]]
    targets: list[str]
    chemistry_fields: list[str]
    material_types: list[str]
    units: dict[str, str]
    precision: dict[str, int]
    iteration_budgets: list[dict[str, Any]]
    capabilities: dict[str, bool]
    limits: dict[str, Any]
    algorithm_versions: dict[str, str]
    model_readiness: dict[str, Any]


class BlendOptimizerContextCreateRequest(BaseModel):
    source_refresh: Literal["use_cached", "refresh"] = "use_cached"
    include_recent_manual_blend: bool = True
    include_diagnostics_summary: bool = True
    chemistry_mode: Literal["latest", "avg"] = "latest"
    chemistry_window_days: int | None = Field(default=None, ge=1, le=365)


class BlendOptimizerContextApiResponse(BaseModel):
    context_id: str
    id: str
    owner_id: str | None = None
    context_version: str
    fingerprint: str
    status: str
    created_at: datetime
    expires_at: datetime | None = None
    as_of_utc: str | None = None
    advisory_only: bool = True
    operator_review_required: bool = True
    eligible_materials: list[dict[str, Any]] = Field(default_factory=list)
    active_pellet_ids: list[str] = Field(default_factory=list)
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    hot_metal_chemistry: dict[str, Any] = Field(default_factory=dict)
    fuel_ash_inputs: list[dict[str, Any]] = Field(default_factory=list)
    flux_inputs: list[dict[str, Any]] = Field(default_factory=list)
    dust_inputs: list[dict[str, Any]] = Field(default_factory=list)
    slag_balance: dict[str, Any] = Field(default_factory=dict)
    recent_fuel_rates: dict[str, Any] = Field(default_factory=dict)
    basicity_defaults: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    model_readiness: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ComputeWarning] = Field(default_factory=list)


class BlendOptimizerDiagnosticsResponse(BaseModel):
    context_id: str
    context_version: str
    diagnostics: dict[str, Any]
    warnings: list[ComputeWarning] = Field(default_factory=list)


class BlendOptimizerPreferencesResponse(BaseModel):
    owner_id: str
    version: int
    preferences: dict[str, Any]
    updated_at: datetime | None = None


class BlendOptimizerPreferencesPatchRequest(BaseModel):
    expected_version: int | None = Field(default=None, ge=0)
    preferences: dict[str, Any]


class BlendOreScenarioInput(BaseModel):
    ore_id: str
    selected: bool = True
    stock_mt: float | None = None
    price_rs_per_mt: float | None = None
    min_share_pct: float | None = None
    max_share_pct: float | None = None
    chemistry: dict[str, float] = Field(default_factory=dict)


class BlendTargetsInput(BaseModel):
    target_hot_metal_mt: float | None = None
    max_slag_mt: float | None = None
    basicity_min: float | None = None
    basicity_max: float | None = None
    t_basicity_min: float | None = None
    t_basicity_max: float | None = None
    feo_in_slag_pct: float | None = None


class BlendScenarioInput(BaseModel):
    targets: BlendTargetsInput = Field(default_factory=BlendTargetsInput)
    ores: list[BlendOreScenarioInput] = Field(default_factory=list)
    fuel_ash_inputs: list[dict[str, Any]] = Field(default_factory=list)
    flux_inputs: list[dict[str, Any]] = Field(default_factory=list)
    dust_inputs: list[dict[str, Any]] = Field(default_factory=list)
    hot_metal_chemistry: dict[str, Any] = Field(default_factory=dict)
    slag_balance: dict[str, Any] = Field(default_factory=dict)
    confirmations: list[str] = Field(default_factory=list)


class BlendRunOptionsInput(BaseModel):
    algorithm_version: str | None = None
    iteration_budget_id: str | None = None
    include_fuel_prediction: bool = True
    include_si_prediction: bool = True
    include_recent_manual_comparison: bool = True
    create_artifacts: list[str] = Field(default_factory=list)


class BlendOptimizerRunCreateRequest(BaseModel):
    mode: Literal["lp_baseline", "total_cost"]
    context_id: str
    expected_context_version: str
    scenario: BlendScenarioInput = Field(default_factory=BlendScenarioInput)
    options: BlendRunOptionsInput = Field(default_factory=BlendRunOptionsInput)


class BlendOptimizerRunResponse(BaseModel):
    run_id: str
    id: str
    owner_id: str | None = None
    mode: Literal["lp_baseline", "total_cost"]
    context_id: str
    context_version: str
    status: str
    progress: float | None = None
    current_step: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    status_path: str
    events_path: str
    result: dict[str, Any] | None = None
    warnings: list[ComputeWarning] = Field(default_factory=list)
    artifacts: list[ComputeArtifact] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    advisory_only: bool = True
    operator_review_required: bool = True


class BlendOptimizerRunEventResponse(BaseModel):
    id: str
    run_id: str
    event_type: str
    sequence: int
    payload: dict[str, Any]
    created_at: datetime


class BlendManualOreInput(BaseModel):
    ore_id: str
    quantity_mt: float
    selected: bool = True
    stock_mt: float | None = None
    price_rs_per_mt: float | None = None
    min_share_pct: float | None = None
    max_share_pct: float | None = None
    chemistry: dict[str, float] = Field(default_factory=dict)


class BlendManualEvaluationRequest(BaseModel):
    ores: list[BlendManualOreInput]


class BlendManualEvaluationResponse(BaseModel):
    run_id: str
    manual_evaluation: dict[str, Any]
    advisory_only: bool = True
    operator_review_required: bool = True
