"""Blend Optimizer API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.api.v1.schemas.compute_common import ComputeArtifact, ComputeWarning, TableData


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
