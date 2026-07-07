from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class OptimizationContext:
    history_df: pd.DataFrame
    latest_row: pd.Series
    process_context: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBundleInfo:
    model: Any | None
    scaler: Any | None
    expected_features: list[str]
    lag_map: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, float] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    feature_manifest: dict[str, Any] = field(default_factory=dict)
    training_metrics: dict[str, Any] = field(default_factory=dict)
    target_name: str | None = None


@dataclass
class FeatureBuildResult:
    vector_df: pd.DataFrame
    missing_features: list[str] = field(default_factory=list)
    imputed_features: list[str] = field(default_factory=list)
    source_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ObjectiveResult:
    objective_value: float
    components: dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    best_solution: dict[str, Any]
    baseline_solution: dict[str, Any] | None
    compare_metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
