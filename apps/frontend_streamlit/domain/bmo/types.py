from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OreChemistry:
    fe_t_pct: float
    feo_pct: float = 0.0
    sio2_pct: float = 0.0
    al2o3_pct: float = 0.0
    cao_pct: float = 0.0
    mgo_pct: float = 0.0
    mno_pct: float = 0.0
    tio2_pct: float = 0.0
    p_pct: float = 0.0


@dataclass
class OreInput:
    ore_id: str
    display_name: str
    stock_mt: float
    price_rs_per_mt: float
    min_share_pct: float
    max_share_pct: float
    chemistry: OreChemistry
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlendEvaluation:
    quantities_mt: dict[str, float]
    shares_pct: dict[str, float]
    total_qty_mt: float
    ore_cost_total_rs: float
    ore_cost_per_thm_rs: float
    fuel_cost_per_thm_rs: float
    objective_rs_per_thm: float
    fe_t_pct: float
    effective_fe_pct: float
    fe_production_mt: float
    slag_pct: float
    slag_mt: float
    feasible: bool
    violations: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPrediction:
    value: float
    model_loaded: bool
    scaler_loaded: bool
    used_fallback: bool
    missing_features: list[str] = field(default_factory=list)
    imputed_features: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
