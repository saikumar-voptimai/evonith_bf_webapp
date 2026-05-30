"""Typed data contracts for the BMO optimization workflow.

This module defines the ore chemistry, ore input, blend evaluation, and model
prediction structures shared by the BMO page, solvers, model service, and UI
components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OreChemistry:
    """
    Store chemistry values for one ore or burden material.

    Fe, oxide, alkali, and moisture fields are kept together because blend
    evaluation must convert wet quantity to dry quantity before applying
    chemistry percentages. Defaults are intentionally zero so partial fallback
    mappings remain valid.

    Args:
         - fe_t_pct: float - Total iron percentage on dry basis.
         - moisture_pct: float - Moisture or total moisture percentage.
         - feo_pct: float - FeO percentage on dry basis.
         - sio2_pct: float - SiO2 percentage on dry basis.
         - al2o3_pct: float - Al2O3 percentage on dry basis.
         - cao_pct: float - CaO percentage on dry basis.
         - mgo_pct: float - MgO percentage on dry basis.
         - mno_pct: float - MnO percentage on dry basis.
         - tio2_pct: float - TiO2 percentage on dry basis.
         - p_pct: float - Phosphorus percentage on dry basis.
         - na2o_pct: float - Na2O percentage on dry basis.
         - k2o_pct: float - K2O percentage on dry basis.

    Returns:
         - return OreChemistry - Chemistry record used by blend calculations.
    """

    fe_t_pct: float
    moisture_pct: float = 0.0
    feo_pct: float = 0.0
    sio2_pct: float = 0.0
    al2o3_pct: float = 0.0
    cao_pct: float = 0.0
    mgo_pct: float = 0.0
    mno_pct: float = 0.0
    tio2_pct: float = 0.0
    p_pct: float = 0.0
    na2o_pct: float = 0.0
    k2o_pct: float = 0.0


@dataclass
class OreInput:
    """
    Store optimizer input data for one selectable ore.

    Quantities, stock, and share limits are treated as wet-basis planning values.
    The nested chemistry record supplies the moisture needed to convert those
    wet quantities into dry weights for Fe and slag calculations.

    Args:
         - ore_id: str - Stable ore identifier used in quantity maps.
         - display_name: str - Human-readable ore name shown in the UI.
         - stock_mt: float - Available wet stock quantity in MT.
         - price_rs_per_mt: float - Ore price in Rs/MT.
         - min_share_pct: float - Minimum wet burden share percentage.
         - max_share_pct: float - Maximum wet burden share percentage.
         - chemistry: OreChemistry - Chemistry and moisture values for the ore.
         - metadata: dict[str, Any] - Source mapping metadata for diagnostics.

    Returns:
         - return OreInput - Typed ore input consumed by LP and DE solvers.
    """

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
    """
    Store calculated metrics for one evaluated BMO blend.

    This object carries both planning values, such as wet quantities and ore
    cost, and chemistry values, such as dry-weight Fe production. Diagnostics
    include per-ore dry weights and Fe contributions for review.

    Args:
         - quantities_mt: dict[str, float] - Wet ore quantities keyed by ore id.
         - shares_pct: dict[str, float] - Wet burden shares keyed by ore id.
         - total_qty_mt: float - Total wet burden quantity in MT.
         - ore_cost_total_rs: float - Total wet ore purchase cost in Rs.
         - ore_cost_per_thm_rs: float - Ore cost divided by Fe production.
         - fuel_cost_per_thm_rs: float - Predicted or fallback fuel cost in Rs/THM.
         - objective_rs_per_thm: float - Ore plus fuel objective in Rs/THM.
         - fe_t_pct: float - Final dry-weight Fe percentage.
         - effective_fe_pct: float - FeO-adjusted Fe percentage retained for diagnostics.
         - fe_production_mt: float - Final Fe contribution from dry weights in MT.
         - slag_pct: float - Dry-weight slag percentage estimate.
         - slag_mt: float - Slag quantity estimate in MT.
         - feasible: bool - Whether the blend satisfies hard constraints.
         - violations: list[str] - Human-readable constraint violations.
         - slag_rate_kg_per_thm: float - Slag rate against the app's THM denominator.
         - diagnostics: dict[str, Any] - Additional calculation and solver details.

    Returns:
         - return BlendEvaluation - Evaluated blend metrics and diagnostics.
    """

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
    slag_rate_kg_per_thm: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPrediction:
    """
    Store BMO fuel-cost model prediction output.

    Prediction records are attached to blend diagnostics so the page can show
    whether the ML model or deterministic fallback produced the fuel-cost term.

    Args:
         - value: float - Predicted or fallback fuel cost in Rs/THM.
         - model_loaded: bool - Whether the model artifact was loaded.
         - scaler_loaded: bool - Whether the scaler artifact was loaded.
         - used_fallback: bool - Whether fallback logic was used for the value.
         - missing_features: list[str] - Feature names not resolved before defaults.
         - imputed_features: list[str] - Feature names populated from defaults.
         - details: dict[str, Any] - Additional inference diagnostics.

    Returns:
         - return ModelPrediction - Fuel-cost prediction record.
    """

    value: float
    model_loaded: bool
    scaler_loaded: bool
    used_fallback: bool
    missing_features: list[str] = field(default_factory=list)
    imputed_features: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
