"""Public BMO utility API.

This module re-exports the calculation, LP, fuel prediction, nonlinear
optimizer, model service, and shared dataclass objects used by the Streamlit
BMO page and tests.
"""

from utils.bmo.calculations import (
    FE_FROM_FEO_FACTOR,
    compute_charging_requirements,
    compute_dry_fraction,
    compute_dry_weight_mt,
    compute_effective_fe_pct,
    compute_fe_contribution_mt,
    compute_flux_dry_weights_mt,
    compute_flux_slag_contribution_mt,
    compute_flux_slag_contributions_mt,
    compute_fuel_ash_slag_contributions_mt,
    compute_fuel_ash_slag_factor_per_thm,
    compute_fuel_wet_weight_mt,
    compute_slag_contribution_mt,
    compute_slag_forming_oxides_pct,
    evaluate_blend,
)
from utils.bmo.constraints import validate_selected_pellet_inputs
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.nonlinear_optimizer import run_nonlinear_optimizer
from utils.bmo.slag_balance import (
    build_dust_component_totals,
    build_flux_component_totals,
    build_fuel_ash_component_totals,
    build_ore_component_totals,
    calculate_full_slag_balance,
)
from utils.bmo.types import (
    BlendEvaluation,
    DustInput,
    FluxInput,
    FuelAshInput,
    ModelPrediction,
    OreChemistry,
    OreInput,
    SlagBalanceResult,
    SlagBalanceSettings,
)

__all__ = [
    "FE_FROM_FEO_FACTOR",
    "compute_charging_requirements",
    "compute_dry_fraction",
    "compute_dry_weight_mt",
    "compute_effective_fe_pct",
    "compute_fe_contribution_mt",
    "compute_flux_dry_weights_mt",
    "compute_flux_slag_contribution_mt",
    "compute_flux_slag_contributions_mt",
    "compute_fuel_ash_slag_contributions_mt",
    "compute_fuel_ash_slag_factor_per_thm",
    "compute_fuel_wet_weight_mt",
    "compute_slag_contribution_mt",
    "compute_slag_forming_oxides_pct",
    "evaluate_blend",
    "validate_selected_pellet_inputs",
    "evaluate_blend_with_fuel_prediction",
    "run_lp_baseline",
    "run_nonlinear_optimizer",
    "FuelUnitCostModelService",
    "build_ore_component_totals",
    "build_flux_component_totals",
    "build_fuel_ash_component_totals",
    "build_dust_component_totals",
    "calculate_full_slag_balance",
    "OreChemistry",
    "OreInput",
    "BlendEvaluation",
    "DustInput",
    "FluxInput",
    "FuelAshInput",
    "SlagBalanceSettings",
    "SlagBalanceResult",
    "ModelPrediction",
]
