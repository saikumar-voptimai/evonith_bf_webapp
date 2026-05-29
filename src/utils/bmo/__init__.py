"""Public BMO utility API.

This module re-exports the calculation, LP, fuel prediction, nonlinear
optimizer, model service, and shared dataclass objects used by the Streamlit
BMO page and tests.
"""

from utils.bmo.calculations import (
    FE_FROM_FEO_FACTOR,
    compute_dry_fraction,
    compute_dry_weight_mt,
    compute_effective_fe_pct,
    compute_fe_contribution_mt,
    compute_slag_contribution_mt,
    compute_slag_forming_oxides_pct,
    evaluate_blend,
)
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.nonlinear_optimizer import run_nonlinear_optimizer
from utils.bmo.types import BlendEvaluation, ModelPrediction, OreChemistry, OreInput

__all__ = [
    "FE_FROM_FEO_FACTOR",
    "compute_dry_fraction",
    "compute_dry_weight_mt",
    "compute_effective_fe_pct",
    "compute_fe_contribution_mt",
    "compute_slag_contribution_mt",
    "compute_slag_forming_oxides_pct",
    "evaluate_blend",
    "evaluate_blend_with_fuel_prediction",
    "run_lp_baseline",
    "run_nonlinear_optimizer",
    "FuelUnitCostModelService",
    "OreChemistry",
    "OreInput",
    "BlendEvaluation",
    "ModelPrediction",
]
