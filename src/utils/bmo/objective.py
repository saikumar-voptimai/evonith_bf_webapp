"""Objective evaluation for BMO nonlinear total-cost optimization.

This module converts DE candidate wet-quantity vectors into blend evaluations,
predicts fuel cost for each candidate, applies constraint penalties, and returns
rich objective diagnostics to the shared optimization runtime.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from domain.optimization_runtime import ObjectiveResult
from utils.bmo.constraints import check_blend_constraints
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)


class BmoObjectiveEvaluator:
    """
    Evaluate BMO candidate quantities as total-cost objective values.

    DE proposes wet ore quantity vectors. This evaluator converts each vector
    into a full blend evaluation, asks the fuel model for Rs/THM, and applies
    soft penalties so infeasible candidates can still guide the search.

    Args:
         - ores: list[OreInput] - Ores available to the optimizer.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
         - process_context: dict[str, float] | None - Latest process variables.
         - history_df: pd.DataFrame | None - Historical process data for lagged features.
         - penalty_cfg: dict[str, Any] - Penalty weights used for soft constraints.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

    Returns:
         - return BmoObjectiveEvaluator - Configured evaluator for quantity vectors.
    """

    def __init__(
        self,
        *,
        ores: list[OreInput],
        target_production_mt: float,
        target_slag_qty_mt: float,
        feo_in_slag_pct: float,
        model_service: FuelUnitCostModelService,
        process_context: dict[str, float] | None,
        history_df: pd.DataFrame | None,
        penalty_cfg: dict[str, Any],
        fuel_ash_inputs: list[FuelAshInput] | None = None,
        flux_inputs: list[FluxInput] | None = None,
        dust_inputs: list[DustInput] | None = None,
        slag_balance_settings: SlagBalanceSettings | None = None,
    ) -> None:
        """
        Store optimizer inputs and precompute array forms of bounds.

        These cached arrays keep each DE objective call lightweight. Constraint
        targets and penalty settings are also normalized to floats so candidate
        evaluation is deterministic across Streamlit reruns and tests.

        Args:
             - ores: list[OreInput] - Ores available to the optimizer.
             - target_production_mt: float - Target hot-metal production in MT.
             - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
             - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
             - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
             - process_context: dict[str, float] | None - Latest process variables.
             - history_df: pd.DataFrame | None - Historical process data for lagged features.
             - penalty_cfg: dict[str, Any] - Penalty weights used for soft constraints.
             - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
             - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
             - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
             - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

        Returns:
             - return None - Initializes evaluator state.
        """

        self.ores = ores
        self.target_production_mt = float(target_production_mt)
        self.target_slag_qty_mt = float(target_slag_qty_mt)
        self.feo_in_slag_pct = float(feo_in_slag_pct)
        self.model_service = model_service
        self.process_context = process_context or {}
        self.history_df = history_df
        self.fuel_ash_inputs = fuel_ash_inputs
        self.flux_inputs = flux_inputs
        self.dust_inputs = dust_inputs
        self.slag_balance_settings = slag_balance_settings
        self.penalty_cfg = penalty_cfg
        self.stocks = np.array([float(ore.stock_mt) for ore in ores], dtype=float)
        self.min_shares = np.array(
            [float(ore.min_share_pct) / 100.0 for ore in ores], dtype=float
        )
        self.max_shares = np.array(
            [float(ore.max_share_pct) / 100.0 for ore in ores], dtype=float
        )

    def evaluate_quantities(self, quantities_vector: np.ndarray) -> ObjectiveResult:
        """
        Evaluate one candidate wet-quantity vector for DE optimization.

        The returned objective contains both the scalar value required by SciPy
        and rich diagnostics for the Streamlit comparison view. Production
        shortfall, production excess, slag, stock, and share violations are
        penalized but also reported explicitly. Penalizing production excess
        keeps DE anchored to the operator's target hot metal without requiring
        a separate maximum-production UI input.

        Args:
             - quantities_vector: np.ndarray - Candidate wet ore quantities in MT.

        Returns:
             - return ObjectiveResult - Penalized objective, components, and diagnostics.
        """

        qty = np.asarray(quantities_vector, dtype=float)
        quantities = {ore.ore_id: float(qty[idx]) for idx, ore in enumerate(self.ores)}

        blend = evaluate_blend_with_fuel_prediction(
            ores=self.ores,
            quantities_mt=quantities,
            feo_in_slag_pct=self.feo_in_slag_pct,
            model_service=self.model_service,
            process_context=self.process_context,
            history_df=self.history_df,
            fuel_ash_inputs=self.fuel_ash_inputs,
            flux_inputs=self.flux_inputs,
            dust_inputs=self.dust_inputs,
            slag_balance_settings=self.slag_balance_settings,
        )

        penalty_stock = float(self.penalty_cfg.get("penalty_stock", 2500.0))
        penalty_share = float(self.penalty_cfg.get("penalty_share", 2500.0))
        penalty_fe = float(self.penalty_cfg.get("penalty_fe", 3000.0))
        penalty_production_excess = float(
            self.penalty_cfg.get("penalty_production_excess", penalty_fe)
        )
        penalty_slag = float(self.penalty_cfg.get("penalty_slag", 3000.0))
        penalty_large = float(self.penalty_cfg.get("penalty_large", 1_000_000.0))

        stock_violation_mt = float(np.sum(np.clip(qty - self.stocks, 0.0, None)))
        stock_penalty = stock_violation_mt * penalty_stock

        shares = (
            qty / float(blend.total_qty_mt)
            if blend.total_qty_mt > 0
            else np.zeros_like(qty)
        )
        share_violation = float(
            np.sum(np.clip(self.min_shares - shares, 0.0, None))
            + np.sum(np.clip(shares - self.max_shares, 0.0, None))
        )
        share_penalty = share_violation * penalty_share

        fe_penalty = 0.0
        if blend.fe_production_mt < self.target_production_mt:
            fe_penalty += (
                self.target_production_mt - blend.fe_production_mt
            ) * penalty_fe
        production_excess_penalty = 0.0
        if blend.fe_production_mt > self.target_production_mt:
            production_excess_penalty += (
                blend.fe_production_mt - self.target_production_mt
            ) * penalty_production_excess

        slag_penalty = 0.0
        if blend.slag_mt > self.target_slag_qty_mt:
            slag_penalty += (blend.slag_mt - self.target_slag_qty_mt) * penalty_slag

        finite_penalty = 0.0
        if not math.isfinite(blend.objective_rs_per_thm):
            finite_penalty = penalty_large
        if blend.total_qty_mt <= 0.0:
            finite_penalty += penalty_large

        total_penalty = (
            stock_penalty
            + share_penalty
            + fe_penalty
            + production_excess_penalty
            + slag_penalty
            + finite_penalty
        )
        objective_value = float(blend.objective_rs_per_thm + total_penalty)

        violations = check_blend_constraints(
            blend,
            self.ores,
            target_production_mt=self.target_production_mt,
            target_slag_qty_mt=self.target_slag_qty_mt,
        )
        feasible = len(violations) == 0

        return ObjectiveResult(
            objective_value=objective_value,
            components={
                "ore_cost_per_thm_rs": float(blend.ore_cost_per_thm_rs),
                "fuel_cost_per_thm_rs": float(blend.fuel_cost_per_thm_rs),
                "base_objective_rs_per_thm": float(blend.objective_rs_per_thm),
                "penalty_total": float(total_penalty),
                "penalty_stock": float(stock_penalty),
                "penalty_share_bounds": float(share_penalty),
                "penalty_fe": float(fe_penalty),
                "penalty_production_excess": float(production_excess_penalty),
                "penalty_slag": float(slag_penalty),
                "penalty_non_finite": float(finite_penalty),
            },
            feasible=feasible,
            violations=violations,
            diagnostics={
                "blend": blend,
                "model_prediction": blend.diagnostics.get("model_prediction"),
                "feature_details": blend.diagnostics.get("feature_details", {}),
            },
        )
