from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from domain.bmo.calculations import evaluate_blend
from domain.bmo.constraints import check_blend_constraints
from domain.bmo.feature_builder import build_feature_payload
from domain.bmo.model_service import FuelUnitCostModelService
from domain.bmo.types import OreInput
from domain.optimization_runtime import ObjectiveResult


class BmoObjectiveEvaluator:
    def __init__(
        self,
        *,
        ores: list[OreInput],
        target_total_qty_mt: float,
        min_fe_production_mt: float,
        max_fe_production_mt: float,
        target_slag_qty_mt: float,
        feo_in_slag_pct: float,
        si_in_slag_pct: float,
        model_service: FuelUnitCostModelService,
        process_context: dict[str, float] | None,
        history_df: pd.DataFrame | None,
        penalty_cfg: dict[str, Any],
    ) -> None:
        self.ores = ores
        self.target_total_qty_mt = float(target_total_qty_mt)
        self.min_fe_production_mt = float(min_fe_production_mt)
        self.max_fe_production_mt = float(max_fe_production_mt)
        self.target_slag_qty_mt = float(target_slag_qty_mt)
        self.feo_in_slag_pct = float(feo_in_slag_pct)
        self.si_in_slag_pct = float(si_in_slag_pct)
        self.model_service = model_service
        self.process_context = process_context or {}
        self.history_df = history_df
        self.penalty_cfg = penalty_cfg
        self.ore_name_by_id = {ore.ore_id: ore.display_name for ore in ores}
        self.stocks = np.array([float(ore.stock_mt) for ore in ores], dtype=float)
        self.min_shares = np.array(
            [float(ore.min_share_pct) / 100.0 for ore in ores], dtype=float
        )
        self.max_shares = np.array(
            [float(ore.max_share_pct) / 100.0 for ore in ores], dtype=float
        )

    def evaluate_shares(self, shares: np.ndarray) -> ObjectiveResult:
        qty = shares * self.target_total_qty_mt
        quantities = {ore.ore_id: float(qty[idx]) for idx, ore in enumerate(self.ores)}

        feature_payload = build_feature_payload(
            quantities_mt=quantities,
            ore_display_name_by_id=self.ore_name_by_id,
            process_context=self.process_context,
        )
        prediction = self.model_service.predict(feature_payload, self.history_df)

        blend = evaluate_blend(
            ores=self.ores,
            quantities_mt=quantities,
            feo_in_slag_pct=self.feo_in_slag_pct,
            si_in_slag_pct=self.si_in_slag_pct,
            fuel_cost_per_thm_rs=float(prediction.value),
        )

        penalty_stock = float(self.penalty_cfg.get("penalty_stock", 2500.0))
        penalty_share = float(self.penalty_cfg.get("penalty_share", 2500.0))
        penalty_sum = float(self.penalty_cfg.get("penalty_sum", 5000.0))
        penalty_fe = float(self.penalty_cfg.get("penalty_fe", 3000.0))
        penalty_slag = float(self.penalty_cfg.get("penalty_slag", 3000.0))
        penalty_large = float(self.penalty_cfg.get("penalty_large", 1_000_000.0))

        share_sum = float(np.sum(shares))
        share_sum_penalty = abs(share_sum - 1.0) * penalty_sum

        stock_violation_mt = float(np.sum(np.clip(qty - self.stocks, 0.0, None)))
        stock_penalty = stock_violation_mt * penalty_stock

        share_violation = float(
            np.sum(np.clip(self.min_shares - shares, 0.0, None))
            + np.sum(np.clip(shares - self.max_shares, 0.0, None))
        )
        share_penalty = share_violation * penalty_share

        fe_penalty = 0.0
        if blend.fe_production_mt < self.min_fe_production_mt:
            fe_penalty += (
                self.min_fe_production_mt - blend.fe_production_mt
            ) * penalty_fe
        if blend.fe_production_mt > self.max_fe_production_mt:
            fe_penalty += (
                blend.fe_production_mt - self.max_fe_production_mt
            ) * penalty_fe

        slag_penalty = 0.0
        if blend.slag_mt > self.target_slag_qty_mt:
            slag_penalty += (blend.slag_mt - self.target_slag_qty_mt) * penalty_slag

        finite_penalty = 0.0
        if not math.isfinite(blend.objective_rs_per_thm):
            finite_penalty = penalty_large

        total_penalty = (
            share_sum_penalty
            + stock_penalty
            + share_penalty
            + fe_penalty
            + slag_penalty
            + finite_penalty
        )
        objective_value = float(blend.objective_rs_per_thm + total_penalty)

        violations = check_blend_constraints(
            blend,
            self.ores,
            target_total_qty_mt=self.target_total_qty_mt,
            min_fe_production_mt=self.min_fe_production_mt,
            max_fe_production_mt=self.max_fe_production_mt,
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
                "penalty_share_sum": float(share_sum_penalty),
                "penalty_stock": float(stock_penalty),
                "penalty_share_bounds": float(share_penalty),
                "penalty_fe": float(fe_penalty),
                "penalty_slag": float(slag_penalty),
                "penalty_non_finite": float(finite_penalty),
            },
            feasible=feasible,
            violations=violations,
            diagnostics={
                "blend": blend,
                "model_prediction": prediction,
                "feature_details": prediction.details,
            },
        )
