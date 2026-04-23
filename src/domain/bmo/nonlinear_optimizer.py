from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domain.bmo.constraints import check_blend_constraints, validate_ore_bounds
from domain.bmo.model_service import FuelUnitCostModelService
from domain.bmo.objective import BmoObjectiveEvaluator
from domain.bmo.types import BlendEvaluation, OreInput
from domain.optimization_runtime import ObjectiveResult, OptimizerRunner


def _project_to_bounds_and_sum_one(
    raw_shares: np.ndarray, min_shares: np.ndarray, max_shares: np.ndarray
) -> np.ndarray:
    x = np.clip(raw_shares.astype(float), min_shares, max_shares)

    for _ in range(8):
        total = float(np.sum(x))
        if abs(total - 1.0) <= 1e-8:
            break

        if total < 1.0:
            deficit = 1.0 - total
            room = np.clip(max_shares - x, 0.0, None)
            room_sum = float(np.sum(room))
            if room_sum <= 1e-10:
                break
            x = x + (room / room_sum) * deficit
        else:
            excess = total - 1.0
            slack = np.clip(x - min_shares, 0.0, None)
            slack_sum = float(np.sum(slack))
            if slack_sum <= 1e-10:
                break
            x = x - (slack / slack_sum) * excess

        x = np.clip(x, min_shares, max_shares)

    total = float(np.sum(x))
    if total > 0:
        x = x / total
        x = np.clip(x, min_shares, max_shares)
    return x


def run_nonlinear_optimizer(
    ores: list[OreInput],
    *,
    target_total_qty_mt: float,
    min_fe_production_mt: float,
    max_fe_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
    si_in_slag_pct: float,
    model_service: FuelUnitCostModelService,
    process_context: dict[str, float] | None,
    history_df: pd.DataFrame | None,
    de_cfg: dict[str, Any],
) -> tuple[BlendEvaluation | None, list[str]]:
    pre_errors = validate_ore_bounds(ores, target_total_qty_mt)
    if pre_errors:
        return None, pre_errors

    min_shares = np.array([float(ore.min_share_pct) / 100.0 for ore in ores], dtype=float)
    max_shares = np.array([float(ore.max_share_pct) / 100.0 for ore in ores], dtype=float)
    bounds = list(zip(min_shares.tolist(), max_shares.tolist()))

    evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_total_qty_mt=float(target_total_qty_mt),
        min_fe_production_mt=float(min_fe_production_mt),
        max_fe_production_mt=float(max_fe_production_mt),
        target_slag_qty_mt=float(target_slag_qty_mt),
        feo_in_slag_pct=float(feo_in_slag_pct),
        si_in_slag_pct=float(si_in_slag_pct),
        model_service=model_service,
        process_context=process_context,
        history_df=history_df,
        penalty_cfg=de_cfg,
    )

    def objective(raw_x: np.ndarray) -> ObjectiveResult:
        shares = _project_to_bounds_and_sum_one(raw_x, min_shares, max_shares)
        result = evaluator.evaluate_shares(shares)
        result.diagnostics["projected_shares"] = shares.tolist()
        return result

    runner = OptimizerRunner(de_cfg)
    optimization_result = runner.run_differential_evolution(
        bounds=bounds,
        objective_fn=objective,
    )

    best_diag = dict(optimization_result.best_solution.get("diagnostics", {}))
    blend = best_diag.get("blend")
    if blend is None:
        msg = (
            optimization_result.diagnostics.get("de_result", {}).get("message")
            or "DE failed."
        )
        return None, [f"Nonlinear optimizer failed: {msg}"]

    blend = blend if isinstance(blend, BlendEvaluation) else None
    if blend is None:
        return None, ["Nonlinear optimizer failed: could not recover blend diagnostics."]

    violations = check_blend_constraints(
        blend,
        ores,
        target_total_qty_mt=target_total_qty_mt,
        min_fe_production_mt=min_fe_production_mt,
        max_fe_production_mt=max_fe_production_mt,
        target_slag_qty_mt=target_slag_qty_mt,
    )
    blend.feasible = len(violations) == 0
    blend.violations = violations
    blend.diagnostics["de_result"] = optimization_result.diagnostics.get("de_result", {})
    blend.diagnostics["runtime"] = {
        "best_solution": optimization_result.best_solution,
        "compare_metrics": optimization_result.compare_metrics,
    }
    return blend, []
