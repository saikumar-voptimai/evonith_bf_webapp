"""Nonlinear BMO optimizer orchestration.

This module prepares quantity bounds, seeds DE with the LP baseline, evaluates
candidate wet-quantity vectors, and returns the best total-cost blend found by
the differential-evolution runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domain.optimization_runtime import ObjectiveResult, OptimizerRunner
from utils.bmo.constraints import check_blend_constraints, validate_ore_bounds
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.model_service import FuelUnitCostModelService
from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.types import BlendEvaluation, OreInput


def run_nonlinear_optimizer(
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
    model_service: FuelUnitCostModelService,
    process_context: dict[str, float] | None,
    history_df: pd.DataFrame | None,
    de_cfg: dict[str, Any],
) -> tuple[BlendEvaluation | None, list[str]]:
    """
    Run nonlinear total-cost BMO optimization with DE.

    The nonlinear path starts from the feasible LP baseline, then explores wet
    quantity vectors with a fuel-aware objective. If the hard LP constraints are
    infeasible, DE is skipped because there is no reliable feasible seed.

    Args:
         - ores: list[OreInput] - Ores selected for optimization.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
         - process_context: dict[str, float] | None - Latest process variables.
         - history_df: pd.DataFrame | None - Historical process data for lagged features.
         - de_cfg: dict[str, Any] - Differential-evolution and penalty settings.

    Returns:
         - return tuple[BlendEvaluation | None, list[str]] - Best blend and errors.
    """

    pre_errors = validate_ore_bounds(ores)
    if pre_errors:
        return None, pre_errors

    lp_blend, lp_errors = run_lp_baseline(
        ores,
        target_production_mt=target_production_mt,
        target_slag_qty_mt=target_slag_qty_mt,
        feo_in_slag_pct=feo_in_slag_pct,
    )
    if lp_blend is None:
        return None, [
            "Total-cost optimizer skipped because hard LP constraints are infeasible.",
            *lp_errors,
        ]

    bounds = [(0.0, max(0.0, float(ore.stock_mt))) for ore in ores]

    evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_production_mt=float(target_production_mt),
        target_slag_qty_mt=float(target_slag_qty_mt),
        feo_in_slag_pct=float(feo_in_slag_pct),
        model_service=model_service,
        process_context=process_context,
        history_df=history_df,
        penalty_cfg=de_cfg,
    )

    def objective(raw_x: np.ndarray) -> ObjectiveResult:
        """
        Evaluate a DE candidate wet-quantity vector.

        This nested adapter keeps candidate diagnostics attached to each SciPy
        evaluation. The outer runner can then recover the best blend and expose
        the quantities that produced it.

        Args:
             - raw_x: np.ndarray - Candidate wet quantities from differential evolution.

        Returns:
             - return ObjectiveResult - Penalized objective result for the quantities.
        """

        result = evaluator.evaluate_quantities(raw_x)
        result.diagnostics["candidate_quantities_mt"] = np.asarray(
            raw_x, dtype=float
        ).tolist()
        return result

    baseline_solution: dict[str, Any] | None = None
    if lp_blend.total_qty_mt > 0:
        lp_quantities = np.array(
            [float(lp_blend.quantities_mt.get(ore.ore_id, 0.0)) for ore in ores],
            dtype=float,
        )
        baseline_result = objective(lp_quantities)
        baseline_solution = {
            "x": lp_quantities.tolist(),
            "objective": float(baseline_result.objective_value),
            "feasible": bool(baseline_result.feasible),
            "components": dict(baseline_result.components),
            "violations": list(baseline_result.violations),
            "diagnostics": dict(baseline_result.diagnostics),
        }

    runner = OptimizerRunner(de_cfg)
    optimization_result = runner.run_differential_evolution(
        bounds=bounds,
        objective_fn=objective,
        baseline_solution=baseline_solution,
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
        return None, [
            "Nonlinear optimizer failed: could not recover blend diagnostics."
        ]

    violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=target_production_mt,
        target_slag_qty_mt=target_slag_qty_mt,
    )
    blend.feasible = len(violations) == 0
    blend.violations = violations
    blend.diagnostics["de_result"] = optimization_result.diagnostics.get(
        "de_result", {}
    )
    blend.diagnostics["runtime"] = {
        "best_solution": optimization_result.best_solution,
        "compare_metrics": optimization_result.compare_metrics,
    }
    return blend, []
