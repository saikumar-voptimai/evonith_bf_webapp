"""Nonlinear BMO optimizer orchestration.

This module prepares share bounds, seeds DE around the LP baseline, evaluates
candidate wet-share vectors, and returns the best total-cost blend found by
the differential-evolution runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from furnace_data.optimization_runtime import ObjectiveResult, OptimizerRunner
from furnace_data.bmo.utils.constraints import check_blend_constraints, validate_ore_bounds
from furnace_data.bmo.utils.lp_solver import run_lp_baseline
from furnace_data.bmo.utils.model_service import FuelUnitCostModelService
from furnace_data.bmo.utils.objective import BmoObjectiveEvaluator
from furnace_data.bmo.utils.types import (
    BlendEvaluation,
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)

def _project_shares(
    raw: np.ndarray, min_shares: np.ndarray, max_shares: np.ndarray
) -> np.ndarray:
    shares = np.clip(np.asarray(raw, dtype=float), min_shares, max_shares)
    for _ in range(20):
        diff = 1.0 - float(np.sum(shares))
        if abs(diff) <= 1e-10:
            break
        if diff > 0:
            capacity = np.clip(max_shares - shares, 0.0, None)
        else:
            capacity = np.clip(shares - min_shares, 0.0, None)
        cap_sum = float(np.sum(capacity))
        if cap_sum <= 1e-12:
            break
        shares += capacity / cap_sum * diff
        shares = np.clip(shares, min_shares, max_shares)
    return shares


def _quantities_from_shares(
    *,
    shares: np.ndarray,
    ores: list[OreInput],
    target_fe_mt: float,
) -> np.ndarray:
    fe_per_wet_mt = np.array(
        [
            max(0.0, 1.0 - float(ore.chemistry.moisture_pct) / 100.0)
            * max(0.0, float(ore.chemistry.fe_t_pct))
            / 100.0
            for ore in ores
        ],
        dtype=float,
    )
    fe_per_blend_mt = float(np.dot(shares, fe_per_wet_mt))
    if fe_per_blend_mt <= 0.0:
        return np.zeros(len(ores), dtype=float)
    total_wet_mt = float(target_fe_mt) / fe_per_blend_mt
    return shares * total_wet_mt


def _build_initial_share_population(
    *,
    lp_shares: np.ndarray,
    min_shares: np.ndarray,
    max_shares: np.ndarray,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    population: list[np.ndarray] = [
        _project_shares(lp_shares, min_shares, max_shares)
    ]
    scales = [0.01, 0.025, 0.05, 0.075, 0.10]
    while len(population) < sample_count:
        scale = scales[(len(population) - 1) % len(scales)]
        noise = rng.normal(0.0, scale, size=lp_shares.shape)
        population.append(_project_shares(lp_shares + noise, min_shares, max_shares))
    return np.vstack(population)


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
    target_slag_basicity_min: float | None = None,
    target_slag_basicity_max: float | None = None,
    target_slag_t_basicity_min: float | None = None,
    target_slag_t_basicity_max: float | None = None,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    hot_metal_target_mt: float | None = None,
    progress_callback: (
        Callable[[int, float, float | None, int, float], bool] | None
    ) = None,
    precomputed_lp_blend: BlendEvaluation | None = None,
    precomputed_lp_errors: list[str] | None = None,
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
         - target_slag_basicity_min: float | None - Minimum CaO / SiO2 basicity.
         - target_slag_basicity_max: float | None - Maximum CaO / SiO2 basicity.
         - target_slag_t_basicity_min: float | None - Minimum (CaO + MgO) / SiO2 basicity.
         - target_slag_t_basicity_max: float | None - Maximum (CaO + MgO) / SiO2 basicity.
         - model_service: FuelUnitCostModelService - Fuel-cost prediction service.
         - process_context: dict[str, float] | None - Latest process variables.
         - history_df: pd.DataFrame | None - Historical process data for lagged features.
         - de_cfg: dict[str, Any] - Differential-evolution and penalty settings.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

    Returns:
         - return tuple[BlendEvaluation | None, list[str]] - Best blend and errors.
    """

    pre_errors = validate_ore_bounds(ores)
    if pre_errors:
        return None, pre_errors

    if precomputed_lp_blend is not None:
        lp_blend = precomputed_lp_blend
        lp_errors = list(precomputed_lp_errors or [])
    else:
        lp_blend, lp_errors = run_lp_baseline(
            ores,
            target_production_mt=target_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            feo_in_slag_pct=feo_in_slag_pct,
            target_slag_basicity_min=target_slag_basicity_min,
            target_slag_basicity_max=target_slag_basicity_max,
            target_slag_t_basicity_min=target_slag_t_basicity_min,
            target_slag_t_basicity_max=target_slag_t_basicity_max,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            hot_metal_target_mt=hot_metal_target_mt,
        )
    if lp_blend is None:
        return None, [
            "Total-cost optimizer skipped because hard LP constraints are infeasible.",
            *lp_errors,
        ]

    min_shares = np.array(
        [float(ore.min_share_pct) / 100.0 for ore in ores], dtype=float
    )
    max_shares = np.array(
        [float(ore.max_share_pct) / 100.0 for ore in ores], dtype=float
    )
    bounds = [(float(lo), float(hi)) for lo, hi in zip(min_shares, max_shares)]

    prebuilt_context = model_service.build_prebuilt_context(
        ores=ores,
        process_context=process_context,
        history_df=history_df,
        hot_metal_target_mt=hot_metal_target_mt,
    )

    evaluator = BmoObjectiveEvaluator(
        ores=ores,
        target_production_mt=float(target_production_mt),
        target_slag_qty_mt=float(target_slag_qty_mt),
        feo_in_slag_pct=float(feo_in_slag_pct),
        target_slag_basicity_min=target_slag_basicity_min,
        target_slag_basicity_max=target_slag_basicity_max,
        target_slag_t_basicity_min=target_slag_t_basicity_min,
        target_slag_t_basicity_max=target_slag_t_basicity_max,
        model_service=model_service,
        process_context=process_context,
        history_df=history_df,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        penalty_cfg=de_cfg,
        prebuilt_context=prebuilt_context,
        hot_metal_target_mt=hot_metal_target_mt,
    )

    def objective(raw_x: np.ndarray) -> ObjectiveResult:
        """
        Evaluate a DE candidate wet-share vector.

        This nested adapter keeps candidate diagnostics attached to each SciPy
        evaluation. The outer runner can then recover the best blend and expose
        the quantities that produced it.

        Args:
             - raw_x: np.ndarray - Candidate shares from differential evolution.

        Returns:
             - return ObjectiveResult - Penalized objective result for the quantities.
        """

        shares = _project_shares(raw_x, min_shares, max_shares)
        quantities = _quantities_from_shares(
            shares=shares,
            ores=ores,
            target_fe_mt=float(target_production_mt),
        )
        result = evaluator.evaluate_quantities(quantities)
        result.diagnostics["candidate_shares_pct"] = (shares * 100.0).tolist()
        result.diagnostics["raw_candidate_shares_pct"] = (
            np.asarray(raw_x, dtype=float) * 100.0
        ).tolist()
        result.diagnostics["candidate_quantities_mt"] = np.asarray(
            quantities, dtype=float
        ).tolist()
        return result

    baseline_solution: dict[str, Any] | None = None
    initial_population: np.ndarray | None = None
    if lp_blend.total_qty_mt > 0:
        lp_quantities = np.array(
            [float(lp_blend.quantities_mt.get(ore.ore_id, 0.0)) for ore in ores],
            dtype=float,
        )
        lp_shares = lp_quantities / float(np.sum(lp_quantities))
        baseline_result = objective(lp_shares)
        baseline_solution = {
            "x": lp_shares.tolist(),
            "objective": float(baseline_result.objective_value),
            "feasible": bool(baseline_result.feasible),
            "components": dict(baseline_result.components),
            "violations": list(baseline_result.violations),
            "diagnostics": dict(baseline_result.diagnostics),
        }
        n_vars = len(ores)
        min_samples = max(5, int(de_cfg.get("popsize", 10)) * n_vars)
        sample_count = int(de_cfg.get("initial_population_samples", min_samples))
        initial_population = _build_initial_share_population(
            lp_shares=lp_shares,
            min_shares=min_shares,
            max_shares=max_shares,
            sample_count=max(min_samples, sample_count),
            seed=int(de_cfg.get("seed", 42)),
        )

    runner = OptimizerRunner(de_cfg)
    optimization_result = runner.run_differential_evolution(
        bounds=bounds,
        objective_fn=objective,
        baseline_solution=baseline_solution,
        initial_population=initial_population,
        progress_callback=progress_callback,
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
        target_slag_basicity_min=target_slag_basicity_min,
        target_slag_basicity_max=target_slag_basicity_max,
        target_slag_t_basicity_min=target_slag_t_basicity_min,
        target_slag_t_basicity_max=target_slag_t_basicity_max,
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
