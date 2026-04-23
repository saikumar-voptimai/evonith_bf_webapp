from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from domain.bmo.calculations import evaluate_blend
from domain.bmo.constraints import check_blend_constraints, validate_ore_bounds
from domain.bmo.feature_builder import build_feature_payload
from domain.bmo.model_service import FuelUnitCostModelService
from domain.bmo.types import BlendEvaluation, OreInput


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

    n = len(ores)
    min_shares = np.array([float(ore.min_share_pct) / 100.0 for ore in ores], dtype=float)
    max_shares = np.array([float(ore.max_share_pct) / 100.0 for ore in ores], dtype=float)
    stocks = np.array([float(ore.stock_mt) for ore in ores], dtype=float)
    bounds = list(zip(min_shares.tolist(), max_shares.tolist()))

    penalty_large = float(de_cfg.get("penalty_large", 1_000_000.0))
    penalty_stock = float(de_cfg.get("penalty_stock", 2500.0))
    penalty_share = float(de_cfg.get("penalty_share", 2500.0))
    penalty_sum = float(de_cfg.get("penalty_sum", 5000.0))
    penalty_fe = float(de_cfg.get("penalty_fe", 3000.0))
    penalty_slag = float(de_cfg.get("penalty_slag", 3000.0))

    ore_name_by_id = {ore.ore_id: ore.display_name for ore in ores}
    best: dict[str, Any] = {"objective": math.inf, "blend": None, "prediction": None}

    def objective(raw_x: np.ndarray) -> float:
        shares = _project_to_bounds_and_sum_one(raw_x, min_shares, max_shares)
        qty = shares * float(target_total_qty_mt)
        quantities = {ore.ore_id: float(qty[idx]) for idx, ore in enumerate(ores)}

        feature_payload = build_feature_payload(
            quantities_mt=quantities,
            ore_display_name_by_id=ore_name_by_id,
            process_context=process_context,
        )
        prediction = model_service.predict(feature_payload, history_df)

        blend = evaluate_blend(
            ores=ores,
            quantities_mt=quantities,
            feo_in_slag_pct=feo_in_slag_pct,
            si_in_slag_pct=si_in_slag_pct,
            fuel_cost_per_thm_rs=float(prediction.value),
        )

        penalties = 0.0

        share_sum = float(np.sum(shares))
        penalties += abs(share_sum - 1.0) * penalty_sum

        stock_violation_mt = float(np.sum(np.clip(qty - stocks, 0.0, None)))
        penalties += stock_violation_mt * penalty_stock

        share_violation = float(
            np.sum(np.clip(min_shares - shares, 0.0, None))
            + np.sum(np.clip(shares - max_shares, 0.0, None))
        )
        penalties += share_violation * penalty_share

        if blend.fe_production_mt < min_fe_production_mt:
            penalties += (min_fe_production_mt - blend.fe_production_mt) * penalty_fe
        if blend.fe_production_mt > max_fe_production_mt:
            penalties += (blend.fe_production_mt - max_fe_production_mt) * penalty_fe
        if blend.slag_mt > target_slag_qty_mt:
            penalties += (blend.slag_mt - target_slag_qty_mt) * penalty_slag

        if not math.isfinite(blend.objective_rs_per_thm):
            penalties += penalty_large

        objective_value = float(blend.objective_rs_per_thm + penalties)
        if objective_value < float(best["objective"]):
            blend.diagnostics["model_prediction"] = prediction.__dict__
            blend.diagnostics["penalties"] = {
                "sum_penalty": abs(share_sum - 1.0) * penalty_sum,
                "stock_penalty": stock_violation_mt * penalty_stock,
                "share_penalty": share_violation * penalty_share,
                "constraint_penalty": penalties,
            }
            best["objective"] = objective_value
            best["blend"] = blend
            best["prediction"] = prediction

        return objective_value

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy=str(de_cfg.get("strategy", "best1bin")),
        maxiter=int(de_cfg.get("maxiter", 40)),
        popsize=int(de_cfg.get("popsize", 12)),
        tol=float(de_cfg.get("tol", 0.01)),
        polish=bool(de_cfg.get("polish", True)),
        seed=int(de_cfg.get("seed", 42)),
        workers=1,
    )

    if best["blend"] is None:
        msg = result.message if hasattr(result, "message") else "DE failed."
        return None, [f"Nonlinear optimizer failed: {msg}"]

    blend: BlendEvaluation = best["blend"]
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
    blend.diagnostics["de_result"] = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
    }
    return blend, []

