from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from furnace_data.bmo.domain.calculations import (
    compute_effective_fe_pct,
    compute_slag_pct,
    evaluate_blend,
)
from furnace_data.bmo.domain.constraints import check_blend_constraints, validate_ore_bounds
from furnace_data.bmo.domain.types import BlendEvaluation, OreInput


def run_lp_baseline(
    ores: list[OreInput],
    *,
    target_total_qty_mt: float,
    min_fe_production_mt: float,
    max_fe_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
    si_in_slag_pct: float,
) -> tuple[BlendEvaluation | None, list[str]]:
    pre_errors = validate_ore_bounds(ores, target_total_qty_mt)
    if pre_errors:
        return None, pre_errors

    n = len(ores)
    c = np.array([float(ore.price_rs_per_mt) for ore in ores], dtype=float)

    fe_coeff = np.array(
        [
            compute_effective_fe_pct(
                ore.chemistry.fe_t_pct, ore.chemistry.feo_pct, feo_in_slag_pct
            )
            / 100.0
            for ore in ores
        ],
        dtype=float,
    )
    slag_coeff = np.array(
        [
            compute_slag_pct(
                sio2_pct=ore.chemistry.sio2_pct,
                al2o3_pct=ore.chemistry.al2o3_pct,
                cao_pct=ore.chemistry.cao_pct,
                mgo_pct=ore.chemistry.mgo_pct,
                mno_pct=ore.chemistry.mno_pct,
                si_in_slag_pct=si_in_slag_pct,
            )
            / 100.0
            for ore in ores
        ],
        dtype=float,
    )

    A_ub = np.vstack(
        [
            -fe_coeff,  # Fe >= min
            fe_coeff,  # Fe <= max
            slag_coeff,  # Slag <= target
        ]
    )
    b_ub = np.array(
        [
            -float(min_fe_production_mt),
            float(max_fe_production_mt),
            float(target_slag_qty_mt),
        ]
    )

    A_eq = np.ones((1, n), dtype=float)
    b_eq = np.array([float(target_total_qty_mt)], dtype=float)

    bounds: list[tuple[float, float]] = []
    for ore in ores:
        min_qty = (float(ore.min_share_pct) / 100.0) * float(target_total_qty_mt)
        max_qty = min(
            (float(ore.max_share_pct) / 100.0) * float(target_total_qty_mt),
            float(ore.stock_mt),
        )
        bounds.append((min_qty, max_qty))

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success or result.x is None:
        err = result.message or "LP solver failed."
        return None, [f"LP infeasible or failed: {err}"]

    quantities = {ore.ore_id: float(result.x[idx]) for idx, ore in enumerate(ores)}
    blend = evaluate_blend(
        ores=ores,
        quantities_mt=quantities,
        feo_in_slag_pct=feo_in_slag_pct,
        si_in_slag_pct=si_in_slag_pct,
        fuel_cost_per_thm_rs=0.0,
    )
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
    return blend, []
