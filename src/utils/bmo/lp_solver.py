"""LP baseline solver for BMO wet burden planning.

This module builds the deterministic ore-cost LP baseline using wet quantity
bounds while applying dry-weight Fe production and final slag as hard process
targets. If no blend can meet those physical limits, LP returns infeasible
instead of showing a low-cost blend that violates the target slag cap.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from utils.bmo.calculations import compute_dry_fraction, evaluate_blend
from utils.bmo.constraints import check_blend_constraints, validate_ore_bounds
from utils.bmo.types import (
    BlendEvaluation,
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)

FE_TOLERANCE_MT = 0.5


def _build_linear_slag_terms(
    ores: list[OreInput],
    *,
    feo_in_slag_pct: float,
    fuel_ash_inputs: list[FuelAshInput] | None,
    flux_inputs: list[FluxInput] | None,
    dust_inputs: list[DustInput] | None,
    slag_balance_settings: SlagBalanceSettings | None,
) -> tuple[np.ndarray, float]:
    """
    Estimate linear final-slag terms for LP hard slag constraint.

    The active BMO slag calculation is linear in ore quantities while fuel ash
    scales with hot-metal production and flux/dust rows remain fixed. This
    helper evaluates the configured slag calculation at zero burden and at one
    wet MT for each ore to derive ``base + sum(coeff_i * qty_i)`` terms that can
    be passed to HiGHS as a hard upper-bound row.

    Args:
         - ores: list[OreInput] - Ores selected for LP.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash rows used by slag calculation.
         - flux_inputs: list[FluxInput] | None - Fixed flux rows used by slag calculation.
         - dust_inputs: list[DustInput] | None - Dust rows deducted from full balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

    Returns:
         - return tuple[np.ndarray, float] - Per-ore slag coefficients and fixed slag base.
    """

    zero_quantities = {ore.ore_id: 0.0 for ore in ores}
    base_blend = evaluate_blend(
        ores=ores,
        quantities_mt=zero_quantities,
        feo_in_slag_pct=feo_in_slag_pct,
        fuel_cost_per_thm_rs=0.0,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
    )
    base_slag_mt = float(base_blend.slag_mt)

    coeffs: list[float] = []
    for ore in ores:
        unit_quantities = dict(zero_quantities)
        unit_quantities[ore.ore_id] = 1.0
        unit_blend = evaluate_blend(
            ores=ores,
            quantities_mt=unit_quantities,
            feo_in_slag_pct=feo_in_slag_pct,
            fuel_cost_per_thm_rs=0.0,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
        )
        coeffs.append(float(unit_blend.slag_mt) - base_slag_mt)

    return np.array(coeffs, dtype=float), base_slag_mt


def run_lp_baseline(
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
) -> tuple[BlendEvaluation | None, list[str]]:
    """
    Run the deterministic LP baseline for selected BMO ores.

    The LP variables are wet ore quantities, and total blend quantity is a
    solver output. Share limits are represented as linear relationships against
    ``sum(qty)`` instead of using a fixed target burden quantity. Fe production
    is held within the standard display tolerance and final slag is enforced as
    a hard cap, so the baseline only returns a blend that satisfies the
    operator's target hot-metal and target slag settings.

    Args:
         - ores: list[OreInput] - Ores selected for optimization.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum slag quantity checked after solve.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

    Returns:
         - return tuple[BlendEvaluation | None, list[str]] - LP blend and errors.
    """

    pre_errors = validate_ore_bounds(ores)
    if pre_errors:
        return None, pre_errors

    n = len(ores)
    c = np.array([float(ore.price_rs_per_mt) for ore in ores], dtype=float)

    fe_coeff = np.array(
        [
            compute_dry_fraction(ore.chemistry.moisture_pct)
            * (float(ore.chemistry.fe_t_pct) / 100.0)
            for ore in ores
        ],
        dtype=float,
    )
    a_ub_rows = [
        -fe_coeff,  # Fe >= target
        fe_coeff,  # Fe <= target + tolerance
    ]
    b_ub_values = [
        -float(target_production_mt),
        float(target_production_mt) + FE_TOLERANCE_MT,
    ]

    slag_coeff, slag_base_mt = _build_linear_slag_terms(
        ores,
        feo_in_slag_pct=feo_in_slag_pct,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
    )
    a_ub_rows.append(slag_coeff)
    b_ub_values.append(float(target_slag_qty_mt) - slag_base_mt)

    for idx, ore in enumerate(ores):
        min_share = float(ore.min_share_pct) / 100.0
        max_share = float(ore.max_share_pct) / 100.0

        min_share_row = np.full(n, min_share, dtype=float)
        min_share_row[idx] -= 1.0
        a_ub_rows.append(min_share_row)
        b_ub_values.append(0.0)

        max_share_row = np.full(n, -max_share, dtype=float)
        max_share_row[idx] += 1.0
        a_ub_rows.append(max_share_row)
        b_ub_values.append(0.0)

    A_ub = np.vstack(a_ub_rows)
    b_ub = np.array(b_ub_values, dtype=float)

    bounds: list[tuple[float, float]] = []
    for ore in ores:
        bounds.append((0.0, max(0.0, float(ore.stock_mt))))

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
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
        fuel_cost_per_thm_rs=0.0,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
    )
    violations = check_blend_constraints(
        blend,
        ores,
        target_production_mt=target_production_mt,
        target_slag_qty_mt=target_slag_qty_mt,
    )
    blend.feasible = len(violations) == 0
    blend.violations = violations
    return blend, []
