"""LP baseline solver for BMO wet burden planning.

This module builds the deterministic ore-cost LP baseline using wet quantity
bounds while applying dry-weight Fe production and final slag as hard process
targets. If no blend can meet those physical limits, LP returns infeasible
instead of showing a low-cost blend that violates the target slag cap.
"""

from __future__ import annotations

from dataclasses import replace

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
LP_EXACT_SLAG_TOLERANCE_MT = 1e-6
LP_EXACT_SLAG_RETRIES = 6


def _structural_infeasibility_reasons(
    ores: list[OreInput], target_fe_mt: float
) -> list[str]:
    """
    Detect infeasibility causes that need no solver, with operator guidance.

    HiGHS only reports a blunt "problem is infeasible". For a furnace operator
    the useful question is *why* and *what to change*. These checks catch the
    common stock/share/Fe mismatches up front so the page can show an actionable
    message instead of the raw solver status.

    Args:
         - ores: list[OreInput] - Ores selected for the LP.
         - target_fe_mt: float - Required dry Fe production in MT.

    Returns:
         - return list[str] - Human-readable, actionable infeasibility reasons.
    """

    reasons: list[str] = []
    stocked = [ore for ore in ores if float(ore.stock_mt) > 0.0]
    if not stocked:
        return ["No selected ore has positive stock; add stock before running."]

    for ore in ores:
        if float(ore.stock_mt) <= 0.0 and float(ore.min_share_pct) > 0.0:
            reasons.append(
                f"{ore.display_name} has a minimum share of "
                f"{float(ore.min_share_pct):.0f}% but zero stock. Add stock for it "
                "or set its minimum share to 0%."
            )

    # Only ores with stock can carry burden (others are pinned to zero by their
    # stock bound). If their max shares cannot sum to a full burden, no positive
    # blend exists: total <= cover * total forces total <= 0.
    cover_pct = sum(float(ore.max_share_pct) for ore in stocked)
    if cover_pct < 100.0 - 1e-6:
        names = ", ".join(ore.display_name for ore in stocked)
        reasons.append(
            f"Only ores with stock can fill the burden ({names}), but their "
            f"maximum shares sum to {cover_pct:.0f}% (< 100%). Add stock for more "
            "ores or raise their max-share limits."
        )

    fe_capacity_mt = sum(
        compute_dry_fraction(ore.chemistry.moisture_pct)
        * float(ore.stock_mt)
        * (float(ore.chemistry.fe_t_pct) / 100.0)
        for ore in stocked
    )
    if fe_capacity_mt < float(target_fe_mt) - FE_TOLERANCE_MT:
        reasons.append(
            f"Ores in stock can supply at most {fe_capacity_mt:,.0f} MT Fe, but the "
            f"target needs {float(target_fe_mt):,.0f} MT. Add stock, select "
            "higher-Fe ores, or lower the target hot metal."
        )

    return reasons


def _build_linear_slag_and_basicity_terms(
    ores: list[OreInput],
    *,
    feo_in_slag_pct: float,
    fuel_ash_inputs: list[FuelAshInput] | None,
    flux_inputs: list[FluxInput] | None,
    dust_inputs: list[DustInput] | None,
    slag_balance_settings: SlagBalanceSettings | None,
    hot_metal_target_mt: float | None,
    variable_fluxes: list[FluxInput] | None = None,
) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray, float]:
    """
    Estimate linear final-slag and basicity terms for LP hard constraints.

    The active BMO slag calculation is linear in ore quantities while fuel ash
    scales with hot-metal production and flux/dust rows remain fixed. This
    helper evaluates the configured slag calculation at zero burden and at one
    wet MT for each ore to derive ``base + sum(coeff_i * qty_i)`` terms for
    total slag, basicity numerator, and basicity denominator.

    Args:
         - ores: list[OreInput] - Ores selected for LP.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash rows used by slag calculation.
         - flux_inputs: list[FluxInput] | None - Fixed flux rows used by slag calculation.
         - dust_inputs: list[DustInput] | None - Dust rows deducted from full balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.

    Returns:
         - return tuple - Slag, basicity numerator, and denominator linear terms.
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
        hot_metal_target_mt=hot_metal_target_mt,
    )
    base_slag_mt = float(base_blend.slag_mt)
    base_basicity_numerator_mt = float(
        base_blend.diagnostics.get("slag_basicity_numerator_mt", 0.0) or 0.0
    )
    base_basicity_denominator_mt = float(
        base_blend.diagnostics.get("slag_basicity_denominator_mt", 0.0) or 0.0
    )
    base_t_basicity_numerator_mt = float(
        base_blend.diagnostics.get("slag_t_basicity_numerator_mt", 0.0) or 0.0
    )

    slag_coeffs: list[float] = []
    basicity_numerator_coeffs: list[float] = []
    basicity_denominator_coeffs: list[float] = []
    t_basicity_numerator_coeffs: list[float] = []
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
            hot_metal_target_mt=hot_metal_target_mt,
        )
        slag_coeffs.append(float(unit_blend.slag_mt) - base_slag_mt)
        basicity_numerator_coeffs.append(
            float(unit_blend.diagnostics.get("slag_basicity_numerator_mt", 0.0) or 0.0)
            - base_basicity_numerator_mt
        )
        basicity_denominator_coeffs.append(
            float(unit_blend.diagnostics.get("slag_basicity_denominator_mt", 0.0) or 0.0)
            - base_basicity_denominator_mt
        )
        t_basicity_numerator_coeffs.append(
            float(unit_blend.diagnostics.get("slag_t_basicity_numerator_mt", 0.0) or 0.0)
            - base_t_basicity_numerator_mt
        )

    # Marginal contribution of each optimisable flux (per 1 wet MT), appended as
    # extra decision-variable columns. Dolomite raises CaO (basicity numerator)
    # and MgO (T-basicity); quartz raises SiO2 (denominator, lowering basicity).
    base_flux_inputs = list(flux_inputs or [])
    for flux in variable_fluxes or []:
        unit_flux_blend = evaluate_blend(
            ores=ores,
            quantities_mt=zero_quantities,
            feo_in_slag_pct=feo_in_slag_pct,
            fuel_cost_per_thm_rs=0.0,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=base_flux_inputs + [replace(flux, wet_qty_mt=1.0, enabled=True)],
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            hot_metal_target_mt=hot_metal_target_mt,
        )
        slag_coeffs.append(float(unit_flux_blend.slag_mt) - base_slag_mt)
        basicity_numerator_coeffs.append(
            float(unit_flux_blend.diagnostics.get("slag_basicity_numerator_mt", 0.0) or 0.0)
            - base_basicity_numerator_mt
        )
        basicity_denominator_coeffs.append(
            float(unit_flux_blend.diagnostics.get("slag_basicity_denominator_mt", 0.0) or 0.0)
            - base_basicity_denominator_mt
        )
        t_basicity_numerator_coeffs.append(
            float(unit_flux_blend.diagnostics.get("slag_t_basicity_numerator_mt", 0.0) or 0.0)
            - base_t_basicity_numerator_mt
        )

    return (
        np.array(slag_coeffs, dtype=float),
        base_slag_mt,
        np.array(basicity_numerator_coeffs, dtype=float),
        base_basicity_numerator_mt,
        np.array(basicity_denominator_coeffs, dtype=float),
        base_basicity_denominator_mt,
        np.array(t_basicity_numerator_coeffs, dtype=float),
        base_t_basicity_numerator_mt,
    )


def _explain_lp_infeasibility(
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
    target_slag_basicity_min: float | None,
    target_slag_basicity_max: float | None,
    target_slag_t_basicity_min: float | None,
    target_slag_t_basicity_max: float | None,
    fuel_ash_inputs: list[FuelAshInput] | None,
    flux_inputs: list[FluxInput] | None,
    dust_inputs: list[DustInput] | None,
    slag_balance_settings: SlagBalanceSettings | None,
    hot_metal_target_mt: float | None,
) -> list[str]:
    """
    Explain an infeasible LP by isolating the binding constraint family.

    Structural stock/share/Fe causes are reported first because they need no
    solve. Otherwise the LP is re-solved with the slag basicity bounds removed
    and then with the slag cap lifted, so the page can tell the operator whether
    the basicity limits or the Max Slag cap is what blocks the blend, along with
    the value actually achievable for this burden.

    Args:
         - ores: list[OreInput] - Ores selected for the LP.
         - target_production_mt: float - Required dry Fe production in MT.
         - target_slag_qty_mt: float - Operator slag cap in MT.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
          - target_slag_basicity_min/max: float | None - CaO/SiO2 bounds.
          - target_slag_t_basicity_min/max: float | None - Legacy display-only inputs,
            ignored by LP feasibility.
         - fuel_ash_inputs / flux_inputs / dust_inputs - Fixed slag contributors.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - hot_metal_target_mt: float | None - HM basis for fuel/slag rates.

    Returns:
         - return list[str] - Actionable reasons; empty if none could be isolated.
    """

    reasons = _structural_infeasibility_reasons(ores, target_production_mt)
    if reasons:
        return reasons

    common = dict(
        feo_in_slag_pct=feo_in_slag_pct,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        hot_metal_target_mt=hot_metal_target_mt,
    )
    has_basicity = any(
        bound is not None
        for bound in (
            target_slag_basicity_min,
            target_slag_basicity_max,
        )
    )
    big_slag_cap_mt = max(float(target_slag_qty_mt) * 10.0, 1.0e6)

    if has_basicity:
        without_basicity, _ = run_lp_baseline(
            ores,
            target_production_mt=target_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            _explain=False,
            **common,
        )
        if without_basicity is not None:
            with_basicity_without_slag_cap, _ = run_lp_baseline(
                ores,
                target_production_mt=target_production_mt,
                target_slag_qty_mt=big_slag_cap_mt,
                target_slag_basicity_min=target_slag_basicity_min,
                target_slag_basicity_max=target_slag_basicity_max,
                target_slag_t_basicity_min=None,
                target_slag_t_basicity_max=None,
                _explain=False,
                **common,
            )
            if with_basicity_without_slag_cap is not None:
                flux_qty = (
                    with_basicity_without_slag_cap.diagnostics.get(
                        "lp_flux_quantities_mt", {}
                    )
                    or {}
                )
                added_flux = {
                    str(flux_id): float(qty)
                    for flux_id, qty in flux_qty.items()
                    if float(qty or 0.0) > 1e-6
                }
                flux_text = ""
                if added_flux:
                    flux_text = " LP would add " + ", ".join(
                        f"{flux_id} {qty:,.0f} MT"
                        for flux_id, qty in added_flux.items()
                    ) + "."
                reasons.append(
                    "The blend can meet the slag basicity limits only if the "
                    "Max Slag cap is lifted. With LP-added flux it reaches "
                    f"CaO/SiO2 ~ {with_basicity_without_slag_cap.slag_basicity:.2f} "
                    f"but slag rises to about {with_basicity_without_slag_cap.slag_mt:,.0f} MT "
                    f"versus the {float(target_slag_qty_mt):,.0f} MT cap."
                    f"{flux_text} Raise Max Slag, relax the basicity bounds, or "
                    "reduce slag from the burden before adding CaO flux."
                )
                return reasons
            reasons.append(
                "The blend is feasible once the slag basicity limits are removed, "
                "so the basicity bounds are the binding limit. This burden reaches "
                f"CaO/SiO2 ~ {without_basicity.slag_basicity:.2f}. Widen the basicity bounds "
                "or adjust flux (CaO source) to match."
            )
            return reasons

    without_slag_cap, _ = run_lp_baseline(
        ores,
        target_production_mt=target_production_mt,
        target_slag_qty_mt=big_slag_cap_mt,
        _explain=False,
        **common,
    )
    if without_slag_cap is not None:
        reasons.append(
            "The blend only becomes feasible when the Max Slag cap is lifted; "
            f"this burden produces about {without_slag_cap.slag_mt:,.0f} MT slag "
            f"versus the {float(target_slag_qty_mt):,.0f} MT cap. Raise Max Slag or "
            "change the burden / flux."
        )
        return reasons

    return reasons


def run_lp_baseline(
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    feo_in_slag_pct: float,
    target_slag_basicity_min: float | None = None,
    target_slag_basicity_max: float | None = None,
    target_slag_t_basicity_min: float | None = None,
    target_slag_t_basicity_max: float | None = None,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    hot_metal_target_mt: float | None = None,
    _explain: bool = True,
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
         - target_slag_basicity_min: float | None - Minimum allowed CaO / SiO2 basicity.
         - target_slag_basicity_max: float | None - Maximum allowed CaO / SiO2 basicity.
         - target_slag_t_basicity_min: float | None - Legacy display-only input,
           ignored by LP optimization.
         - target_slag_t_basicity_max: float | None - Legacy display-only input,
           ignored by LP optimization.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - hot_metal_target_mt: float | None - Operator HM target used as THM denominator.

    Returns:
         - return tuple[BlendEvaluation | None, list[str]] - LP blend and errors.
    """

    pre_errors = validate_ore_bounds(ores)
    if (
        target_slag_basicity_min is not None
        and target_slag_basicity_max is not None
        and float(target_slag_basicity_min) > float(target_slag_basicity_max)
    ):
        pre_errors.append("Min slag basicity cannot be greater than max slag basicity.")
    if pre_errors:
        return None, pre_errors

    n = len(ores)
    # Optimisable fluxes (e.g. dolomite, quartz) become extra LP decision
    # variables so the solver can add just enough flux to hold slag basicity
    # within bounds; all other fluxes are fixed additions folded into the base.
    all_fluxes = list(flux_inputs or [])
    variable_fluxes = [
        flux
        for flux in all_fluxes
        if flux.optimizable and flux.enabled and float(flux.stock_mt) > 0.0
    ]
    variable_flux_ids = {flux.flux_id for flux in variable_fluxes}
    fixed_fluxes = [flux for flux in all_fluxes if flux.flux_id not in variable_flux_ids]
    n_flux = len(variable_fluxes)

    ore_prices = [float(ore.price_rs_per_mt) for ore in ores]
    flux_prices = [float(flux.price_rs_per_mt) for flux in variable_fluxes]
    c = np.array(ore_prices + flux_prices, dtype=float)

    fe_coeff = np.concatenate(
        [
            np.array(
                [
                    compute_dry_fraction(ore.chemistry.moisture_pct)
                    * (float(ore.chemistry.fe_t_pct) / 100.0)
                    for ore in ores
                ],
                dtype=float,
            ),
            # Flux iron reports to slag, not hot metal, so flux adds no Fe here.
            np.zeros(n_flux, dtype=float),
        ]
    )
    a_ub_rows = [
        -fe_coeff,  # Fe >= target
        fe_coeff,  # Fe <= target + tolerance
    ]
    b_ub_values = [
        -float(target_production_mt),
        float(target_production_mt) + FE_TOLERANCE_MT,
    ]

    (
        slag_coeff,
        slag_base_mt,
        basicity_numerator_coeff,
        basicity_numerator_base_mt,
        basicity_denominator_coeff,
        basicity_denominator_base_mt,
        _t_basicity_numerator_coeff,
        _t_basicity_numerator_base_mt,
    ) = _build_linear_slag_and_basicity_terms(
        ores,
        feo_in_slag_pct=feo_in_slag_pct,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=fixed_fluxes,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        hot_metal_target_mt=hot_metal_target_mt,
        variable_fluxes=variable_fluxes,
    )
    slag_row_idx = len(a_ub_rows)
    a_ub_rows.append(slag_coeff)
    b_ub_values.append(float(target_slag_qty_mt) - slag_base_mt)

    def _add_basicity_bounds(
        *,
        numerator_coeff: np.ndarray,
        numerator_base_mt: float,
        denominator_coeff: np.ndarray,
        denominator_base_mt: float,
        min_value: float | None,
        max_value: float | None,
    ) -> None:
        if min_value is not None:
            min_basicity = float(min_value)
            a_ub_rows.append(min_basicity * denominator_coeff - numerator_coeff)
            b_ub_values.append(numerator_base_mt - min_basicity * denominator_base_mt)

        if max_value is not None:
            max_basicity = float(max_value)
            a_ub_rows.append(numerator_coeff - max_basicity * denominator_coeff)
            b_ub_values.append(max_basicity * denominator_base_mt - numerator_base_mt)

    _add_basicity_bounds(
        numerator_coeff=basicity_numerator_coeff,
        numerator_base_mt=basicity_numerator_base_mt,
        denominator_coeff=basicity_denominator_coeff,
        denominator_base_mt=basicity_denominator_base_mt,
        min_value=target_slag_basicity_min,
        max_value=target_slag_basicity_max,
    )
    for idx, ore in enumerate(ores):
        min_share = float(ore.min_share_pct) / 100.0
        max_share = float(ore.max_share_pct) / 100.0

        # Share bounds apply to ore columns only; fluxes are separate additions
        # (padded with zeros) that don't participate in the ore-share split.
        min_share_row = np.concatenate(
            [np.full(n, min_share, dtype=float), np.zeros(n_flux, dtype=float)]
        )
        min_share_row[idx] -= 1.0
        a_ub_rows.append(min_share_row)
        b_ub_values.append(0.0)

        max_share_row = np.concatenate(
            [np.full(n, -max_share, dtype=float), np.zeros(n_flux, dtype=float)]
        )
        max_share_row[idx] += 1.0
        a_ub_rows.append(max_share_row)
        b_ub_values.append(0.0)

    A_ub = np.vstack(a_ub_rows)
    base_b_ub = np.array(b_ub_values, dtype=float)

    bounds: list[tuple[float, float]] = []
    for ore in ores:
        bounds.append((0.0, max(0.0, float(ore.stock_mt))))
    for flux in variable_fluxes:
        bounds.append((0.0, max(0.0, float(flux.stock_mt))))

    slag_tightening_mt = 0.0
    last_slag_mt: float | None = None
    for attempt in range(LP_EXACT_SLAG_RETRIES + 1):
        b_ub = base_b_ub.copy()
        b_ub[slag_row_idx] -= slag_tightening_mt

        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if not result.success or result.x is None:
            err = result.message or "LP solver failed."
            if attempt > 0:
                return None, [
                    "LP infeasible or failed after exact slag tightening: "
                    f"{err}"
                ]
            messages = [f"LP infeasible or failed: {err}"]
            if _explain:
                messages.extend(
                    _explain_lp_infeasibility(
                        ores,
                        target_production_mt=target_production_mt,
                        target_slag_qty_mt=target_slag_qty_mt,
                        feo_in_slag_pct=feo_in_slag_pct,
                        target_slag_basicity_min=target_slag_basicity_min,
                        target_slag_basicity_max=target_slag_basicity_max,
                        target_slag_t_basicity_min=None,
                        target_slag_t_basicity_max=None,
                        fuel_ash_inputs=fuel_ash_inputs,
                        flux_inputs=flux_inputs,
                        dust_inputs=dust_inputs,
                        slag_balance_settings=slag_balance_settings,
                        hot_metal_target_mt=hot_metal_target_mt,
                    )
                )
            return None, messages

        quantities = {
            ore.ore_id: float(result.x[idx]) for idx, ore in enumerate(ores)
        }
        # Rebuild the flux list with the LP-decided quantities for optimisable
        # fluxes so the final evaluation reflects the flux the LP actually added.
        solved_flux_quantities = {
            flux.flux_id: float(result.x[n + j])
            for j, flux in enumerate(variable_fluxes)
        }
        solved_flux_inputs = list(fixed_fluxes) + [
            replace(flux, wet_qty_mt=float(result.x[n + j]))
            for j, flux in enumerate(variable_fluxes)
        ]
        blend = evaluate_blend(
            ores=ores,
            quantities_mt=quantities,
            feo_in_slag_pct=feo_in_slag_pct,
            fuel_cost_per_thm_rs=0.0,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=solved_flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            hot_metal_target_mt=hot_metal_target_mt,
        )
        blend.diagnostics["lp_flux_quantities_mt"] = solved_flux_quantities
        # Cost of the flux the LP bought (per THM), on the same HM basis the blend
        # uses, so the displayed total cost includes optimizer-added flux.
        flux_cost_total_rs = sum(
            float(result.x[n + j]) * float(flux.price_rs_per_mt)
            for j, flux in enumerate(variable_fluxes)
        )
        flux_thm_basis = float(
            hot_metal_target_mt or blend.fe_production_mt or 0.0
        )
        blend.diagnostics["flux_cost_per_thm_rs"] = (
            float(flux_cost_total_rs / flux_thm_basis) if flux_thm_basis > 0.0 else 0.0
        )
        violations = check_blend_constraints(
            blend,
            ores,
            target_production_mt=target_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            target_slag_basicity_min=target_slag_basicity_min,
            target_slag_basicity_max=target_slag_basicity_max,
            target_slag_t_basicity_min=None,
            target_slag_t_basicity_max=None,
            slag_tolerance_mt=LP_EXACT_SLAG_TOLERANCE_MT,
        )
        if not violations:
            blend.feasible = True
            blend.violations = []
            blend.diagnostics["lp_exact_slag_tightening_mt"] = float(
                slag_tightening_mt
            )
            blend.diagnostics["lp_exact_slag_attempts"] = int(attempt + 1)
            return blend, []

        blend.feasible = False
        blend.violations = violations
        last_slag_mt = float(blend.slag_mt)
        slag_excess_mt = last_slag_mt - float(target_slag_qty_mt)
        if slag_excess_mt <= LP_EXACT_SLAG_TOLERANCE_MT:
            return None, [
                "LP solved the linear model but failed final exact validation: "
                + "; ".join(violations)
            ]
        slag_tightening_mt += slag_excess_mt + LP_EXACT_SLAG_TOLERANCE_MT

    return None, [
        "LP could not satisfy the exact slag cap after tightening "
        f"({last_slag_mt:.2f} > {float(target_slag_qty_mt):.2f} MT)."
    ]
