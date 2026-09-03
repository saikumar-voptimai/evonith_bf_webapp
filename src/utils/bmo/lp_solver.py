"""LP baseline solver for BMO wet burden planning.

This module builds the deterministic ore-cost LP baseline using wet quantity
bounds while applying dry-weight Fe production and final slag as hard process
targets. If no blend can meet those physical limits, LP returns infeasible
instead of showing a low-cost blend that violates the target slag cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linprog

from utils.bmo.calculations import compute_dry_fraction, evaluate_blend
from utils.bmo.coke_correction import (
    CokeCorrectionReference,
    CokeCorrectionSettings,
    build_linear_coke_correction_cost_coeffs,
)
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


@dataclass
class LinearSlagTerms:
    """
    First-order ``base + coeff . x`` expansion of every slag quantity the LP bounds.

    Each field pairs a coefficient vector over the LP decision variables (ore
    columns then optimisable-flux columns) with the intercept that reproduces the
    exact calculation at the expansion point. Every constraint the LP places on
    slag is a ratio of two of these linear quantities, which is what keeps the
    problem a true LP rather than a fractional program:

        CaO / SiO2      >= b_min      ->  b_min * SiO2 - CaO      <= 0
        Al2O3 / slag    <= a_max      ->  Al2O3 - a_max * slag    <= 0
        MgO / Al2O3     >= r_min      ->  r_min * Al2O3 - MgO     <= 0

    Args:
         - slag_coeff / slag_base_mt: Total final slag.
         - basicity_numerator_*: CaO in final slag (B2 numerator).
         - basicity_denominator_*: SiO2 in final slag (shared B2 / T-basicity denominator).
         - t_basicity_numerator_*: CaO + MgO in final slag (T-basicity numerator).
         - al2o3_*: Al2O3 in final slag.
         - mgo_*: MgO in final slag.

    Returns:
         - return LinearSlagTerms - Linearised slag terms for LP row construction.
    """

    slag_coeff: np.ndarray
    slag_base_mt: float
    basicity_numerator_coeff: np.ndarray
    basicity_numerator_base_mt: float
    basicity_denominator_coeff: np.ndarray
    basicity_denominator_base_mt: float
    t_basicity_numerator_coeff: np.ndarray
    t_basicity_numerator_base_mt: float
    al2o3_coeff: np.ndarray = field(default_factory=lambda: np.zeros(0))
    al2o3_base_mt: float = 0.0
    mgo_coeff: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mgo_base_mt: float = 0.0
    production_coeff: np.ndarray = field(default_factory=lambda: np.zeros(0))
    production_base_mt: float = 0.0


def _structural_infeasibility_reasons(
    ores: list[OreInput],
    target_fe_mt: float,
    max_burden_qty_mt: float | None = None,
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

    # Even a burden made entirely of the richest available ore needs a minimum
    # tonnage to carry the Fe target. If that alone breaches the charging cap,
    # no blend can fit and the operator needs richer material, not a re-solve.
    if max_burden_qty_mt is not None and float(max_burden_qty_mt) > 0.0:
        best_fe_per_wet_mt = max(
            (
                compute_dry_fraction(ore.chemistry.moisture_pct)
                * (float(ore.chemistry.fe_t_pct) / 100.0)
                for ore in stocked
            ),
            default=0.0,
        )
        if best_fe_per_wet_mt > 0.0:
            min_burden_mt = float(target_fe_mt) / best_fe_per_wet_mt
            if min_burden_mt > float(max_burden_qty_mt) + 1e-6:
                reasons.append(
                    f"The Fe target needs at least {min_burden_mt:,.0f} MT of burden "
                    f"even using only the richest ore in stock, but the furnace can "
                    f"charge {float(max_burden_qty_mt):,.0f} MT of IBRM + flux per "
                    "day. Select higher-Fe material, or lower the target hot metal."
                )

    return reasons


def _nominal_linearisation_quantities(
    ores: list[OreInput], target_production_mt: float
) -> dict[str, float]:
    """
    Build a burden that actually delivers the Fe target, for use as the
    linearisation point.

    The slag balance nets BF gas dust off the component totals and clamps the
    result at zero. Probing at an EMPTY burden puts those clamps in a different
    branch from any real solution: with 48 MT of dust at 33% Fe the deduction
    (15.8 MT) exceeds the fuel-ash Fe (~6.9 MT), so net Fe clamps to zero, pig
    iron is zero, and the probe sees none of the SiO2 that silicon reduction
    removes. The linear model then under-states basicity by ~11% and the LP
    hands back a blend that fails its own exact validation.

    Probing around a realistic burden keeps every clamp in the same branch the
    solution lives in, so the dust subtraction stays linear where it matters.

    Args:
         - ores: list[OreInput] - Ores selected for LP.
         - target_production_mt: float - Required dry Fe production in MT.

    Returns:
         - return dict[str, float] - Wet quantities keyed by ore id.
    """

    count = len(ores)
    if count == 0:
        return {}
    lo = np.array([max(0.0, float(o.min_share_pct)) / 100.0 for o in ores])
    hi = np.array([max(0.0, float(o.max_share_pct)) / 100.0 for o in ores])
    shares = np.clip((lo + hi) / 2.0, lo, np.maximum(hi, lo))
    total_share = float(np.sum(shares))
    shares = (
        shares / total_share
        if total_share > 0.0
        else np.full(count, 1.0 / count, dtype=float)
    )
    fe_per_wet = np.array(
        [
            compute_dry_fraction(o.chemistry.moisture_pct)
            * max(0.0, float(o.chemistry.fe_t_pct))
            / 100.0
            for o in ores
        ],
        dtype=float,
    )
    fe_per_blend = float(np.dot(shares, fe_per_wet))
    if fe_per_blend <= 0.0 or target_production_mt <= 0.0:
        return {ore.ore_id: 0.0 for ore in ores}
    total_wet = float(target_production_mt) / fe_per_blend
    return {ore.ore_id: float(shares[i] * total_wet) for i, ore in enumerate(ores)}


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
    target_production_mt: float = 0.0,
    charge_mass_mt: float = 26.4,
) -> LinearSlagTerms:
    """
    Estimate linear final-slag, basicity, and slag-chemistry terms for LP constraints.

    The active BMO slag calculation is linear in ore quantities while fuel ash
    scales with hot-metal production and flux/dust rows remain fixed. This
    helper takes a first-order expansion of the configured slag calculation and
    returns ``base + sum(coeff_i * qty_i)`` terms for total slag, basicity
    numerator, and basicity denominator.

    The expansion point is a NOMINAL burden that meets the Fe target, not an
    empty one: BF gas dust is netted off the component totals with a zero clamp,
    and at an empty burden those clamps sit in a different branch from any real
    solution. See ``_nominal_linearisation_quantities``. The returned terms are
    still of the form ``base + coeff . x`` - the expansion point is folded back
    into ``base`` - so every caller is unaffected.

    Args:
         - ores: list[OreInput] - Ores selected for LP.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash rows used by slag calculation.
         - flux_inputs: list[FluxInput] | None - Fixed flux rows used by slag calculation.
         - dust_inputs: list[DustInput] | None - Dust rows deducted from full balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - target_production_mt: float - Fe target, used to size the expansion point.

    Returns:
         - return LinearSlagTerms - Slag, basicity, and slag-chemistry linear terms.
    """

    # Every quantity the LP needs to bound, and where to read it off a blend.
    # ``None`` means "total slag", which lives on the blend rather than in
    # diagnostics. Adding a bounded slag quantity means adding one row here.
    probes: dict[str, str | None] = {
        "slag": None,
        "production": "iron_closure_production_mt",
        "basicity_numerator": "slag_basicity_numerator_mt",
        "basicity_denominator": "slag_basicity_denominator_mt",
        "t_basicity_numerator": "slag_t_basicity_numerator_mt",
        "al2o3": "slag_al2o3_mt",
        "mgo": "slag_mgo_mt",
    }

    def _read(blend: BlendEvaluation, key: str) -> float:
        diagnostic_key = probes[key]
        if diagnostic_key is None:
            return float(blend.slag_mt)
        return float(blend.diagnostics.get(diagnostic_key, 0.0) or 0.0)

    zero_quantities = _nominal_linearisation_quantities(ores, target_production_mt)
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
        charge_mass_mt=charge_mass_mt,
    )
    base_values = {key: _read(base_blend, key) for key in probes}
    coeffs: dict[str, list[float]] = {key: [] for key in probes}

    for ore in ores:
        # Forward difference of one wet MT AROUND the expansion point, not from
        # an empty burden, so the dust clamps stay in the solution's branch.
        unit_quantities = dict(zero_quantities)
        unit_quantities[ore.ore_id] = float(zero_quantities.get(ore.ore_id, 0.0)) + 1.0
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
            charge_mass_mt=charge_mass_mt,
        )
        for key in probes:
            coeffs[key].append(_read(unit_blend, key) - base_values[key])

    # Marginal contribution of each optimisable flux (per 1 wet MT), appended as
    # extra decision-variable columns. Dolomite raises CaO (basicity numerator)
    # and MgO (T-basicity, and the MgO floor); quartz raises SiO2 (denominator,
    # lowering basicity).
    base_flux_inputs = list(flux_inputs or [])
    for flux in variable_fluxes or []:
        unit_flux_blend = evaluate_blend(
            ores=ores,
            quantities_mt=zero_quantities,
            feo_in_slag_pct=feo_in_slag_pct,
            fuel_cost_per_thm_rs=0.0,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=base_flux_inputs
            + [replace(flux, wet_qty_mt=1.0, enabled=True)],
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            hot_metal_target_mt=hot_metal_target_mt,
            charge_mass_mt=charge_mass_mt,
        )
        for key in probes:
            coeffs[key].append(_read(unit_flux_blend, key) - base_values[key])

    # Fold the expansion point back into the intercept. Callers consume these as
    # ``base + coeff . x`` over absolute quantities, so a first-order expansion
    # taken at x0 has to report base = f(x0) - coeff . x0. Flux columns expand at
    # zero quantity, so only the ore columns contribute to the shift.
    expansion_point = np.array(
        [float(zero_quantities.get(ore.ore_id, 0.0)) for ore in ores]
        + [0.0] * len(variable_fluxes or []),
        dtype=float,
    )
    arrays = {key: np.array(values, dtype=float) for key, values in coeffs.items()}
    intercepts = {
        key: base_values[key] - float(arrays[key] @ expansion_point) for key in probes
    }

    return LinearSlagTerms(
        slag_coeff=arrays["slag"],
        slag_base_mt=intercepts["slag"],
        basicity_numerator_coeff=arrays["basicity_numerator"],
        basicity_numerator_base_mt=intercepts["basicity_numerator"],
        basicity_denominator_coeff=arrays["basicity_denominator"],
        basicity_denominator_base_mt=intercepts["basicity_denominator"],
        t_basicity_numerator_coeff=arrays["t_basicity_numerator"],
        t_basicity_numerator_base_mt=intercepts["t_basicity_numerator"],
        al2o3_coeff=arrays["al2o3"],
        al2o3_base_mt=intercepts["al2o3"],
        mgo_coeff=arrays["mgo"],
        mgo_base_mt=intercepts["mgo"],
        production_coeff=arrays["production"],
        production_base_mt=intercepts["production"],
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
    target_slag_al2o3_max_pct: float | None,
    target_slag_mgo_min_pct: float | None,
    target_slag_mgo_al2o3_ratio_min: float | None,
    max_burden_qty_mt: float | None,
    fuel_ash_inputs: list[FuelAshInput] | None,
    flux_inputs: list[FluxInput] | None,
    dust_inputs: list[DustInput] | None,
    slag_balance_settings: SlagBalanceSettings | None,
    hot_metal_target_mt: float | None,
    coke_correction_settings: CokeCorrectionSettings | None = None,
    coke_correction_reference: CokeCorrectionReference | None = None,
    price_coke_correction: bool = False,
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
          - target_slag_t_basicity_min/max: float | None - (CaO+MgO)/SiO2 bounds.
          - target_slag_al2o3_max_pct / target_slag_mgo_min_pct /
            target_slag_mgo_al2o3_ratio_min: float | None - Slag-quality limits, each
            dropped in turn to find which one blocks the blend.
         - fuel_ash_inputs / flux_inputs / dust_inputs - Fixed slag contributors.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - hot_metal_target_mt: float | None - HM basis for fuel/slag rates.
         - coke_correction_settings / coke_correction_reference - Forwarded to
           every re-solve. Omitting them would make the explanation probe a
           different objective than the one that actually failed, so it could
           report the wrong binding constraint.

    Returns:
         - return list[str] - Actionable reasons; empty if none could be isolated.
    """

    reasons = _structural_infeasibility_reasons(
        ores, target_production_mt, max_burden_qty_mt
    )
    if reasons:
        return reasons

    # Values every re-solve below must carry unless it is deliberately dropping
    # one of them to test whether that limit is the blocker.
    quality_kwargs: dict[str, float | None] = {
        "target_slag_al2o3_max_pct": target_slag_al2o3_max_pct,
        "target_slag_mgo_min_pct": target_slag_mgo_min_pct,
        "target_slag_mgo_al2o3_ratio_min": target_slag_mgo_al2o3_ratio_min,
        "target_slag_t_basicity_min": target_slag_t_basicity_min,
        "target_slag_t_basicity_max": target_slag_t_basicity_max,
    }

    # Isolate the charging cap first: it is the newest limit and the one an
    # operator is least likely to suspect, and re-solving without it is cheap.
    if max_burden_qty_mt is not None and float(max_burden_qty_mt) > 0.0:
        without_burden_cap, _ = run_lp_baseline(
            ores,
            target_production_mt=target_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            feo_in_slag_pct=feo_in_slag_pct,
            target_slag_basicity_min=target_slag_basicity_min,
            target_slag_basicity_max=target_slag_basicity_max,
            **quality_kwargs,
            max_burden_qty_mt=None,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            hot_metal_target_mt=hot_metal_target_mt,
            coke_correction_settings=coke_correction_settings,
            coke_correction_reference=coke_correction_reference,
            price_coke_correction=price_coke_correction,
            _explain=False,
        )
        if without_burden_cap is not None:
            needed_mt = float(
                without_burden_cap.diagnostics.get(
                    "total_burden_qty_mt", without_burden_cap.total_qty_mt
                )
                or 0.0
            )
            reasons.append(
                "The blend only becomes feasible when the charging-capacity cap is "
                f"lifted; it needs about {needed_mt:,.0f} MT of IBRM + flux versus "
                f"the {float(max_burden_qty_mt):,.0f} MT the furnace can charge per "
                "day. Select higher-Fe material (this burden is too lean to make the "
                "target within the charge rate), or lower the target hot metal."
            )
            return reasons

    common = dict(
        feo_in_slag_pct=feo_in_slag_pct,
        max_burden_qty_mt=max_burden_qty_mt,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=flux_inputs,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        hot_metal_target_mt=hot_metal_target_mt,
        coke_correction_settings=coke_correction_settings,
        coke_correction_reference=coke_correction_reference,
        price_coke_correction=price_coke_correction,
    )
    # Every slag limit currently in force, and how to describe it once we know
    # which one is doing the blocking. Each is dropped in turn, cheapest and
    # most-likely-culprit first, and the first drop that makes the LP solvable
    # names the binding limit AND reports what this burden can actually reach.
    slag_quality_limits: list[dict[str, object]] = [
        {
            "keys": {"target_slag_al2o3_max_pct": target_slag_al2o3_max_pct},
            "active": target_slag_al2o3_max_pct is not None,
            "label": "the slag Al2O3 cap",
            "attr": "slag_al2o3_pct",
            "format": "Al2O3 ~ {value:.2f}% against a {limit:.2f}% cap",
            "limit": target_slag_al2o3_max_pct,
            "advice": (
                "Al2O3 is inert, so its mass is fixed by what you charge and its "
                "percentage rises as slag falls. Select lower-alumina ore, or raise "
                "the cap / Max Slag."
            ),
        },
        {
            "keys": {
                "target_slag_mgo_al2o3_ratio_min": target_slag_mgo_al2o3_ratio_min
            },
            "active": target_slag_mgo_al2o3_ratio_min is not None,
            "label": "the slag MgO/Al2O3 floor",
            "attr": "slag_mgo_al2o3_ratio",
            "format": "MgO/Al2O3 ~ {value:.3f} against a {limit:.3f} floor",
            "limit": target_slag_mgo_al2o3_ratio_min,
            "advice": (
                "This ratio does not move with slag rate at all. Add an MgO source "
                "(dolomite) or drop the high-alumina material."
            ),
        },
        {
            "keys": {"target_slag_mgo_min_pct": target_slag_mgo_min_pct},
            "active": target_slag_mgo_min_pct is not None,
            "label": "the slag MgO floor",
            "attr": "slag_mgo_pct",
            "format": "MgO ~ {value:.2f}% against a {limit:.2f}% floor",
            "limit": target_slag_mgo_min_pct,
            "advice": "Add dolomite stock, or lower the MgO floor.",
        },
        {
            "keys": {
                "target_slag_t_basicity_min": target_slag_t_basicity_min,
                "target_slag_t_basicity_max": target_slag_t_basicity_max,
            },
            "active": (
                target_slag_t_basicity_min is not None
                or target_slag_t_basicity_max is not None
            ),
            "label": "the slag T Basicity bounds",
            "attr": "slag_t_basicity",
            "format": "T Basicity ~ {value:.3f}",
            "limit": None,
            "advice": "Widen the T Basicity bounds, or adjust the CaO / MgO flux split.",
        },
    ]
    active_slag_quality = [limit for limit in slag_quality_limits if limit["active"]]

    for limit in active_slag_quality:
        relaxed = dict(quality_kwargs)
        for key in limit["keys"]:
            relaxed[key] = None
        without_limit, _ = run_lp_baseline(
            ores,
            target_production_mt=target_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            target_slag_basicity_min=target_slag_basicity_min,
            target_slag_basicity_max=target_slag_basicity_max,
            _explain=False,
            **relaxed,
            **common,
        )
        if without_limit is None:
            continue
        reached = float(getattr(without_limit, str(limit["attr"]), 0.0) or 0.0)
        detail = str(limit["format"]).format(
            value=reached,
            limit=float(limit["limit"] or 0.0),
        )
        reasons.append(
            f"The blend becomes feasible once {limit['label']} is removed, so that "
            f"is the binding limit. This burden reaches {detail}. {limit['advice']}"
        )
        return reasons

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
            **quality_kwargs,
            **common,
        )
        if without_basicity is not None:
            with_basicity_without_slag_cap, _ = run_lp_baseline(
                ores,
                target_production_mt=target_production_mt,
                target_slag_qty_mt=big_slag_cap_mt,
                target_slag_basicity_min=target_slag_basicity_min,
                target_slag_basicity_max=target_slag_basicity_max,
                _explain=False,
                **quality_kwargs,
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
                    flux_text = (
                        " LP would add "
                        + ", ".join(
                            f"{flux_id} {qty:,.0f} MT"
                            for flux_id, qty in added_flux.items()
                        )
                        + "."
                    )
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
        **quality_kwargs,
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
    target_slag_al2o3_max_pct: float | None = None,
    target_slag_mgo_min_pct: float | None = None,
    target_slag_mgo_al2o3_ratio_min: float | None = None,
    max_burden_qty_mt: float | None = None,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    hot_metal_target_mt: float | None = None,
    coke_correction_settings: CokeCorrectionSettings | None = None,
    coke_correction_reference: CokeCorrectionReference | None = None,
    price_coke_correction: bool = False,
    charge_mass_mt: float = 26.4,
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
         - target_slag_t_basicity_min: float | None - Minimum (CaO + MgO) / SiO2.
         - target_slag_t_basicity_max: float | None - Maximum (CaO + MgO) / SiO2.
         - target_slag_al2o3_max_pct: float | None - Maximum Al2O3 % of final slag.
         - target_slag_mgo_min_pct: float | None - Minimum MgO % of final slag.
         - target_slag_mgo_al2o3_ratio_min: float | None - Minimum MgO / Al2O3 ratio.
         - max_burden_qty_mt: float | None - Charging-throughput ceiling on total wet
           IBRM + flux in MT. ``None`` leaves the burden quantity unbounded.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - Dust rows deducted in final balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - hot_metal_target_mt: float | None - Operator HM target used as THM denominator.
         - coke_correction_settings: CokeCorrectionSettings | None - Physics
           coke-rate correction. Its slag-heat, flux-calcination, and burden-oxygen
           terms are linear in the LP decision variables, so they are added to the
           cost vector and the problem stays a true LP.
         - coke_correction_reference: CokeCorrectionReference | None - Recent
           observed operating point. It shifts the correction by a constant, which
           does not change ``argmin``, so the LP needs it only for reporting.

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
    if (
        target_slag_t_basicity_min is not None
        and target_slag_t_basicity_max is not None
        and float(target_slag_t_basicity_min) > float(target_slag_t_basicity_max)
    ):
        pre_errors.append(
            "Min slag T Basicity cannot be greater than max slag T Basicity."
        )
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
    fixed_fluxes = [
        flux for flux in all_fluxes if flux.flux_id not in variable_flux_ids
    ]
    n_flux = len(variable_fluxes)

    ore_prices = [float(ore.price_rs_per_mt) for ore in ores]
    flux_prices = [float(flux.price_rs_per_mt) for flux in variable_fluxes]
    c = np.array(ore_prices + flux_prices, dtype=float)

    terms = _build_linear_slag_and_basicity_terms(
        ores,
        feo_in_slag_pct=feo_in_slag_pct,
        fuel_ash_inputs=fuel_ash_inputs,
        flux_inputs=fixed_fluxes,
        dust_inputs=dust_inputs,
        slag_balance_settings=slag_balance_settings,
        hot_metal_target_mt=hot_metal_target_mt,
        variable_fluxes=variable_fluxes,
        target_production_mt=target_production_mt,
        charge_mass_mt=charge_mass_mt,
    )
    use_full_iron_closure = bool(
        slag_balance_settings is not None
        and slag_balance_settings.enabled
        and hot_metal_target_mt is not None
        and float(hot_metal_target_mt) > 0.0
    )
    if use_full_iron_closure:
        production_coeff = terms.production_coeff
        production_base_mt = float(terms.production_base_mt)
        production_target_mt = float(hot_metal_target_mt)
    else:
        production_coeff = np.concatenate(
            [
                np.array(
                    [
                        compute_dry_fraction(ore.chemistry.moisture_pct)
                        * (float(ore.chemistry.fe_t_pct) / 100.0)
                        for ore in ores
                    ],
                    dtype=float,
                ),
                np.zeros(n_flux, dtype=float),
            ]
        )
        production_base_mt = 0.0
        production_target_mt = float(target_production_mt)
    # base + coeff.x >= target and <= target+tolerance.  On the full-balance
    # path this is actual pig iron, so fuel/flux Fe, dust Fe, FeO loss, and PI
    # loss are all present in the production closure.
    a_ub_rows = [-production_coeff, production_coeff]
    b_ub_values = [
        production_base_mt - production_target_mt,
        production_target_mt + FE_TOLERANCE_MT - production_base_mt,
    ]
    slag_coeff = terms.slag_coeff
    slag_base_mt = terms.slag_base_mt
    slag_row_idx = len(a_ub_rows)
    a_ub_rows.append(slag_coeff)
    b_ub_values.append(float(target_slag_qty_mt) - slag_base_mt)

    # Optionally price the physics coke correction into the objective.
    #
    # OFF by default. The LP's job is to pick the cheapest ore + flux mix that
    # satisfies the slag rate, basicity, T-basicity, Al2O3, MgO and MgO/Al2O3
    # limits - nothing else. Slag is governed by those hard constraints, not by
    # a fuel term in the cost vector, so the reported cost is the purchase cost
    # of a real basket of material and nothing has to be unpicked to explain it.
    #
    # DE is a different optimiser with a different objective (ore + predicted
    # fuel, slag limits as soft penalties). Its seed LP must be optimised against
    # the objective DE will actually use, otherwise the seed lands in a corner DE
    # only explores a few percent around, so ``run_nonlinear_optimizer`` turns
    # this on for its seed call.
    #
    # This changes only ``c``, never ``A_ub``/``b_ub``, so every constraint and
    # the exact-slag retry loop below behave identically either way.
    coke_correction_coeffs = np.zeros(n + n_flux, dtype=float)
    if price_coke_correction:
        coke_correction_coeffs = build_linear_coke_correction_cost_coeffs(
            ores=ores,
            variable_fluxes=variable_fluxes,
            settings=coke_correction_settings or CokeCorrectionSettings(),
            slag_coeff=slag_coeff,
            hot_metal_target_mt=float(
                hot_metal_target_mt or target_production_mt or 0.0
            ),
        )
        c = c + coke_correction_coeffs

    # Charging-throughput ceiling: every ore and flux tonne occupies room in the
    # same charges, so they share one budget. Without it the LP can meet the Fe
    # target from low-Fe material simply by charging more of it, which the plant
    # cannot do at capacity.
    if max_burden_qty_mt is not None and float(max_burden_qty_mt) > 0.0:
        fixed_flux_qty_mt = sum(
            max(0.0, float(flux.wet_qty_mt or 0.0))
            for flux in fixed_fluxes
            if flux.enabled
        )
        a_ub_rows.append(np.ones(n + n_flux, dtype=float))
        # Fixed flux is folded into the linear-model base and has no decision
        # variable, but it still travels through the charging system. Reserve
        # its tonnes from the shared ore + optimisable-flux budget.
        b_ub_values.append(float(max_burden_qty_mt) - fixed_flux_qty_mt)

    def _add_ratio_bounds(
        *,
        numerator_coeff: np.ndarray,
        numerator_base_mt: float,
        denominator_coeff: np.ndarray,
        denominator_base_mt: float,
        min_value: float | None,
        max_value: float | None,
    ) -> None:
        """
        Bound ``numerator / denominator`` between two limits as LP rows.

        Both sides are linear in x, so cross-multiplying keeps the constraint
        linear: ``num/den >= m`` becomes ``m*den - num <= 0``. This holds only
        while the denominator is positive, which it is for every quantity used
        here (SiO2 mass, total slag mass, Al2O3 mass) on any burden that makes
        the Fe target.
        """

        if min_value is not None:
            minimum = float(min_value)
            a_ub_rows.append(minimum * denominator_coeff - numerator_coeff)
            b_ub_values.append(numerator_base_mt - minimum * denominator_base_mt)

        if max_value is not None:
            maximum = float(max_value)
            a_ub_rows.append(numerator_coeff - maximum * denominator_coeff)
            b_ub_values.append(maximum * denominator_base_mt - numerator_base_mt)

    # CaO / SiO2
    _add_ratio_bounds(
        numerator_coeff=terms.basicity_numerator_coeff,
        numerator_base_mt=terms.basicity_numerator_base_mt,
        denominator_coeff=terms.basicity_denominator_coeff,
        denominator_base_mt=terms.basicity_denominator_base_mt,
        min_value=target_slag_basicity_min,
        max_value=target_slag_basicity_max,
    )
    # (CaO + MgO) / SiO2
    _add_ratio_bounds(
        numerator_coeff=terms.t_basicity_numerator_coeff,
        numerator_base_mt=terms.t_basicity_numerator_base_mt,
        denominator_coeff=terms.basicity_denominator_coeff,
        denominator_base_mt=terms.basicity_denominator_base_mt,
        min_value=target_slag_t_basicity_min,
        max_value=target_slag_t_basicity_max,
    )
    # Al2O3 % of slag. Expressed against total slag rather than as an absolute
    # mass because that is how the plant measures and controls it -- and because
    # it is the limit that binds when the slag rate is cut: Al2O3 is inert, so
    # its mass stays put while the denominator shrinks.
    _add_ratio_bounds(
        numerator_coeff=terms.al2o3_coeff,
        numerator_base_mt=terms.al2o3_base_mt,
        denominator_coeff=slag_coeff,
        denominator_base_mt=slag_base_mt,
        min_value=None,
        max_value=(
            float(target_slag_al2o3_max_pct) / 100.0
            if target_slag_al2o3_max_pct is not None
            else None
        ),
    )
    # MgO % of slag.
    _add_ratio_bounds(
        numerator_coeff=terms.mgo_coeff,
        numerator_base_mt=terms.mgo_base_mt,
        denominator_coeff=slag_coeff,
        denominator_base_mt=slag_base_mt,
        min_value=(
            float(target_slag_mgo_min_pct) / 100.0
            if target_slag_mgo_min_pct is not None
            else None
        ),
        max_value=None,
    )
    # MgO / Al2O3. Scale-free -- total slag cancels -- so this one constrains the
    # burden alone and cannot be satisfied by moving the slag rate.
    _add_ratio_bounds(
        numerator_coeff=terms.mgo_coeff,
        numerator_base_mt=terms.mgo_base_mt,
        denominator_coeff=terms.al2o3_coeff,
        denominator_base_mt=terms.al2o3_base_mt,
        min_value=target_slag_mgo_al2o3_ratio_min,
        max_value=None,
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
                    "LP infeasible or failed after exact slag tightening: " f"{err}"
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
                        target_slag_t_basicity_min=target_slag_t_basicity_min,
                        target_slag_t_basicity_max=target_slag_t_basicity_max,
                        target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
                        target_slag_mgo_min_pct=target_slag_mgo_min_pct,
                        target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
                        max_burden_qty_mt=max_burden_qty_mt,
                        fuel_ash_inputs=fuel_ash_inputs,
                        flux_inputs=flux_inputs,
                        dust_inputs=dust_inputs,
                        slag_balance_settings=slag_balance_settings,
                        hot_metal_target_mt=hot_metal_target_mt,
                        coke_correction_settings=coke_correction_settings,
                        coke_correction_reference=coke_correction_reference,
                        price_coke_correction=price_coke_correction,
                    )
                )
            return None, messages

        quantities = {ore.ore_id: float(result.x[idx]) for idx, ore in enumerate(ores)}
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
            charge_mass_mt=charge_mass_mt,
        )
        blend.diagnostics["lp_flux_quantities_mt"] = solved_flux_quantities
        # Cost of the flux the LP bought (per THM), on the same HM basis the blend
        # uses, so the displayed total cost includes optimizer-added flux.
        flux_cost_total_rs = sum(
            float(result.x[n + j]) * float(flux.price_rs_per_mt)
            for j, flux in enumerate(variable_fluxes)
        )
        flux_thm_basis = float(hot_metal_target_mt or blend.fe_production_mt or 0.0)
        blend.diagnostics["flux_cost_per_thm_rs"] = (
            float(flux_cost_total_rs / flux_thm_basis) if flux_thm_basis > 0.0 else 0.0
        )
        # What the LP actually priced, so the linear signal can be reconciled
        # against the clamped nonlinear correction the page reports.
        if float(np.abs(coke_correction_coeffs).max(initial=0.0)) > 0.0:
            blend.diagnostics["lp_coke_correction_linear_terms"] = {
                "coefficients_rs_per_wet_mt": {
                    column_id: float(coke_correction_coeffs[idx])
                    for idx, column_id in enumerate(
                        [ore.ore_id for ore in ores]
                        + [flux.flux_id for flux in variable_fluxes]
                    )
                },
                "cost_rs": float(
                    np.dot(coke_correction_coeffs, np.asarray(result.x, dtype=float))
                ),
            }
        violations = check_blend_constraints(
            blend,
            ores,
            target_production_mt=target_production_mt,
            target_slag_qty_mt=target_slag_qty_mt,
            target_slag_basicity_min=target_slag_basicity_min,
            target_slag_basicity_max=target_slag_basicity_max,
            target_slag_t_basicity_min=target_slag_t_basicity_min,
            target_slag_t_basicity_max=target_slag_t_basicity_max,
            target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
            target_slag_mgo_min_pct=target_slag_mgo_min_pct,
            target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
            max_burden_qty_mt=max_burden_qty_mt,
            slag_tolerance_mt=LP_EXACT_SLAG_TOLERANCE_MT,
        )
        if not violations:
            blend.feasible = True
            blend.violations = []
            blend.diagnostics["lp_exact_slag_tightening_mt"] = float(slag_tightening_mt)
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
