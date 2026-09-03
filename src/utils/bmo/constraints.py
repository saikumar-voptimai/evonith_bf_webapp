"""Constraint validation helpers for BMO blend candidates.

This module checks optimized blend outputs against target production, slag,
stock, and ore-share requirements. The same checks are used by LP and DE paths
so the UI reports constraint warnings consistently across optimization methods.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from math import isfinite
from typing import Any

import pandas as pd

from utils.bmo.types import BlendEvaluation, OreInput

# The furnace charges round the clock; this is not an operating choice.
CHARGING_HOURS_PER_DAY = 24.0

# Fallbacks mirroring ``setting_bmo.yml -> bmo.burden_capacity``; used when the
# config block is missing so the cap still reflects the plant's real ceiling.
_DEFAULT_BURDEN_CAPACITY = {
    "max_charges_per_hour": 7.5,
    "charge_mass_mt": 26.4,
}

# Plant operating set point. Nut coke is held here rather than optimised, so its
# tonnage follows directly from the hot-metal target.
DEFAULT_NUT_COKE_RATE_KG_PER_THM = 70.0

# HALF THE DISPLAY RESOLUTION OF THE VIOLATION MESSAGE, which prints MT to two
# decimals. Below this, the two numbers in the message round to the same text
# and the operator is told "Slag exceeds bound: 775.50 > 775.50 MT" - a message
# that cannot be acted on and that contradicts the solver, which had already
# declared the blend feasible.
#
# It is not float noise alone. The LP drives slag exactly onto the cap, and the
# page then RE-EVALUATES the displayed blend through a different path - fuel
# re-priced, slag recomputed on corrected fuel rates - so the value that reaches
# this check has drifted by more than an ulp. The LP's own post-solve check uses
# a tighter 1e-6 because at that point nothing has been re-evaluated yet.
#
# 0.005 MT is 5 kg of slag a day against a cap in the hundreds of tonnes, so it
# concedes nothing physically. The same reasoning already governs
# ``basicity_tolerance`` below - see its note about "0.940 < 0.940".
SLAG_DISPLAY_TOLERANCE_MT = 0.005


def calculate_wet_nut_coke_mt(
    base_rate_kg_per_thm: float,
    production_mt: float,
    moisture_pct: float = 0.0,
) -> float:
    """Return wet nut-coke MT after adding moisture to its base kg/THM rate.

    The plant's nut-coke rate is a base quantity. Moisture is added using the
    operator formula ``base + (base * moisture / 100)`` before converting kg
    to MT.
    """

    def _nonnegative(value: float, fallback: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return max(0.0, parsed) if isfinite(parsed) else fallback

    rate = _nonnegative(base_rate_kg_per_thm, DEFAULT_NUT_COKE_RATE_KG_PER_THM)
    production = _nonnegative(production_mt)
    moisture = min(100.0, _nonnegative(moisture_pct))
    base_quantity_kg = rate * production
    return float((base_quantity_kg + base_quantity_kg * moisture / 100.0) / 1000.0)


def max_ibrm_flux_capacity_mt(
    burden_capacity_cfg: Mapping[str, Any] | None = None,
    *,
    target_hot_metal_mt: float,
    nut_coke_rate_kg_per_thm: float = DEFAULT_NUT_COKE_RATE_KG_PER_THM,
    nut_coke_moisture_pct: float = 0.0,
) -> float:
    """
    Return the wet IBRM + flux tonnage the charging system can deliver in a day.

    The skips carry at most ``max_charges_per_hour`` charges of ``charge_mass_mt``
    each, around the clock. Nut coke rides in those same charges but is not an
    optimizer variable - it is pinned at a rate - so its tonnage is deducted off
    the top and the remainder is the room for IBRM (ore, sinter, pellet) plus
    flux::

        capacity = 7.5 * 26.4 * 24                = 4752 MT/day
        nut coke = (70 * 2350) * (1 + 8.9 / 100) / 1000 = 179 MT/day
        room     = 4752 - 179                          = 4573 MT/day

    This is an ABSOLUTE daily tonnage. It deliberately does not scale with the
    hot-metal target, because the charging system runs the same 24 hours no
    matter how much iron is being asked for. An earlier version expressed the
    ceiling as a per-MT-HM ratio taken against a 2350 t reference day, which
    understated the room by ~15% at a 2000 t target and, worse, allowed 5,076 MT
    at a 2600 t target - more than the skips can physically deliver. Only the
    nut-coke deduction moves with the target, and it does so directly.

    Without this cap the optimizer is free to answer a low-Fe burden by simply
    charging more of it. The plant cannot: charges are already at capacity, so
    more tonnes at a lower Fe% means holding the charge rate and losing yield.

    Args:
         - burden_capacity_cfg: Mapping[str, Any] | None - ``bmo.burden_capacity``
           config block. Missing keys fall back to the plant defaults above.
         - target_hot_metal_mt: float - Operator hot-metal target, used only to
           convert the nut-coke rate into tonnes.
         - nut_coke_rate_kg_per_thm: float - Nut coke set point in kg/THM.
         - nut_coke_moisture_pct: float - Moisture added to the base nut-coke
           quantity before its wet tonnes are deducted.

    Returns:
         - return float - Wet IBRM + flux MT available per day, or 0.0 when the
           configured numbers leave no positive room.
    """

    cfg = dict(_DEFAULT_BURDEN_CAPACITY)
    cfg.update(dict(burden_capacity_cfg or {}))

    def _value(key: str) -> float:
        try:
            return float(cfg.get(key, _DEFAULT_BURDEN_CAPACITY[key]))
        except (TypeError, ValueError):
            return float(_DEFAULT_BURDEN_CAPACITY[key])

    charge_capacity_mt = (
        _value("charge_mass_mt")
        * _value("max_charges_per_hour")
        * CHARGING_HOURS_PER_DAY
    )
    nut_coke_mt = calculate_wet_nut_coke_mt(
        nut_coke_rate_kg_per_thm,
        target_hot_metal_mt,
        nut_coke_moisture_pct,
    )

    return max(0.0, float(charge_capacity_mt - nut_coke_mt))


def check_blend_constraints(
    blend: BlendEvaluation,
    ores: list[OreInput],
    *,
    target_production_mt: float,
    target_slag_qty_mt: float,
    target_slag_basicity_min: float | None = None,
    target_slag_basicity_max: float | None = None,
    target_slag_t_basicity_min: float | None = None,
    target_slag_t_basicity_max: float | None = None,
    target_slag_al2o3_max_pct: float | None = None,
    target_slag_mgo_min_pct: float | None = None,
    target_slag_mgo_al2o3_ratio_min: float | None = None,
    max_burden_qty_mt: float | None = None,
    fe_tolerance_mt: float = 0.5,
    slag_tolerance_mt: float = SLAG_DISPLAY_TOLERANCE_MT,
    basicity_tolerance: float = 1e-3,
    slag_pct_tolerance: float = 1e-2,
    slag_ratio_tolerance: float = 1e-3,
    burden_tolerance_mt: float = 1e-6,
) -> list[str]:
    """
    Check a completed blend against BMO physical and planning constraints.

    LP and DE both call this helper after evaluating a candidate. Keeping the
    final validation shared ensures the UI reports stock, share, target hot
    metal, and slag violations the same way for every optimization path. Target
    hot metal is checked on both sides so DE cannot treat over-production as a
    feasible result.

    Args:
         - blend: BlendEvaluation - Evaluated blend to validate.
         - ores: list[OreInput] - Ore inputs used to produce the blend.
         - target_production_mt: float - Target hot-metal production in MT.
         - target_slag_qty_mt: float - Maximum allowed slag quantity in MT.
         - target_slag_basicity_min: float | None - Minimum CaO / SiO2 basicity.
         - target_slag_basicity_max: float | None - Maximum CaO / SiO2 basicity.
         - target_slag_t_basicity_min: float | None - Minimum (CaO + MgO) / SiO2 basicity.
         - target_slag_t_basicity_max: float | None - Maximum (CaO + MgO) / SiO2 basicity.
         - target_slag_al2o3_max_pct: float | None - Maximum Al2O3 percentage of slag.
           This is the limit that binds when the slag rate is cut: Al2O3 is inert, so
           its mass is conserved and its percentage rises as total slag falls.
         - target_slag_mgo_min_pct: float | None - Minimum MgO percentage of slag.
           Cutting slag rate relaxes this one, for the same concentration reason.
         - target_slag_mgo_al2o3_ratio_min: float | None - Minimum MgO / Al2O3 mass
           ratio. Scale-free: it does not move with total slag mass at all, so it is
           purely a statement about which materials are charged.
         - max_burden_qty_mt: float | None - Charging-throughput ceiling on total wet
           IBRM + flux in MT. ``None`` disables the check. See
           ``max_ibrm_flux_capacity_mt``.
         - fe_tolerance_mt: float - Allowed Fe production tolerance in MT.
         - slag_tolerance_mt: float - Allowed slag tolerance in MT. Defaults to
           ``SLAG_DISPLAY_TOLERANCE_MT``; see that constant for why it is not
           zero.
         - basicity_tolerance: float - Numeric tolerance for basicity bounds.
           The LP minimises cost, so it drives basicity onto the min (or max)
           bound exactly; the linearised LP basicity then differs from the exact
           re-evaluated basicity by a small drift. A physically meaningful 1e-3
           tolerance (~0.1% on a ~0.94 basicity, far finer than real slag control)
           absorbs that drift so a value equal to the bound at display precision
           is not flagged as "0.940 < 0.940".
         - slag_pct_tolerance: float - Numeric tolerance on the Al2O3/MgO percentage
           bounds, in percentage points. Same LP-drift reason as basicity: 0.01 pp is
           far below both display precision and real slag-analysis repeatability.
         - slag_ratio_tolerance: float - Numeric tolerance on the MgO/Al2O3 bound.

    Returns:
         - return list[str] - Human-readable constraint violation messages.
    """

    violations: list[str] = []

    production_value_mt = float(blend.fe_production_mt)
    production_target_mt = float(target_production_mt)
    production_label = "Fe production"
    if blend.diagnostics.get("iron_closure_basis") == "actual_pig_iron":
        production_value_mt = float(
            blend.diagnostics.get("iron_closure_production_mt", 0.0) or 0.0
        )
        production_target_mt = float(
            blend.diagnostics.get("iron_closure_target_mt", 0.0)
            or blend.diagnostics.get("hot_metal_target_mt", 0.0)
            or target_production_mt
        )
        production_label = "Chemical hot metal"

    if production_value_mt < production_target_mt - fe_tolerance_mt:
        violations.append(
            f"{production_label} below required target: "
            f"{production_value_mt:.2f} < {production_target_mt:.2f} MT."
        )
    if production_value_mt > production_target_mt + fe_tolerance_mt:
        violations.append(
            f"{production_label} above required target: "
            f"{production_value_mt:.2f} > {production_target_mt:.2f} MT."
        )

    if blend.slag_mt > target_slag_qty_mt + slag_tolerance_mt:
        violations.append(
            f"Slag exceeds bound: {blend.slag_mt:.2f} > {target_slag_qty_mt:.2f} MT."
        )

    if max_burden_qty_mt is not None and float(max_burden_qty_mt) > 0.0:
        burden_qty_mt = float(
            blend.diagnostics.get("total_burden_qty_mt", blend.total_qty_mt) or 0.0
        )
        if burden_qty_mt > float(max_burden_qty_mt) + burden_tolerance_mt:
            violations.append(
                f"Charging capacity exceeded: IBRM + flux {burden_qty_mt:,.2f} > "
                f"{float(max_burden_qty_mt):,.2f} MT. The burden needs more tonnes "
                "than the furnace can charge in a day; use higher-Fe material."
            )

    def _check_basicity(
        *,
        label: str,
        value: float,
        denominator_key: str,
        min_value: float | None,
        max_value: float | None,
    ) -> None:
        if min_value is None and max_value is None:
            return
        denominator = float(blend.diagnostics.get(denominator_key, 0.0) or 0.0)
        basicity = float(value or 0.0)
        if denominator <= 0.0 or not isfinite(basicity):
            violations.append(f"{label} unavailable: SiO2 denominator is zero.")
        else:
            if min_value is not None and basicity < min_value - basicity_tolerance:
                violations.append(
                    f"{label} below bound: {basicity:.3f} < {min_value:.3f}."
                )
            if max_value is not None and basicity > max_value + basicity_tolerance:
                violations.append(
                    f"{label} above bound: {basicity:.3f} > {max_value:.3f}."
                )

    _check_basicity(
        label="Slag basicity",
        value=float(getattr(blend, "slag_basicity", 0.0) or 0.0),
        denominator_key="slag_basicity_denominator_mt",
        min_value=(
            float(target_slag_basicity_min)
            if target_slag_basicity_min is not None
            else None
        ),
        max_value=(
            float(target_slag_basicity_max)
            if target_slag_basicity_max is not None
            else None
        ),
    )
    _check_basicity(
        label="Slag T Basicity",
        value=float(getattr(blend, "slag_t_basicity", 0.0) or 0.0),
        denominator_key="slag_t_basicity_denominator_mt",
        min_value=(
            float(target_slag_t_basicity_min)
            if target_slag_t_basicity_min is not None
            else None
        ),
        max_value=(
            float(target_slag_t_basicity_max)
            if target_slag_t_basicity_max is not None
            else None
        ),
    )

    def _check_slag_quality(
        *,
        label: str,
        value: float,
        denominator_key: str,
        unit: str,
        tolerance: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Validate one slag-quality ratio against its configured bound."""

        if min_value is None and max_value is None:
            return
        denominator = float(blend.diagnostics.get(denominator_key, 0.0) or 0.0)
        ratio = float(value or 0.0)
        if denominator <= 0.0 or not isfinite(ratio):
            violations.append(f"{label} unavailable: no slag mass to measure against.")
            return
        if min_value is not None and ratio < min_value - tolerance:
            violations.append(
                f"{label} below bound: {ratio:.3f}{unit} < {min_value:.3f}{unit}."
            )
        if max_value is not None and ratio > max_value + tolerance:
            violations.append(
                f"{label} above bound: {ratio:.3f}{unit} > {max_value:.3f}{unit}."
            )

    _check_slag_quality(
        label="Slag Al2O3",
        value=float(getattr(blend, "slag_al2o3_pct", 0.0) or 0.0),
        denominator_key="slag_chemistry_denominator_mt",
        unit="%",
        tolerance=slag_pct_tolerance,
        max_value=(
            float(target_slag_al2o3_max_pct)
            if target_slag_al2o3_max_pct is not None
            else None
        ),
    )
    _check_slag_quality(
        label="Slag MgO",
        value=float(getattr(blend, "slag_mgo_pct", 0.0) or 0.0),
        denominator_key="slag_chemistry_denominator_mt",
        unit="%",
        tolerance=slag_pct_tolerance,
        min_value=(
            float(target_slag_mgo_min_pct)
            if target_slag_mgo_min_pct is not None
            else None
        ),
    )
    _check_slag_quality(
        label="Slag MgO/Al2O3",
        value=float(getattr(blend, "slag_mgo_al2o3_ratio", 0.0) or 0.0),
        denominator_key="slag_mgo_al2o3_denominator_mt",
        unit="",
        tolerance=slag_ratio_tolerance,
        min_value=(
            float(target_slag_mgo_al2o3_ratio_min)
            if target_slag_mgo_al2o3_ratio_min is not None
            else None
        ),
    )

    for ore in ores:
        qty = float(blend.quantities_mt.get(ore.ore_id, 0.0))
        share_pct = float(blend.shares_pct.get(ore.ore_id, 0.0))

        if qty - float(ore.stock_mt) > 1e-6:
            violations.append(
                f"{ore.display_name}: quantity {qty:.2f} MT exceeds stock {ore.stock_mt:.2f} MT."
            )

        if share_pct + 1e-6 < float(ore.min_share_pct):
            violations.append(
                f"{ore.display_name}: share {share_pct:.2f}% below min {ore.min_share_pct:.2f}%."
            )

        if share_pct - 1e-6 > float(ore.max_share_pct):
            violations.append(
                f"{ore.display_name}: share {share_pct:.2f}% above max {ore.max_share_pct:.2f}%."
            )

    return violations


def validate_ore_bounds(ores: list[OreInput]) -> list[str]:
    """
    Validate ore share and stock bounds before running an optimizer.

    These checks catch impossible input limits before SciPy or HiGHS is called.
    They are intentionally limited to static ore properties that can be verified
    without solving the full blend problem.

    Args:
         - ores: list[OreInput] - Candidate ores with stock and share limits.

    Returns:
         - return list[str] - Pre-run infeasibility or input validation errors.
    """

    errors: list[str] = []
    min_sum = 0.0
    max_sum = 0.0

    for ore in ores:
        if ore.min_share_pct > ore.max_share_pct:
            errors.append(
                f"{ore.display_name}: min_share_pct ({ore.min_share_pct}) > max_share_pct ({ore.max_share_pct})."
            )
        if ore.stock_mt < 0:
            errors.append(f"{ore.display_name}: stock_mt cannot be negative.")
        if ore.price_rs_per_mt < 0:
            errors.append(f"{ore.display_name}: price_rs_per_mt cannot be negative.")

        min_sum += ore.min_share_pct / 100.0
        max_sum += ore.max_share_pct / 100.0

    if min_sum > 1.0 + 1e-6:
        errors.append(
            f"Sum of minimum shares is {min_sum * 100.0:.2f}% (must be <= 100%)."
        )
    if max_sum < 1.0 - 1e-6:
        errors.append(
            f"Sum of maximum shares is {max_sum * 100.0:.2f}% (must be >= 100%)."
        )
    if all(float(ore.stock_mt) <= 0.0 for ore in ores):
        errors.append("At least one selected ore must have positive stock.")

    return errors


def validate_selected_pellet_inputs(
    ores: list[OreInput],
    *,
    max_chemistry_age_days: int = 30,
    now: datetime | None = None,
) -> list[str]:
    """
    Return blocking pre-run issues for selected pellet materials.

    Pellet can materially change Fe, slag, and fuel-model inputs. If stock or
    chemistry is not backed by fresh source data, the operator must explicitly
    review the editor values before running BMO.
    """

    issues: list[str] = []
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    for ore in ores:
        material_key = str(ore.metadata.get("material_key", "")).lower()
        if "pellet" not in material_key and "pellet" not in ore.display_name.lower():
            continue

        stock_source = str(ore.metadata.get("stock_source", ""))
        if float(ore.stock_mt or 0.0) <= 0.0:
            issues.append(
                f"{ore.display_name}: enter positive pellet stock before running."
            )
        elif stock_source != "offline_db":
            issues.append(
                f"{ore.display_name}: stock is not from raw_material_stock; confirm the editor stock value."
            )

        chemistry_source = str(ore.metadata.get("chemistry_source", ""))
        sample_timestamp = str(ore.metadata.get("chemistry_sample_timestamp", "") or "")
        if chemistry_source == "fallback":
            issues.append(
                f"{ore.display_name}: chemistry is using fallback values; confirm or edit chemistry before running."
            )
            continue
        if not sample_timestamp:
            issues.append(
                f"{ore.display_name}: chemistry timestamp is missing; confirm or edit chemistry before running."
            )
            continue

        sample_time = pd.to_datetime(sample_timestamp, utc=True, errors="coerce")
        if pd.isna(sample_time):
            issues.append(
                f"{ore.display_name}: chemistry timestamp is invalid; confirm or edit chemistry before running."
            )
            continue
        age_days = (pd.Timestamp(current) - sample_time).total_seconds() / 86400.0
        if age_days > float(max_chemistry_age_days):
            issues.append(
                f"{ore.display_name}: chemistry sample is {age_days:.0f} days old; confirm or edit chemistry before running."
            )

    return issues
