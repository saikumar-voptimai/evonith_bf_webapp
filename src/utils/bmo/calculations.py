"""Blend chemistry and cost calculations for BMO.

This module evaluates solved wet ore quantities by first converting each ore
to dry weight, then calculating dry-weight Fe contribution, slag-forming oxide
contribution, final Fe percent, slag rate, and Rs/THM economics for BMO results.
"""

from __future__ import annotations

from math import isfinite

from utils.bmo.slag_balance import calculate_full_slag_balance
from utils.bmo.types import (
    BlendEvaluation,
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceSettings,
)

FE_FROM_FEO_FACTOR = 55.845 / 71.844


def _safe_pct(value: float) -> float:
    """
    Normalize a percentage value into a safe numeric range.

    BMO chemistry values come from configuration, live chemistry rows, and user
    editable inputs, so this guard prevents negative or impossible percentages
    from leaking into dry-weight and Fe contribution math.

    Args:
         - value: float - Raw percentage value.

    Returns:
         - return float - Percentage clamped between 0 and 100.
    """

    try:
        pct = float(value or 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    # Mirrors slag_balance._safe_pct: cap at 99.0 so a moisture entry of 100
    # cannot zero out dry-weight Fe calculations.
    return min(max(pct, 0.0), 99.0)


def compute_dry_fraction(moisture_pct: float) -> float:
    """
    Calculate the dry fraction remaining after removing moisture.

    This implements the first part of the plant formula:
    ``DryWeight = WetWeight * (100 - Moisture%) / 100``. The returned fraction
    can be multiplied by any wet weight to get dry weight.

    Args:
         - moisture_pct: float - Moisture percentage in wet material.

    Returns:
         - return float - Dry fraction between 0 and 1.
    """

    return (100.0 - _safe_pct(moisture_pct)) / 100.0


def compute_dry_weight_mt(wet_weight_mt: float, moisture_pct: float) -> float:
    """
    Convert wet material quantity into dry material quantity.

    The optimizer still plans wet burden quantities because stock, share, and
    purchase values are tracked on that basis. Chemistry calculations use this
    helper to remove moisture before calculating Fe contribution.

    Args:
         - wet_weight_mt: float - Wet material quantity in MT.
         - moisture_pct: float - Moisture percentage in wet material.

    Returns:
         - return float - Dry material quantity in MT.
    """

    try:
        wet_weight = float(wet_weight_mt or 0.0)
    except (TypeError, ValueError):
        wet_weight = 0.0
    return max(0.0, wet_weight) * compute_dry_fraction(moisture_pct)


def compute_fe_contribution_mt(dry_weight_mt: float, fe_t_pct: float) -> float:
    """
    Calculate Fe contribution from dry material weight and Fe percentage.

    This is the second step of the final Fe formula. Each material contributes
    ``DryWeight * Fe% / 100`` MT of Fe, and the blend-level Fe production is the
    sum of these contributions.

    Args:
         - dry_weight_mt: float - Dry material quantity in MT.
         - fe_t_pct: float - Total Fe percentage on dry basis.

    Returns:
         - return float - Fe contribution in MT.
    """

    try:
        dry_weight = float(dry_weight_mt or 0.0)
    except (TypeError, ValueError):
        dry_weight = 0.0
    return max(0.0, dry_weight) * (_safe_pct(fe_t_pct) / 100.0)


def compute_slag_forming_oxides_pct(
    *,
    sio2_pct: float,
    al2o3_pct: float,
    cao_pct: float,
    mgo_pct: float,
    tio2_pct: float,
    mno_pct: float,
    na2o_pct: float = 0.0,
    k2o_pct: float = 0.0,
) -> float:
    """
    Sum the dry-basis oxides that report to slag.

    This follows the simplified plant formula and intentionally excludes Fe,
    moisture, LOI, and fuel volatiles. Each input is treated as an oxide
    percentage on dry material weight.

    Args:
         - sio2_pct: float - Dry-basis SiO2 percentage.
         - al2o3_pct: float - Dry-basis Al2O3 percentage.
         - cao_pct: float - Dry-basis CaO percentage.
         - mgo_pct: float - Dry-basis MgO percentage.
         - tio2_pct: float - Dry-basis TiO2 percentage.
         - mno_pct: float - Dry-basis MnO percentage.
         - na2o_pct: float - Dry-basis Na2O percentage.
         - k2o_pct: float - Dry-basis K2O percentage.

    Returns:
         - return float - Total slag-forming oxide percentage.
    """

    return float(
        _safe_pct(sio2_pct)
        + _safe_pct(al2o3_pct)
        + _safe_pct(cao_pct)
        + _safe_pct(mgo_pct)
        + _safe_pct(tio2_pct)
        + _safe_pct(mno_pct)
        + _safe_pct(na2o_pct)
        + _safe_pct(k2o_pct)
    )


def compute_slag_contribution_mt(
    dry_weight_mt: float,
    *,
    sio2_pct: float,
    al2o3_pct: float,
    cao_pct: float,
    mgo_pct: float,
    tio2_pct: float,
    mno_pct: float,
    na2o_pct: float = 0.0,
    k2o_pct: float = 0.0,
) -> float:
    """
    Calculate slag contribution from one dry material quantity.

    This implements ``DryWeight * slag-forming oxides / 100`` for one ore or
    burden material. Blend-level slag MT is the sum of this value across all
    selected materials.

    Args:
         - dry_weight_mt: float - Dry material quantity in MT.
         - sio2_pct: float - Dry-basis SiO2 percentage.
         - al2o3_pct: float - Dry-basis Al2O3 percentage.
         - cao_pct: float - Dry-basis CaO percentage.
         - mgo_pct: float - Dry-basis MgO percentage.
         - tio2_pct: float - Dry-basis TiO2 percentage.
         - mno_pct: float - Dry-basis MnO percentage.
         - na2o_pct: float - Dry-basis Na2O percentage.
         - k2o_pct: float - Dry-basis K2O percentage.

    Returns:
         - return float - Slag contribution in MT.
    """

    try:
        dry_weight = float(dry_weight_mt or 0.0)
    except (TypeError, ValueError):
        dry_weight = 0.0
    slag_pct = compute_slag_forming_oxides_pct(
        sio2_pct=sio2_pct,
        al2o3_pct=al2o3_pct,
        cao_pct=cao_pct,
        mgo_pct=mgo_pct,
        tio2_pct=tio2_pct,
        mno_pct=mno_pct,
        na2o_pct=na2o_pct,
        k2o_pct=k2o_pct,
    )
    return max(0.0, dry_weight) * (slag_pct / 100.0)


def compute_slag_basicity(numerator_mt: float, denominator_mt: float) -> float:
    """
    Calculate total slag basicity from oxide masses.

    Basicity ratios use slag oxide masses. A zero SiO2 denominator
    denominator means the value cannot be used as a reliable process constraint,
    so the helper returns 0.0 and the constraint checker reports it as
    unavailable when bounds are active.
    """

    try:
        numerator = float(numerator_mt or 0.0)
        denominator = float(denominator_mt or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_fuel_wet_weight_mt(rate_kg_per_thm: float, hot_metal_mt: float) -> float:
    """
    Convert a fuel rate into wet fuel quantity for a hot-metal quantity.

    Coke, nut coke, and PCI are commonly tracked as kg/THM. BMO uses final Fe
    production as its THM denominator, so the wet fuel quantity is
    ``rate_kg_per_thm * hot_metal_mt / 1000``.

    Args:
         - rate_kg_per_thm: float - Wet fuel rate in kg per THM.
         - hot_metal_mt: float - Hot-metal or Fe production quantity in MT.

    Returns:
         - return float - Wet fuel quantity in MT.
    """

    try:
        rate = float(rate_kg_per_thm or 0.0)
    except (TypeError, ValueError):
        rate = 0.0
    try:
        hot_metal = float(hot_metal_mt or 0.0)
    except (TypeError, ValueError):
        hot_metal = 0.0
    return max(0.0, rate) * max(0.0, hot_metal) / 1000.0


def compute_fuel_ash_slag_factor_per_thm(
    fuel_ash_inputs: list[FuelAshInput] | None,
) -> float:
    """
    Calculate fuel-ash slag MT generated per MT of hot metal.

    Each enabled fuel contributes ``rate kg/THM / 1000`` wet MT per MT hot
    metal. Moisture is removed, ash percentage is applied, then the configured
    ash oxide percentages are summed to estimate the slag-forming part of that
    ash. Fe2O3, S, and P are kept in the fuel input for traceability but are not
    included in this simplified oxide-sum slag term.

    Args:
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records from UI/config.

    Returns:
         - return float - Fuel ash slag contribution in MT per MT hot metal.
    """

    factor = 0.0
    for fuel in fuel_ash_inputs or []:
        if not fuel.enabled:
            continue
        wet_fuel_per_thm_mt = compute_fuel_wet_weight_mt(
            fuel.rate_kg_per_thm, hot_metal_mt=1.0
        )
        dry_fuel_per_thm_mt = wet_fuel_per_thm_mt * compute_dry_fraction(
            fuel.moisture_pct
        )
        ash_mt_per_thm = dry_fuel_per_thm_mt * (_safe_pct(fuel.ash_pct) / 100.0)
        ash_oxide_pct = compute_slag_forming_oxides_pct(
            sio2_pct=fuel.sio2_pct,
            al2o3_pct=fuel.al2o3_pct,
            cao_pct=fuel.cao_pct,
            mgo_pct=fuel.mgo_pct,
            tio2_pct=fuel.tio2_pct,
            mno_pct=0.0,
            na2o_pct=fuel.na2o_pct,
            k2o_pct=fuel.k2o_pct,
        )
        factor += ash_mt_per_thm * (ash_oxide_pct / 100.0)
    return float(factor)


def compute_fuel_ash_slag_contributions_mt(
    fuel_ash_inputs: list[FuelAshInput] | None,
    hot_metal_mt: float,
) -> dict[str, float]:
    """
    Calculate slag contribution from each enabled fuel ash input.

    This helper expands the same fuel-ash formula used by LP and DE into
    per-fuel diagnostics. It makes the displayed total slag auditable by
    showing coke, nut coke, and PCI ash contributions separately.

    Args:
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records from UI/config.
         - hot_metal_mt: float - Hot-metal or Fe production quantity in MT.

    Returns:
         - return dict[str, float] - Fuel ash slag MT keyed by fuel id.
    """

    contributions: dict[str, float] = {}
    for fuel in fuel_ash_inputs or []:
        if not fuel.enabled:
            contributions[fuel.fuel_id] = 0.0
            continue
        wet_fuel_mt = compute_fuel_wet_weight_mt(
            fuel.rate_kg_per_thm, hot_metal_mt=hot_metal_mt
        )
        dry_fuel_mt = compute_dry_weight_mt(wet_fuel_mt, fuel.moisture_pct)
        ash_mt = dry_fuel_mt * (_safe_pct(fuel.ash_pct) / 100.0)
        ash_oxide_pct = compute_slag_forming_oxides_pct(
            sio2_pct=fuel.sio2_pct,
            al2o3_pct=fuel.al2o3_pct,
            cao_pct=fuel.cao_pct,
            mgo_pct=fuel.mgo_pct,
            tio2_pct=fuel.tio2_pct,
            mno_pct=0.0,
            na2o_pct=fuel.na2o_pct,
            k2o_pct=fuel.k2o_pct,
        )
        contributions[fuel.fuel_id] = float(ash_mt * (ash_oxide_pct / 100.0))
    return contributions


def compute_flux_slag_contribution_mt(flux: FluxInput) -> float:
    """
    Calculate slag contribution from one fixed flux input.

    The simplified flux calculation uses the same dry-weight oxide-sum approach
    as ore slag. Moisture is removed first, then SiO2, Al2O3, CaO, and MgO are
    summed as slag-forming oxides. LOI and Fe2O3 are retained in the input for
    traceability but excluded from this simplified slag term.

    Args:
         - flux: FluxInput - Fixed flux quantity and chemistry row.

    Returns:
         - return float - Flux slag contribution in MT.
    """

    if not flux.enabled:
        return 0.0
    dry_flux_mt = compute_dry_weight_mt(flux.wet_qty_mt, flux.moisture_pct)
    return compute_slag_contribution_mt(
        dry_flux_mt,
        sio2_pct=flux.sio2_pct,
        al2o3_pct=flux.al2o3_pct,
        cao_pct=flux.cao_pct,
        mgo_pct=flux.mgo_pct,
        tio2_pct=0.0,
        mno_pct=0.0,
        na2o_pct=0.0,
        k2o_pct=0.0,
    )


def compute_flux_slag_contributions_mt(
    flux_inputs: list[FluxInput] | None,
) -> dict[str, float]:
    """
    Calculate slag contribution from each configured flux input.

    This helper provides auditable per-flux diagnostics for Limestone, Dolomite,
    Quartz, or any future fixed flux row. Disabled rows are kept with zero value
    so the diagnostics match the UI table.

    Args:
         - flux_inputs: list[FluxInput] | None - Flux records from UI/config.

    Returns:
         - return dict[str, float] - Flux slag MT keyed by flux id.
    """

    return {
        flux.flux_id: float(compute_flux_slag_contribution_mt(flux))
        for flux in flux_inputs or []
    }


def compute_flux_dry_weights_mt(
    flux_inputs: list[FluxInput] | None,
) -> dict[str, float]:
    """
    Calculate dry quantity for each configured flux input.

    The dry weights are exposed in diagnostics so operators can verify the
    wet-to-dry conversion used before oxide percentages are applied.

    Args:
         - flux_inputs: list[FluxInput] | None - Flux records from UI/config.

    Returns:
         - return dict[str, float] - Dry flux quantities keyed by flux id.
    """

    dry_weights: dict[str, float] = {}
    for flux in flux_inputs or []:
        dry_weights[flux.flux_id] = (
            compute_dry_weight_mt(flux.wet_qty_mt, flux.moisture_pct)
            if flux.enabled
            else 0.0
        )
    return dry_weights


def _dry_quantities_by_ore(
    ores: list[OreInput], quantities: dict[str, float]
) -> dict[str, float]:
    """
    Convert wet ore quantities into dry quantities by ore id.

    The returned dictionary keeps the original ore ids so downstream diagnostics
    and UI tables can show how each wet quantity changes after moisture removal.

    Args:
         - ores: list[OreInput] - Ores included in the evaluated blend.
         - quantities: dict[str, float] - Wet quantities keyed by ore id.

    Returns:
         - return dict[str, float] - Dry quantities keyed by ore id.
    """

    return {
        ore.ore_id: compute_dry_weight_mt(
            quantities.get(ore.ore_id, 0.0), ore.chemistry.moisture_pct
        )
        for ore in ores
    }


def _weighted_avg(ores: list[OreInput], weights: dict[str, float], attr: str) -> float:
    """
    Calculate a weighted chemistry average using supplied ore weights.

    For BMO chemistry metrics this should usually receive dry weights, because
    oxide and Fe percentages are interpreted after moisture is removed.

    Args:
         - ores: list[OreInput] - Ores included in the evaluated blend.
         - weights: dict[str, float] - Weights keyed by ore id.
         - attr: str - OreChemistry attribute to average.

    Returns:
         - return float - Weighted average chemistry value.
    """

    total_qty = sum(weights.values())
    if total_qty <= 0:
        return 0.0
    acc = 0.0
    for ore in ores:
        qty = float(weights.get(ore.ore_id, 0.0))
        if qty <= 0:
            continue
        value = float(getattr(ore.chemistry, attr, 0.0) or 0.0)
        acc += (qty / total_qty) * value
    return acc


def compute_charging_requirements(
    blend: BlendEvaluation,
    *,
    charge_mass_mt: float = 26.4,
    hours_per_day: float = 24.0,
) -> dict[str, float | None]:
    """Calculate fuel quantities and charging rates for a solved blend.

    The charging system carries wet IBRM, wet flux, and nut coke. Nut coke is
    stored as kg/THM in the fuel-rate diagnostics, so its daily tonnes are
    ``nut_coke_kg_thm * hot_metal_mt / 1000``. Required charges per hour are:

    ``(IBRM + flux + nut coke) / hours_per_day / charge_mass_mt``.

    Coke and PCI tonnes use the same kg/THM-to-MT conversion. Hot metal per
    charge is the production target divided by the number of daily charges.

    Args:
         - blend: BlendEvaluation - Evaluated LP, DE, or manual blend.
         - charge_mass_mt: float - Tonnes carried by one furnace charge.
         - hours_per_day: float - Charging hours represented by the blend target.

    Returns:
         - return dict[str, float | None] - Flux kg/THM, fuel tonnes, total
           charge-mix tonnes, tonnes/hour, required charges/hour, and hot metal
           per charge. Rate-dependent results are ``None`` when unavailable.
    """

    diagnostics = blend.diagnostics or {}
    hot_metal_mt = float(diagnostics.get("hot_metal_target_mt", 0.0) or 0.0)
    flux_mt = max(0.0, float(diagnostics.get("total_flux_wet_qty_mt", 0.0) or 0.0))
    burden_mt = max(
        0.0,
        float(diagnostics.get("total_burden_qty_mt", blend.total_qty_mt) or 0.0),
    )
    flux_rate_kg_thm = (
        flux_mt * 1000.0 / hot_metal_mt if hot_metal_mt > 0.0 else None
    )

    fuel_rates = diagnostics.get("fuel_rate_estimate") or {}

    def _nonnegative_rate(key: str) -> float | None:
        raw_value = fuel_rates.get(key) if isinstance(fuel_rates, dict) else None
        try:
            return max(0.0, float(raw_value)) if raw_value is not None else None
        except (TypeError, ValueError):
            return None

    coke_rate_kg_thm = _nonnegative_rate("coke_rate_kg_thm")
    nut_coke_rate_kg_thm = _nonnegative_rate("nut_coke_rate_kg_thm")
    pci_rate_kg_thm = _nonnegative_rate("pci_rate_kg_thm")

    coke_total_mt: float | None = None
    nut_coke_total_mt: float | None = None
    pci_total_mt: float | None = None
    total_charge_mix_mt: float | None = None
    charge_mix_mt_per_hour: float | None = None
    required_charges_per_hour: float | None = None
    hot_metal_per_charge_mt: float | None = None
    if coke_rate_kg_thm is not None and hot_metal_mt > 0.0:
        coke_total_mt = coke_rate_kg_thm * hot_metal_mt / 1000.0
    if pci_rate_kg_thm is not None and hot_metal_mt > 0.0:
        pci_total_mt = pci_rate_kg_thm * hot_metal_mt / 1000.0
    if nut_coke_rate_kg_thm is not None and hot_metal_mt > 0.0:
        nut_coke_total_mt = nut_coke_rate_kg_thm * hot_metal_mt / 1000.0
        total_charge_mix_mt = burden_mt + nut_coke_total_mt
        if float(hours_per_day) > 0.0:
            charge_mix_mt_per_hour = total_charge_mix_mt / float(hours_per_day)
            if float(charge_mass_mt) > 0.0:
                required_charges_per_hour = (
                    charge_mix_mt_per_hour / float(charge_mass_mt)
                )
                if required_charges_per_hour > 0.0:
                    hot_metal_per_charge_mt = hot_metal_mt / (
                        required_charges_per_hour * float(hours_per_day)
                    )

    return {
        "total_flux_wet_qty_mt": float(flux_mt),
        "flux_rate_kg_per_thm": flux_rate_kg_thm,
        "total_burden_qty_mt": float(burden_mt),
        "coke_rate_kg_per_thm": coke_rate_kg_thm,
        "coke_total_mt": coke_total_mt,
        "nut_coke_rate_kg_per_thm": nut_coke_rate_kg_thm,
        "nut_coke_total_mt": nut_coke_total_mt,
        "pci_rate_kg_per_thm": pci_rate_kg_thm,
        "pci_total_mt": pci_total_mt,
        "total_charge_mix_mt": total_charge_mix_mt,
        "charge_mix_mt_per_hour": charge_mix_mt_per_hour,
        "required_charges_per_hour": required_charges_per_hour,
        "hot_metal_per_charge_mt": hot_metal_per_charge_mt,
    }


def compute_effective_fe_pct(
    fe_t_pct: float, feo_pct: float, feo_in_slag_pct: float
) -> float:
    """
    Calculate FeO-adjusted Fe percentage retained for diagnostics.

    Final Fe production now follows the dry-weight Fe contribution formula.
    This helper is kept as a separate reported metric so existing users can
    still inspect the FeO-adjusted value without changing the final Fe formula.

    Args:
         - fe_t_pct: float - Final dry-weight Fe percentage.
         - feo_pct: float - Dry-weight FeO percentage.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.

    Returns:
         - return float - FeO-adjusted Fe percentage.
    """

    return float(fe_t_pct + (feo_pct - feo_in_slag_pct) * FE_FROM_FEO_FACTOR)


def evaluate_blend(
    ores: list[OreInput],
    quantities_mt: dict[str, float],
    feo_in_slag_pct: float,
    fuel_cost_per_thm_rs: float = 0.0,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    hot_metal_target_mt: float | None = None,
) -> BlendEvaluation:
    """
    Evaluate one BMO blend using dry-weight Fe calculation.

    This is the central BMO calculation path. It starts with optimized wet
    quantities, converts each material to dry weight, calculates Fe MT per
    material, and then derives final Fe% as ``total Fe MT / total dry MT * 100``.
    Slag MT is calculated separately as the sum of dry-weight ore oxide
    contributions from SiO2, Al2O3, CaO, MgO, TiO2, MnO, Na2O, and K2O.
    When fuel ash or flux inputs are provided, their dry-basis slag-forming
    oxides are added to the simplified slag estimate. If full slag-balance
    settings are enabled, final slag MT is replaced with the full BF component
    balance. Slag contribution diagnostics are scaled by the configured
    correction factor for display, while raw uncorrected values remain in
    ``raw_*`` diagnostics for audit.

    Args:
         - ores: list[OreInput] - Ores included in the blend.
         - quantities_mt: dict[str, float] - Wet ore quantities keyed by ore id.
         - feo_in_slag_pct: float - FeO percentage assumed to report into slag.
         - fuel_cost_per_thm_rs: float - Fuel cost to include in total objective.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash records used for slag.
         - flux_inputs: list[FluxInput] | None - Fixed flux records used for slag.
         - dust_inputs: list[DustInput] | None - BF gas dust rows deducted from balance.
         - slag_balance_settings: SlagBalanceSettings | None - Full balance settings.
         - hot_metal_target_mt: float | None - Operator-entered HM basis for Rs/THM and kg/THM.

    Returns:
         - return BlendEvaluation - Blend cost, chemistry, production, and diagnostics.
    """

    total_qty_mt = float(sum(quantities_mt.values()))
    shares_pct: dict[str, float] = {}

    if total_qty_mt > 0:
        shares_pct = {
            ore_id: (qty / total_qty_mt) * 100.0
            for ore_id, qty in quantities_mt.items()
        }
    else:
        shares_pct = {ore_id: 0.0 for ore_id in quantities_mt}

    dry_quantities_mt = _dry_quantities_by_ore(ores, quantities_mt)
    total_dry_qty_mt = float(sum(dry_quantities_mt.values()))
    fe_contribution_mt_by_ore = {
        ore.ore_id: compute_fe_contribution_mt(
            dry_quantities_mt.get(ore.ore_id, 0.0), ore.chemistry.fe_t_pct
        )
        for ore in ores
    }
    fe_production_mt = float(sum(fe_contribution_mt_by_ore.values()))
    hm_basis_mt = float(hot_metal_target_mt or 0.0)
    if hm_basis_mt <= 0.0:
        hm_basis_mt = fe_production_mt

    fe_t_pct = (
        (fe_production_mt / total_dry_qty_mt) * 100.0 if total_dry_qty_mt > 0 else 0.0
    )
    feo_pct = _weighted_avg(ores, dry_quantities_mt, "feo_pct")
    sio2_pct = _weighted_avg(ores, dry_quantities_mt, "sio2_pct")
    al2o3_pct = _weighted_avg(ores, dry_quantities_mt, "al2o3_pct")
    cao_pct = _weighted_avg(ores, dry_quantities_mt, "cao_pct")
    mgo_pct = _weighted_avg(ores, dry_quantities_mt, "mgo_pct")
    mno_pct = _weighted_avg(ores, dry_quantities_mt, "mno_pct")
    tio2_pct = _weighted_avg(ores, dry_quantities_mt, "tio2_pct")
    na2o_pct = _weighted_avg(ores, dry_quantities_mt, "na2o_pct")
    k2o_pct = _weighted_avg(ores, dry_quantities_mt, "k2o_pct")
    slag_contribution_mt_by_ore = {
        ore.ore_id: compute_slag_contribution_mt(
            dry_quantities_mt.get(ore.ore_id, 0.0),
            sio2_pct=ore.chemistry.sio2_pct,
            al2o3_pct=ore.chemistry.al2o3_pct,
            cao_pct=ore.chemistry.cao_pct,
            mgo_pct=ore.chemistry.mgo_pct,
            tio2_pct=ore.chemistry.tio2_pct,
            mno_pct=ore.chemistry.mno_pct,
            na2o_pct=ore.chemistry.na2o_pct,
            k2o_pct=ore.chemistry.k2o_pct,
        )
        for ore in ores
    }

    effective_fe_pct = compute_effective_fe_pct(fe_t_pct, feo_pct, feo_in_slag_pct)
    slag_pct = compute_slag_forming_oxides_pct(
        sio2_pct=sio2_pct,
        al2o3_pct=al2o3_pct,
        cao_pct=cao_pct,
        mgo_pct=mgo_pct,
        tio2_pct=tio2_pct,
        mno_pct=mno_pct,
        na2o_pct=na2o_pct,
        k2o_pct=k2o_pct,
    )

    ore_slag_mt = float(sum(slag_contribution_mt_by_ore.values()))
    fuel_ash_contribution_mt_by_fuel = compute_fuel_ash_slag_contributions_mt(
        fuel_ash_inputs, hot_metal_mt=hm_basis_mt
    )
    fuel_ash_slag_mt = float(sum(fuel_ash_contribution_mt_by_fuel.values()))
    flux_contribution_mt_by_flux = compute_flux_slag_contributions_mt(flux_inputs)
    flux_slag_mt = float(sum(flux_contribution_mt_by_flux.values()))
    total_flux_wet_qty_mt = float(
        sum(
            max(0.0, float(flux.wet_qty_mt or 0.0))
            for flux in flux_inputs or []
            if flux.enabled
        )
    )
    basicity_cao_mt = total_dry_qty_mt * (_safe_pct(cao_pct) / 100.0)
    basicity_mgo_mt = total_dry_qty_mt * (_safe_pct(mgo_pct) / 100.0)
    basicity_sio2_mt = total_dry_qty_mt * (_safe_pct(sio2_pct) / 100.0)
    # Al2O3 is inert: none of it is reduced or volatilised, so every tonne
    # charged reports to slag. It is tracked alongside the basicity components
    # because the slag-quality limits (Al2O3 cap, MgO floor, MgO/Al2O3 floor)
    # need its mass on exactly the same basis.
    slag_al2o3_mt = total_dry_qty_mt * (_safe_pct(al2o3_pct) / 100.0)
    for fuel in fuel_ash_inputs or []:
        if not fuel.enabled:
            continue
        wet_fuel_mt = compute_fuel_wet_weight_mt(
            fuel.rate_kg_per_thm, hot_metal_mt=hm_basis_mt
        )
        dry_fuel_mt = compute_dry_weight_mt(wet_fuel_mt, fuel.moisture_pct)
        ash_mt = dry_fuel_mt * (_safe_pct(fuel.ash_pct) / 100.0)
        basicity_cao_mt += ash_mt * (_safe_pct(fuel.cao_pct) / 100.0)
        basicity_mgo_mt += ash_mt * (_safe_pct(fuel.mgo_pct) / 100.0)
        basicity_sio2_mt += ash_mt * (_safe_pct(fuel.sio2_pct) / 100.0)
        slag_al2o3_mt += ash_mt * (_safe_pct(fuel.al2o3_pct) / 100.0)
    for flux in flux_inputs or []:
        if not flux.enabled:
            continue
        dry_flux_mt = compute_dry_weight_mt(flux.wet_qty_mt, flux.moisture_pct)
        basicity_cao_mt += dry_flux_mt * (_safe_pct(flux.cao_pct) / 100.0)
        basicity_mgo_mt += dry_flux_mt * (_safe_pct(flux.mgo_pct) / 100.0)
        basicity_sio2_mt += dry_flux_mt * (_safe_pct(flux.sio2_pct) / 100.0)
        slag_al2o3_mt += dry_flux_mt * (_safe_pct(flux.al2o3_pct) / 100.0)
    slag_basicity_source = "simplified_ore_fuel_flux"
    simplified_slag_mt = ore_slag_mt + fuel_ash_slag_mt + flux_slag_mt
    slag_mt = simplified_slag_mt
    full_slag_balance = None
    if slag_balance_settings is not None and slag_balance_settings.enabled:
        full_slag_balance = calculate_full_slag_balance(
            ores=ores,
            quantities_mt=quantities_mt,
            hot_metal_mt=hm_basis_mt,
            settings=slag_balance_settings,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
        )
        slag_mt = full_slag_balance.total_slag_mt
        slag_components = full_slag_balance.slag_components_mt
        basicity_cao_mt = float(slag_components.get("cao", 0.0))
        basicity_mgo_mt = float(slag_components.get("mgo", 0.0))
        basicity_sio2_mt = float(slag_components.get("sio2", 0.0))
        slag_al2o3_mt = float(slag_components.get("al2o3", 0.0))
        slag_basicity_source = "full_slag_balance"
    slag_basicity = compute_slag_basicity(basicity_cao_mt, basicity_sio2_mt)
    slag_t_basicity = compute_slag_basicity(
        basicity_cao_mt + basicity_mgo_mt, basicity_sio2_mt
    )
    # Percentages of FINAL slag, so they are directly comparable to the plant's
    # own slag analysis. The ratio is deliberately taken on masses rather than on
    # the two percentages: it is then independent of total slag mass, and stays
    # meaningful even if the slag total is uncertain.
    slag_al2o3_pct = (slag_al2o3_mt / slag_mt) * 100.0 if slag_mt > 0.0 else 0.0
    slag_mgo_pct = (basicity_mgo_mt / slag_mt) * 100.0 if slag_mt > 0.0 else 0.0
    slag_mgo_al2o3_ratio = (
        basicity_mgo_mt / slag_al2o3_mt if slag_al2o3_mt > 0.0 else 0.0
    )
    slag_source_correction_factor = 1.0
    if full_slag_balance is not None:
        slag_source_correction_factor = float(
            full_slag_balance.diagnostics.get("slag_correction_factor", 1.0)
        )
    corrected_slag_contribution_mt_by_ore = {
        ore_id: float(value * slag_source_correction_factor)
        for ore_id, value in slag_contribution_mt_by_ore.items()
    }
    corrected_fuel_ash_contribution_mt_by_fuel = {
        fuel_id: float(value * slag_source_correction_factor)
        for fuel_id, value in fuel_ash_contribution_mt_by_fuel.items()
    }
    corrected_flux_contribution_mt_by_flux = {
        flux_id: float(value * slag_source_correction_factor)
        for flux_id, value in flux_contribution_mt_by_flux.items()
    }
    corrected_ore_slag_mt = ore_slag_mt * slag_source_correction_factor
    corrected_fuel_ash_slag_mt = fuel_ash_slag_mt * slag_source_correction_factor
    corrected_flux_slag_mt = flux_slag_mt * slag_source_correction_factor
    hm_reduction_sio2_mt = 0.0
    hm_reduction_mno_mt = 0.0
    hm_reduction_tio2_mt = 0.0
    hm_reduction_s_mt = 0.0
    hm_reduction_alkali_mt = 0.0
    tio2_unaccounted_mt = 0.0
    raw_ore_components_total_mt = 0.0
    raw_flux_components_total_mt = 0.0
    raw_fuel_ash_components_total_mt = 0.0
    if full_slag_balance is not None:
        fb_diag = full_slag_balance.diagnostics
        # Replace simplified source totals with post-HM-reduction attributions
        # so each tile reflects the source's true contribution to final slag.
        corrected_ore_slag_mt = float(fb_diag.get("ore_slag_in_final_mt", 0.0))
        corrected_fuel_ash_slag_mt = float(
            fb_diag.get("fuel_ash_slag_in_final_mt", 0.0)
        )
        corrected_flux_slag_mt = float(fb_diag.get("flux_slag_in_final_mt", 0.0))
        hm_reduction_sio2_mt = float(fb_diag.get("hm_reduction_sio2_mt", 0.0))
        hm_reduction_mno_mt = float(fb_diag.get("hm_reduction_mno_mt", 0.0))
        hm_reduction_tio2_mt = float(fb_diag.get("hm_reduction_tio2_mt", 0.0))
        hm_reduction_s_mt = float(fb_diag.get("hm_reduction_s_mt", 0.0))
        hm_reduction_alkali_mt = float(fb_diag.get("hm_reduction_alkali_mt", 0.0))
        tio2_unaccounted_mt = float(fb_diag.get("tio2_unaccounted_mt", 0.0))
    if total_dry_qty_mt > 0:
        slag_pct = (slag_mt / total_dry_qty_mt) * 100.0
    slag_rate_kg_per_thm = (
        (slag_mt / hm_basis_mt) * 1000.0 if hm_basis_mt > 0 else 0.0
    )

    ore_cost_total_rs = 0.0
    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0))
        ore_cost_total_rs += qty * float(ore.price_rs_per_mt)

    if hm_basis_mt > 0 and isfinite(hm_basis_mt):
        ore_cost_per_thm_rs = ore_cost_total_rs / hm_basis_mt
    else:
        ore_cost_per_thm_rs = float("inf")

    objective_rs_per_thm = ore_cost_per_thm_rs + float(fuel_cost_per_thm_rs)

    return BlendEvaluation(
        quantities_mt=quantities_mt,
        shares_pct=shares_pct,
        total_qty_mt=total_qty_mt,
        ore_cost_total_rs=float(ore_cost_total_rs),
        ore_cost_per_thm_rs=float(ore_cost_per_thm_rs),
        fuel_cost_per_thm_rs=float(fuel_cost_per_thm_rs),
        objective_rs_per_thm=float(objective_rs_per_thm),
        fe_t_pct=float(fe_t_pct),
        effective_fe_pct=float(effective_fe_pct),
        fe_production_mt=float(fe_production_mt),
        slag_pct=float(slag_pct),
        slag_mt=float(slag_mt),
        feasible=True,
        violations=[],
        slag_rate_kg_per_thm=float(slag_rate_kg_per_thm),
        slag_basicity=float(slag_basicity),
        slag_t_basicity=float(slag_t_basicity),
        slag_al2o3_pct=float(slag_al2o3_pct),
        slag_mgo_pct=float(slag_mgo_pct),
        slag_mgo_al2o3_ratio=float(slag_mgo_al2o3_ratio),
        diagnostics={
            "formula": "dry_weight_fe_and_ore_fuel_ash_flux_slag",
            "total_dry_qty_mt": float(total_dry_qty_mt),
            # Wet tonnes actually going through the charging system: IBRM plus
            # flux, whether that flux is a fixed addition or one the optimizer
            # bought. This is what the charging-throughput cap applies to.
            "total_flux_wet_qty_mt": float(total_flux_wet_qty_mt),
            "total_burden_qty_mt": float(total_qty_mt + total_flux_wet_qty_mt),
            "dry_weight_mt_by_ore": {
                ore_id: float(value) for ore_id, value in dry_quantities_mt.items()
            },
            "fe_contribution_mt_by_ore": {
                ore_id: float(value)
                for ore_id, value in fe_contribution_mt_by_ore.items()
            },
            "moisture_pct_by_ore": {
                ore.ore_id: float(ore.chemistry.moisture_pct) for ore in ores
            },
            "slag_contribution_mt_by_ore": {
                ore_id: float(value)
                for ore_id, value in corrected_slag_contribution_mt_by_ore.items()
            },
            "raw_slag_contribution_mt_by_ore": {
                ore_id: float(value)
                for ore_id, value in slag_contribution_mt_by_ore.items()
            },
            "ore_slag_mt": float(corrected_ore_slag_mt),
            "raw_ore_slag_mt": float(ore_slag_mt),
            "fuel_ash_slag_mt": float(corrected_fuel_ash_slag_mt),
            "raw_fuel_ash_slag_mt": float(fuel_ash_slag_mt),
            "fuel_ash_contribution_mt_by_fuel": {
                fuel_id: float(value)
                for fuel_id, value in corrected_fuel_ash_contribution_mt_by_fuel.items()
            },
            "raw_fuel_ash_contribution_mt_by_fuel": {
                fuel_id: float(value)
                for fuel_id, value in fuel_ash_contribution_mt_by_fuel.items()
            },
            "fuel_ash_slag_factor_per_thm": float(
                compute_fuel_ash_slag_factor_per_thm(fuel_ash_inputs)
            ),
            "flux_slag_mt": float(corrected_flux_slag_mt),
            "raw_flux_slag_mt": float(flux_slag_mt),
            "flux_contribution_mt_by_flux": {
                flux_id: float(value)
                for flux_id, value in corrected_flux_contribution_mt_by_flux.items()
            },
            "raw_flux_contribution_mt_by_flux": {
                flux_id: float(value)
                for flux_id, value in flux_contribution_mt_by_flux.items()
            },
            "flux_dry_weight_mt_by_flux": {
                flux_id: float(value)
                for flux_id, value in compute_flux_dry_weights_mt(flux_inputs).items()
            },
            "simplified_slag_mt": float(simplified_slag_mt),
            "slag_source_correction_factor": float(slag_source_correction_factor),
            "slag_balance_enabled": bool(
                slag_balance_settings.enabled if slag_balance_settings else False
            ),
            "hm_reduction_sio2_mt": float(hm_reduction_sio2_mt),
            "hm_reduction_mno_mt": float(hm_reduction_mno_mt),
            "hm_reduction_tio2_mt": float(hm_reduction_tio2_mt),
            "hm_reduction_s_mt": float(hm_reduction_s_mt),
            "hm_reduction_alkali_mt": float(hm_reduction_alkali_mt),
            "tio2_unaccounted_mt": float(tio2_unaccounted_mt),
            "full_slag_balance": (
                {
                    "total_slag_mt": float(full_slag_balance.total_slag_mt),
                    "actual_pig_iron_mt": float(full_slag_balance.actual_pig_iron_mt),
                    "theoretical_pig_iron_mt": float(
                        full_slag_balance.theoretical_pig_iron_mt
                    ),
                    "ore_components_mt": full_slag_balance.ore_components_mt,
                    "flux_components_mt": full_slag_balance.flux_components_mt,
                    "fuel_ash_components_mt": (
                        full_slag_balance.fuel_ash_components_mt
                    ),
                    "dust_components_mt": full_slag_balance.dust_components_mt,
                    "total_into_bf_mt": full_slag_balance.total_into_bf_mt,
                    "net_into_bf_mt": full_slag_balance.net_into_bf_mt,
                    "slag_components_mt": full_slag_balance.slag_components_mt,
                    "diagnostics": full_slag_balance.diagnostics,
                }
                if full_slag_balance is not None
                else {}
            ),
            "slag_rate_kg_per_thm": float(slag_rate_kg_per_thm),
            "slag_rate_denominator_mt": float(hm_basis_mt),
            "slag_rate_denominator": "hot_metal_target_mt",
            "hot_metal_target_mt": float(hm_basis_mt),
            "slag_basicity": float(slag_basicity),
            "slag_t_basicity": float(slag_t_basicity),
            "slag_basicity_numerator_mt": float(basicity_cao_mt),
            "slag_basicity_denominator_mt": float(basicity_sio2_mt),
            "slag_t_basicity_numerator_mt": float(basicity_cao_mt + basicity_mgo_mt),
            "slag_t_basicity_denominator_mt": float(basicity_sio2_mt),
            "slag_basicity_cao_mt": float(basicity_cao_mt),
            "slag_basicity_mgo_mt": float(basicity_mgo_mt),
            "slag_basicity_sio2_mt": float(basicity_sio2_mt),
            # Slag-quality limits. The Al2O3 cap and MgO floor are ratios against
            # total slag, so they are validated with slag_mt as the denominator;
            # the MgO/Al2O3 floor uses the Al2O3 mass instead.
            "slag_al2o3_mt": float(slag_al2o3_mt),
            "slag_mgo_mt": float(basicity_mgo_mt),
            "slag_al2o3_pct": float(slag_al2o3_pct),
            "slag_mgo_pct": float(slag_mgo_pct),
            "slag_mgo_al2o3_ratio": float(slag_mgo_al2o3_ratio),
            "slag_chemistry_denominator_mt": float(slag_mt),
            "slag_mgo_al2o3_denominator_mt": float(slag_al2o3_mt),
            "slag_chemistry_source": str(slag_basicity_source),
            "slag_basicity_source": slag_basicity_source,
        },
    )
