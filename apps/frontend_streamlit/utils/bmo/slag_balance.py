"""Full BF slag-balance calculations for BMO.

This module implements the plant-style component balance used to calculate
slag from ores, fluxes, fuel ash, dust deduction, pig-iron chemistry, recovery
assumptions, and final slag component conversion.
"""

from __future__ import annotations

from collections.abc import Mapping

from furnace_data.bmo.utils.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreInput,
    SlagBalanceResult,
    SlagBalanceSettings,
)

COMPONENT_KEYS = (
    "sio2",
    "al2o3",
    "cao",
    "mgo",
    "mn",
    "p",
    "s",
    "ti",
    "zn",
    "alkali",
    "fe",
    "caf2",
)

MN_FROM_MNO_FACTOR = 54.938 / 70.937
TI_FROM_TIO2_FACTOR = 47.867 / 79.866
FE_FROM_FE2O3_FACTOR = 111.69 / 159.69


def _safe_float(value: float | int | str | None) -> float:
    """
    Convert a possibly blank numeric value into a safe float.

    Slag-balance inputs can come from YAML defaults, Streamlit tables, or future
    online sources. This helper keeps invalid values from breaking the balance
    and lets downstream clamping decide whether the value should be negative.

    Args:
         - value: float | int | str | None - Raw numeric value.

    Returns:
         - return float - Parsed numeric value, or zero when invalid.
    """

    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_pct(value: float | int | str | None) -> float:
    """
    Normalize a percentage into the allowed zero-to-hundred range.

    Component percentages are expected on dry basis. Clamping protects the
    workbook-style calculations from impossible values while preserving normal
    plant chemistry percentages.

    Args:
         - value: float | int | str | None - Raw percentage value.

    Returns:
         - return float - Percentage clamped between 0 and 100.
    """

    # Upper bound clamps to 99.0 so a moisture entry of 100 (typo or stale
    # input) cannot collapse dry weight to exactly zero in _dry_weight_mt
    # below. 99.0 still yields a vanishingly small dry weight so a real
    # all-water material is still surfaced as ~0 Fe / ~0 slag.
    return min(max(_safe_float(value), 0.0), 99.0)


def _dry_weight_mt(wet_weight_mt: float, moisture_pct: float) -> float:
    """
    Convert wet material quantity to dry material quantity.

    This helper follows the plant formula
    ``DryWeight = WetWeight * (100 - Moisture%) / 100`` for all ores, fluxes,
    fuel, and dust rows handled by the full slag balance.

    Args:
         - wet_weight_mt: float - Wet material quantity in MT.
         - moisture_pct: float - Moisture percentage in wet material.

    Returns:
         - return float - Dry material quantity in MT.
    """

    wet_weight = max(0.0, _safe_float(wet_weight_mt))
    dry_fraction = (100.0 - _safe_pct(moisture_pct)) / 100.0
    return wet_weight * dry_fraction


def _component_mass_mt(dry_weight_mt: float, component_pct: float) -> float:
    """
    Calculate component mass from dry weight and component percentage.

    This is the common component formula used by the balance:
    ``Component MT = DryWeight * Component% / 100``.

    Args:
         - dry_weight_mt: float - Dry material quantity in MT.
         - component_pct: float - Component percentage on dry basis.

    Returns:
         - return float - Component mass in MT.
    """

    return max(0.0, _safe_float(dry_weight_mt)) * (_safe_pct(component_pct) / 100.0)


def _empty_components() -> dict[str, float]:
    """
    Create a zeroed component map for slag-balance accumulation.

    Keeping a stable component key set makes diagnostics and tests predictable
    even when some inputs do not provide every chemistry field.

    Args:
         - None

    Returns:
         - return dict[str, float] - Zeroed component mass map.
    """

    return {key: 0.0 for key in COMPONENT_KEYS}


def _add_components(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    """
    Add two component maps using the shared component key set.

    Args:
         - left: Mapping[str, float] - First component map.
         - right: Mapping[str, float] - Second component map.

    Returns:
         - return dict[str, float] - Summed component map.
    """

    return {
        key: _safe_float(left.get(key, 0.0)) + _safe_float(right.get(key, 0.0))
        for key in COMPONENT_KEYS
    }


def _subtract_components(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    """
    Subtract one component map from another and clamp at zero.

    Dust deduction should not create negative net component masses. Clamping
    keeps the final balance physically safe if a user enters an excessive dust
    loss for one component.

    Args:
         - left: Mapping[str, float] - Component map before deduction.
         - right: Mapping[str, float] - Component map to deduct.

    Returns:
         - return dict[str, float] - Non-negative net component map.
    """

    return {
        key: max(
            0.0,
            _safe_float(left.get(key, 0.0)) - _safe_float(right.get(key, 0.0)),
        )
        for key in COMPONENT_KEYS
    }


def build_ore_component_totals(
    ores: list[OreInput], quantities_mt: Mapping[str, float]
) -> dict[str, float]:
    """
    Calculate total component masses contributed by iron-bearing ores.

    Each ore wet quantity is converted to dry quantity before Fe, oxide,
    phosphorus, sulphur, zinc, titanium, manganese, and alkali components are
    accumulated. Existing MnO and TiO2 ore fields are converted to Mn and Ti
    element masses for the full balance.

    Args:
         - ores: list[OreInput] - Ores included in the evaluated blend.
         - quantities_mt: Mapping[str, float] - Wet ore quantities keyed by ore id.

    Returns:
         - return dict[str, float] - Ore component totals in MT.
    """

    totals = _empty_components()
    for ore in ores:
        dry_weight = _dry_weight_mt(
            quantities_mt.get(ore.ore_id, 0.0), ore.chemistry.moisture_pct
        )
        totals["fe"] += _component_mass_mt(dry_weight, ore.chemistry.fe_t_pct)
        totals["sio2"] += _component_mass_mt(dry_weight, ore.chemistry.sio2_pct)
        totals["al2o3"] += _component_mass_mt(dry_weight, ore.chemistry.al2o3_pct)
        totals["cao"] += _component_mass_mt(dry_weight, ore.chemistry.cao_pct)
        totals["mgo"] += _component_mass_mt(dry_weight, ore.chemistry.mgo_pct)
        totals["mn"] += (
            _component_mass_mt(dry_weight, ore.chemistry.mno_pct) * MN_FROM_MNO_FACTOR
        )
        totals["p"] += _component_mass_mt(dry_weight, ore.chemistry.p_pct)
        totals["s"] += _component_mass_mt(
            dry_weight, getattr(ore.chemistry, "s_pct", 0.0)
        )
        totals["ti"] += (
            _component_mass_mt(dry_weight, ore.chemistry.tio2_pct) * TI_FROM_TIO2_FACTOR
        )
        totals["zn"] += _component_mass_mt(
            dry_weight, getattr(ore.chemistry, "zn_pct", 0.0)
        )
        totals["alkali"] += _component_mass_mt(
            dry_weight,
            _safe_pct(ore.chemistry.na2o_pct) + _safe_pct(ore.chemistry.k2o_pct),
        )
    return totals


def build_flux_component_totals(
    flux_inputs: list[FluxInput] | None,
) -> dict[str, float]:
    """
    Calculate total component masses contributed by fixed fluxes.

    Flux wet quantities are converted to dry quantity before oxide and minor
    component percentages are applied. Fe2O3 is converted to Fe element mass
    because the full balance tracks Fe before converting remaining Fe to FeO.

    Args:
         - flux_inputs: list[FluxInput] | None - Fixed flux rows from UI/config.

    Returns:
         - return dict[str, float] - Flux component totals in MT.
    """

    totals = _empty_components()
    for flux in flux_inputs or []:
        if not flux.enabled:
            continue
        dry_weight = _dry_weight_mt(flux.wet_qty_mt, flux.moisture_pct)
        totals["sio2"] += _component_mass_mt(dry_weight, flux.sio2_pct)
        totals["al2o3"] += _component_mass_mt(dry_weight, flux.al2o3_pct)
        totals["cao"] += _component_mass_mt(dry_weight, flux.cao_pct)
        totals["mgo"] += _component_mass_mt(dry_weight, flux.mgo_pct)
        totals["fe"] += (
            _component_mass_mt(dry_weight, flux.fe2o3_pct) * FE_FROM_FE2O3_FACTOR
        )
        totals["mn"] += (
            _component_mass_mt(dry_weight, flux.mno_pct) * MN_FROM_MNO_FACTOR
        )
        totals["p"] += _component_mass_mt(dry_weight, flux.p_pct)
        totals["s"] += _component_mass_mt(dry_weight, flux.s_pct)
        totals["ti"] += (
            _component_mass_mt(dry_weight, flux.tio2_pct) * TI_FROM_TIO2_FACTOR
        )
        totals["zn"] += _component_mass_mt(dry_weight, flux.zn_pct)
        totals["alkali"] += _component_mass_mt(
            dry_weight, _safe_pct(flux.na2o_pct) + _safe_pct(flux.k2o_pct)
        )
        totals["caf2"] += _component_mass_mt(dry_weight, flux.caf2_pct)
    return totals


def build_fuel_ash_component_totals(
    fuel_ash_inputs: list[FuelAshInput] | None, hot_metal_mt: float
) -> dict[str, float]:
    """
    Calculate total component masses contributed by fuel ash.

    Fuel rates are converted from kg/THM into wet MT, moisture is removed, ash
    percentage is applied, and ash oxide chemistry is added to the BF component
    balance. Sulphur and phosphorus are treated on dry fuel basis because their
    configured records are not ash-oxide percentages.

    Args:
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash rows from UI/config.
         - hot_metal_mt: float - Hot-metal denominator used for kg/THM fuel rates.

    Returns:
         - return dict[str, float] - Fuel ash component totals in MT.
    """

    totals = _empty_components()
    hot_metal = max(0.0, _safe_float(hot_metal_mt))
    for fuel in fuel_ash_inputs or []:
        if not fuel.enabled:
            continue
        wet_fuel_mt = max(0.0, _safe_float(fuel.rate_kg_per_thm)) * hot_metal / 1000.0
        dry_fuel_mt = _dry_weight_mt(wet_fuel_mt, fuel.moisture_pct)
        ash_mt = dry_fuel_mt * (_safe_pct(fuel.ash_pct) / 100.0)
        totals["sio2"] += _component_mass_mt(ash_mt, fuel.sio2_pct)
        totals["al2o3"] += _component_mass_mt(ash_mt, fuel.al2o3_pct)
        totals["cao"] += _component_mass_mt(ash_mt, fuel.cao_pct)
        totals["mgo"] += _component_mass_mt(ash_mt, fuel.mgo_pct)
        totals["fe"] += (
            _component_mass_mt(ash_mt, fuel.fe2o3_pct) * FE_FROM_FE2O3_FACTOR
        )
        totals["ti"] += _component_mass_mt(ash_mt, fuel.tio2_pct) * TI_FROM_TIO2_FACTOR
        totals["alkali"] += _component_mass_mt(
            ash_mt, _safe_pct(fuel.na2o_pct) + _safe_pct(fuel.k2o_pct)
        )
        totals["s"] += _component_mass_mt(dry_fuel_mt, fuel.s_pct)
        totals["p"] += _component_mass_mt(dry_fuel_mt, fuel.p_pct)
    return totals


def build_dust_component_totals(
    dust_inputs: list[DustInput] | None,
) -> dict[str, float]:
    """
    Calculate component masses deducted as BF gas dust loss.

    Dust rows are converted from wet to dry quantity before chemistry
    percentages are applied. The returned totals are subtracted from total BF
    input components in the full slag balance.

    Args:
         - dust_inputs: list[DustInput] | None - Dust rows from UI/config.

    Returns:
         - return dict[str, float] - Dust component totals in MT.
    """

    totals = _empty_components()
    for dust in dust_inputs or []:
        if not dust.enabled:
            continue
        dry_weight = _dry_weight_mt(dust.wet_qty_mt, dust.moisture_pct)
        totals["sio2"] += _component_mass_mt(dry_weight, dust.sio2_pct)
        totals["al2o3"] += _component_mass_mt(dry_weight, dust.al2o3_pct)
        totals["cao"] += _component_mass_mt(dry_weight, dust.cao_pct)
        totals["mgo"] += _component_mass_mt(dry_weight, dust.mgo_pct)
        totals["fe"] += _component_mass_mt(dry_weight, dust.fe_pct)
        totals["mn"] += _component_mass_mt(dry_weight, dust.mn_pct)
        totals["p"] += _component_mass_mt(dry_weight, dust.p_pct)
        totals["s"] += _component_mass_mt(dry_weight, dust.s_pct)
        totals["ti"] += _component_mass_mt(dry_weight, dust.ti_pct)
        totals["zn"] += _component_mass_mt(dry_weight, dust.zn_pct)
        totals["alkali"] += _component_mass_mt(
            dry_weight, _safe_pct(dust.na2o_pct) + _safe_pct(dust.k2o_pct)
        )
        totals["caf2"] += _component_mass_mt(dry_weight, dust.caf2_pct)
    return totals


def _attribute_per_source_slag(
    *,
    ore_components: Mapping[str, float],
    flux_components: Mapping[str, float],
    fuel_ash_components: Mapping[str, float],
    total_into_bf: Mapping[str, float],
    net_into_bf: Mapping[str, float],
    raw_slag_components: Mapping[str, float],
) -> dict[str, float]:
    """
    Pro-rata attribute final slag MT to ore, flux, and fuel-ash sources.

    Each input component (sio2, al2o3, cao, mgo, alkali, caf2, s) has a
    retention ratio = raw final slag mass / net input mass after dust deduction.
    Fe and Mn change form on the way out: their retention converts element mass
    to FeO/MnO mass via the residual fraction and stoichiometric factor.
    Ti/P/Zn report to pig iron and have zero slag retention. Dust is applied
    pro-rata to each source by the (net/total) ratio per component so the per-
    source contributions sum to the pre-correction total slag.

    Args:
         - ore_components: Mapping[str, float] - Ore-side component totals in MT.
         - flux_components: Mapping[str, float] - Flux-side component totals in MT.
         - fuel_ash_components: Mapping[str, float] - Fuel-ash-side component totals in MT.
         - total_into_bf: Mapping[str, float] - Ore + flux + fuel ash totals before dust.
         - net_into_bf: Mapping[str, float] - Totals after dust deduction.
         - raw_slag_components: Mapping[str, float] - Final slag MT per component pre-correction.

    Returns:
         - return dict[str, float] - {ore, flux, fuel_ash} → raw slag MT contributions.
    """

    retention: dict[str, float] = {}
    direct_keys = ("sio2", "al2o3", "cao", "mgo", "alkali", "caf2", "s")
    for key in direct_keys:
        net_val = _safe_float(net_into_bf.get(key, 0.0))
        slag_val = _safe_float(raw_slag_components.get(key, 0.0))
        retention[key] = (slag_val / net_val) if net_val > 0 else 0.0

    net_fe = _safe_float(net_into_bf.get("fe", 0.0))
    feo_slag = _safe_float(raw_slag_components.get("feo", 0.0))
    retention["fe"] = (feo_slag / net_fe) if net_fe > 0 else 0.0

    net_mn = _safe_float(net_into_bf.get("mn", 0.0))
    mno_slag = _safe_float(raw_slag_components.get("mno", 0.0))
    retention["mn"] = (mno_slag / net_mn) if net_mn > 0 else 0.0

    def _source_total(source_components: Mapping[str, float]) -> float:
        total = 0.0
        for component, mass in source_components.items():
            gross = _safe_float(total_into_bf.get(component, 0.0))
            net_share = (
                _safe_float(net_into_bf.get(component, 0.0)) / gross if gross > 0 else 1.0
            )
            total += _safe_float(mass) * net_share * retention.get(component, 0.0)
        return total

    return {
        "ore": _source_total(ore_components),
        "flux": _source_total(flux_components),
        "fuel_ash": _source_total(fuel_ash_components),
    }


def calculate_full_slag_balance(
    *,
    ores: list[OreInput],
    quantities_mt: Mapping[str, float],
    hot_metal_mt: float,
    settings: SlagBalanceSettings,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
) -> SlagBalanceResult:
    """
    Calculate full BF slag balance from ore, flux, fuel ash, and dust inputs.

    This function implements the workbook-style sequence: dry weights,
    component masses, ore totals, flux totals, fuel ash totals, total BF input,
    dust deduction, pig iron calculation, SiO2 consumption, FeO conversion, MnO
    conversion, sulphur split, alkali-to-slag split, final slag summation, and
    plant correction factor application.

    Args:
         - ores: list[OreInput] - Ores included in the evaluated blend.
         - quantities_mt: Mapping[str, float] - Wet ore quantities keyed by ore id.
         - hot_metal_mt: float - Hot-metal denominator for fuel rates and slag rate.
         - settings: SlagBalanceSettings - Recovery and pig iron assumptions.
         - fuel_ash_inputs: list[FuelAshInput] | None - Fuel ash rows from UI/config.
         - flux_inputs: list[FluxInput] | None - Fixed flux rows from UI/config.
         - dust_inputs: list[DustInput] | None - Dust rows to deduct from BF input.

    Returns:
         - return SlagBalanceResult - Full slag balance result and diagnostics.
    """

    ore_components = build_ore_component_totals(ores, quantities_mt)
    flux_components = build_flux_component_totals(flux_inputs)
    fuel_ash_components = build_fuel_ash_component_totals(
        fuel_ash_inputs, hot_metal_mt=hot_metal_mt
    )
    dust_components = build_dust_component_totals(dust_inputs)

    total_into_bf = _add_components(
        _add_components(ore_components, flux_components), fuel_ash_components
    )
    net_into_bf = _subtract_components(total_into_bf, dust_components)

    pig_iron_metal_pct = max(
        1.0,
        100.0
        - _safe_pct(settings.carbon_pct)
        - _safe_pct(settings.silicon_pct)
        - _safe_pct(settings.sulphur_pct)
        - _safe_pct(settings.other_pct),
    )
    fe_to_pi_fraction = min(
        max(_safe_float(settings.fe_to_pig_iron_fraction), 0.0), 1.0
    )
    fe_pi_mt = net_into_bf["fe"] * fe_to_pi_fraction
    fe_remaining_mt = max(0.0, net_into_bf["fe"] - fe_pi_mt)

    recovery_fraction = _safe_pct(settings.mn_recovery_pct) / 100.0
    # Pass 1 estimate: use recovery factors for Mn/Ti so we have an initial PI
    # mass; pass 2 below refines Mn/Ti partitioning using actual HM percentages
    # so the slag balance reflects observed plant chemistry instead of a fixed
    # 60% recovery.
    mn_pi_initial_mt = net_into_bf["mn"] * recovery_fraction
    ti_pi_initial_mt = net_into_bf["ti"] * recovery_fraction
    metallic_mass_initial_mt = (
        fe_pi_mt
        + mn_pi_initial_mt
        + ti_pi_initial_mt
        + net_into_bf["p"]
        + net_into_bf["zn"]
    )
    theoretical_pi_initial_mt = (
        metallic_mass_initial_mt * 100.0 / pig_iron_metal_pct
    )
    pi_loss_pct = min(_safe_pct(settings.pi_loss_pct), 99.0)
    actual_pi_initial_mt = theoretical_pi_initial_mt * (
        (100.0 - pi_loss_pct) / 100.0
    )

    hm_mn_pct = _safe_pct(getattr(settings, "mn_pct", 0.0))
    hm_ti_pct = _safe_pct(getattr(settings, "ti_pct", 0.0))
    if hm_mn_pct > 0:
        mn_pi_mt = min(
            net_into_bf["mn"],
            actual_pi_initial_mt * hm_mn_pct / 100.0,
        )
    else:
        mn_pi_mt = mn_pi_initial_mt
    if hm_ti_pct > 0:
        ti_pi_mt = min(
            net_into_bf["ti"],
            actual_pi_initial_mt * hm_ti_pct / 100.0,
        )
    else:
        ti_pi_mt = ti_pi_initial_mt
    mn_remaining_mt = max(0.0, net_into_bf["mn"] - mn_pi_mt)
    ti_remaining_mt = max(0.0, net_into_bf["ti"] - ti_pi_mt)

    metallic_mass_mt = (
        fe_pi_mt + mn_pi_mt + ti_pi_mt + net_into_bf["p"] + net_into_bf["zn"]
    )
    theoretical_pi_mt = metallic_mass_mt * 100.0 / pig_iron_metal_pct
    actual_pi_mt = theoretical_pi_mt * ((100.0 - pi_loss_pct) / 100.0)

    sio2_consumed_mt = (
        actual_pi_mt
        * (_safe_pct(settings.silicon_pct) / 100.0)
        * max(0.0, _safe_float(settings.si_to_sio2_factor))
    )
    sio2_slag_mt = max(0.0, net_into_bf["sio2"] - sio2_consumed_mt)

    feo_slag_mt = fe_remaining_mt * max(0.0, _safe_float(settings.fe_to_feo_factor))

    mno_slag_mt = mn_remaining_mt * max(0.0, _safe_float(settings.mn_to_mno_factor))

    sulphur_pi_mt = actual_pi_mt * (_safe_pct(settings.sulphur_pct) / 100.0)
    sulphur_gas_mt = net_into_bf["s"] * (
        _safe_pct(settings.sulphur_gas_loss_pct) / 100.0
    )
    sulphur_slag_mt = max(0.0, net_into_bf["s"] - sulphur_pi_mt - sulphur_gas_mt)

    alkali_fraction = min(max(_safe_float(settings.alkali_to_slag_fraction), 0.0), 1.0)
    alkali_slag_mt = net_into_bf["alkali"] * alkali_fraction

    raw_slag_components = {
        "alkali": float(alkali_slag_mt),
        "sio2": float(sio2_slag_mt),
        "al2o3": float(net_into_bf["al2o3"]),
        "cao": float(net_into_bf["cao"]),
        "mgo": float(net_into_bf["mgo"]),
        "feo": float(feo_slag_mt),
        "mno": float(mno_slag_mt),
        "s": float(sulphur_slag_mt),
        "caf2": float(net_into_bf["caf2"]),
    }
    raw_total_slag_mt = float(sum(raw_slag_components.values()))
    slag_correction_factor = max(0.0, _safe_float(settings.slag_correction_factor))
    slag_components = {
        key: float(value * slag_correction_factor)
        for key, value in raw_slag_components.items()
    }
    total_slag_mt = float(raw_total_slag_mt * slag_correction_factor)

    raw_source_slag = _attribute_per_source_slag(
        ore_components=ore_components,
        flux_components=flux_components,
        fuel_ash_components=fuel_ash_components,
        total_into_bf=total_into_bf,
        net_into_bf=net_into_bf,
        raw_slag_components=raw_slag_components,
    )
    ore_slag_in_final_mt = raw_source_slag["ore"] * slag_correction_factor
    flux_slag_in_final_mt = raw_source_slag["flux"] * slag_correction_factor
    fuel_ash_slag_in_final_mt = raw_source_slag["fuel_ash"] * slag_correction_factor

    mn_to_mno_factor = max(0.0, _safe_float(settings.mn_to_mno_factor))
    ti_to_tio2_factor = 1.0 / TI_FROM_TIO2_FACTOR  # element Ti -> TiO2 mass
    hm_reduction_sio2_mt = float(sio2_consumed_mt)
    hm_reduction_mno_mt = float(mn_pi_mt * mn_to_mno_factor)
    # Only the Ti that actually went to HM is counted here. The remaining Ti
    # (not added to slag per spec) is exposed as a separate diagnostic so the
    # imbalance is transparent.
    hm_reduction_tio2_mt = float(ti_pi_mt * ti_to_tio2_factor)
    tio2_unaccounted_mt = float(ti_remaining_mt * ti_to_tio2_factor)
    hm_reduction_s_mt = float(sulphur_pi_mt + sulphur_gas_mt)
    hm_reduction_alkali_mt = float(
        max(0.0, _safe_float(net_into_bf.get("alkali", 0.0))) * (1.0 - alkali_fraction)
    )

    return SlagBalanceResult(
        total_slag_mt=total_slag_mt,
        actual_pig_iron_mt=float(actual_pi_mt),
        theoretical_pig_iron_mt=float(theoretical_pi_mt),
        ore_components_mt={key: float(value) for key, value in ore_components.items()},
        flux_components_mt={
            key: float(value) for key, value in flux_components.items()
        },
        fuel_ash_components_mt={
            key: float(value) for key, value in fuel_ash_components.items()
        },
        dust_components_mt={
            key: float(value) for key, value in dust_components.items()
        },
        total_into_bf_mt={key: float(value) for key, value in total_into_bf.items()},
        net_into_bf_mt={key: float(value) for key, value in net_into_bf.items()},
        slag_components_mt={
            key: float(value) for key, value in slag_components.items()
        },
        diagnostics={
            "pig_iron_metal_pct": float(pig_iron_metal_pct),
            "metallic_mass_mt": float(metallic_mass_mt),
            "raw_total_slag_mt": float(raw_total_slag_mt),
            "raw_slag_components_mt": {
                key: float(value) for key, value in raw_slag_components.items()
            },
            "slag_correction_factor": float(slag_correction_factor),
            "sio2_consumed_by_si_mt": float(sio2_consumed_mt),
            "fe_to_pig_iron_mt": float(fe_pi_mt),
            "fe_remaining_mt": float(fe_remaining_mt),
            "mn_to_pig_iron_mt": float(mn_pi_mt),
            "mn_remaining_mt": float(mn_remaining_mt),
            "ti_to_pig_iron_mt": float(ti_pi_mt),
            "ti_remaining_mt": float(ti_remaining_mt),
            "sulphur_to_pig_iron_mt": float(sulphur_pi_mt),
            "sulphur_to_gas_mt": float(sulphur_gas_mt),
            "alkali_to_slag_fraction": float(alkali_fraction),
            "ore_slag_in_final_mt": float(ore_slag_in_final_mt),
            "flux_slag_in_final_mt": float(flux_slag_in_final_mt),
            "fuel_ash_slag_in_final_mt": float(fuel_ash_slag_in_final_mt),
            "hm_reduction_sio2_mt": hm_reduction_sio2_mt,
            "hm_reduction_mno_mt": hm_reduction_mno_mt,
            "hm_reduction_tio2_mt": hm_reduction_tio2_mt,
            "hm_reduction_s_mt": hm_reduction_s_mt,
            "hm_reduction_alkali_mt": hm_reduction_alkali_mt,
            "tio2_unaccounted_mt": tio2_unaccounted_mt,
            "hm_mn_pct_used": float(hm_mn_pct),
            "hm_ti_pct_used": float(hm_ti_pct),
            "calculation_sequence": "dry_weight_component_balance_full_bf_slag",
        },
    )
