"""Golden regression for ``BF-2 Burden (2).xlsx`` uppercase ``BURDEN`` sheet."""

from __future__ import annotations

from dataclasses import replace

import pytest

from utils.bmo.calculations import (
    compute_charging_requirements,
    evaluate_blend,
    scale_ore_quantities_to_hot_metal,
)
from utils.bmo.slag_balance import FE_FROM_FE2O3_FACTOR, fuel_quantities_mt
from utils.bmo.types import (
    DustInput,
    FluxInput,
    FuelAshInput,
    OreChemistry,
    OreInput,
    SlagBalanceSettings,
    oxide_pct_from_basis,
)

EXCEL_ACTUAL_PI_MT = 14.456321890636


def _oxide(value: float, basis: str, element: str) -> float:
    return oxide_pct_from_basis(value, basis, element=element)


def _golden_case():
    # Positive wet-material rows D22:D40. Chemistry columns P/S in BURDEN are
    # elemental Mn/Ti, so ingestion converts them once into canonical MnO/TiO2.
    ore_rows = [
        (
            "sinter_02",
            17.05547,
            0.0,
            55.0,
            5.4,
            2.13,
            10.8,
            2.06,
            0.21,
            0.05,
            0.01,
            0.15,
            0.073,
        ),
        (
            "lloyds_washed",
            0.9778,
            5.6,
            63.46,
            2.67,
            2.91,
            0.045,
            0.02,
            0.017,
            0.051,
            0.05,
            0.156,
            0.053,
        ),
        (
            "lloyds_pellet",
            2.525,
            1.1,
            63.86,
            4.45,
            2.18,
            0.97,
            0.04,
            0.02,
            0.052,
            0.05,
            0.132,
            0.045,
        ),
        (
            "giomine",
            4.696,
            4.3,
            50.84,
            9.62,
            8.46,
            0.06,
            0.06,
            0.12,
            0.105,
            0.01,
            0.834,
            0.053,
        ),
    ]
    ores: list[OreInput] = []
    quantities: dict[str, float] = {}
    for (
        ore_id,
        wet_mt,
        moisture,
        fe,
        sio2,
        al2o3,
        cao,
        mgo,
        mn,
        p,
        s,
        ti,
        alkali,
    ) in ore_rows:
        ores.append(
            OreInput(
                ore_id=ore_id,
                display_name=ore_id,
                stock_mt=100.0,
                price_rs_per_mt=0.0,
                min_share_pct=0.0,
                max_share_pct=100.0,
                chemistry=OreChemistry(
                    fe_t_pct=fe,
                    moisture_pct=moisture,
                    sio2_pct=sio2,
                    al2o3_pct=al2o3,
                    cao_pct=cao,
                    mgo_pct=mgo,
                    mno_pct=_oxide(mn, "mn", "mn"),
                    p_pct=p,
                    s_pct=s,
                    tio2_pct=_oxide(ti, "ti", "ti"),
                    k2o_pct=alkali,
                ),
            )
        )
        quantities[ore_id] = wet_mt

    fluxes = [
        FluxInput(
            flux_id="dolomite",
            display_name="Dolomite",
            wet_qty_mt=0.1,
            moisture_pct=0.3,
            sio2_pct=1.31,
            al2o3_pct=0.12,
            cao_pct=30.25,
            mgo_pct=22.85,
            fe2o3_pct=0.088 / FE_FROM_FE2O3_FACTOR,
            tio2_pct=_oxide(0.02, "ti", "ti"),
            k2o_pct=0.3,
            p_pct=0.002,
            s_pct=0.05,
        )
    ]

    # Wet kg/charge D60/D58/M9 converted to wet kg/THM using actual PI V69.
    fuel_rows = [
        (
            "coke",
            4.520,
            0.0,
            12.176,
            57.69,
            26.38,
            2.7,
            1.05,
            6.49,
            0.02,
            1.59,
            2.0778580814717476,
            0.72,
            0.032,
        ),
        (
            "nut_coke",
            1.105,
            8.5,
            12.55,
            56.85,
            26.81,
            2.67,
            1.03,
            7.1,
            0.02,
            1.598,
            2.0,
            0.78,
            0.032,
        ),
        (
            "pci",
            2.6021379403144347,
            2.05,
            8.63,
            50.0,
            28.27,
            5.19,
            1.93,
            5.81,
            0.05,
            1.031,
            2.1900347624565466,
            0.38,
            0.021,
        ),
    ]
    fuels = [
        FuelAshInput(
            fuel_id=fuel_id,
            display_name=fuel_id,
            rate_kg_per_thm=wet_mt * 1000.0 / EXCEL_ACTUAL_PI_MT,
            rate_basis="wet",
            moisture_pct=moisture,
            ash_pct=ash,
            sio2_pct=sio2,
            al2o3_pct=al2o3,
            cao_pct=cao,
            mgo_pct=mgo,
            fe2o3_pct=fe2o3,
            mno_pct=_oxide(mn, "mn", "mn"),
            tio2_pct=_oxide(ti, "ti", "ti"),
            alkali_pct=alkali,
            s_pct=s,
            p_pct=p,
            chemistry_source="BF-2 Burden (2).xlsx/BURDEN",
        )
        for (
            fuel_id,
            wet_mt,
            moisture,
            ash,
            sio2,
            al2o3,
            cao,
            mgo,
            fe2o3,
            mn,
            ti,
            alkali,
            s,
            p,
        ) in fuel_rows
    ]

    dust = [
        DustInput(
            dust_id="bf_gas_dust",
            display_name="BF Gas Dust",
            rate_basis="kg_per_charge",
            quantity_kg_per_charge=530.0,
            source="BF-2 Burden (2).xlsx/BURDEN",
            sio2_pct=4.5,
            al2o3_pct=4.86,
            cao_pct=5.63,
            mgo_pct=3.57,
            fe_pct=33.5,
            p_pct=0.054,
            s_pct=0.1,
        )
    ]
    settings = SlagBalanceSettings(
        enabled=True,
        carbon_pct=4.2,
        silicon_pct=0.6,
        sulphur_pct=0.03,
        other_pct=0.0,
        pi_loss_pct=0.2,
        fe_to_pig_iron_fraction=0.999,
        mn_recovery_pct=60.0,
        sulphur_gas_loss_pct=10.0,
        alkali_to_slag_fraction=0.8,
        si_to_sio2_factor=2.14,
        fe_to_feo_factor=72.0 / 56.0,
        mn_to_mno_factor=1.291,
        slag_correction_factor=1.0,
    )
    return ores, quantities, fuels, fluxes, dust, settings


def test_uppercase_burden_sheet_golden_balance() -> None:
    ores, quantities, fuels, fluxes, dust, settings = _golden_case()
    blend = evaluate_blend(
        ores=ores,
        quantities_mt=quantities,
        feo_in_slag_pct=0.4,
        fuel_ash_inputs=fuels,
        flux_inputs=fluxes,
        dust_inputs=dust,
        slag_balance_settings=settings,
        hot_metal_target_mt=EXCEL_ACTUAL_PI_MT,
        charge_mass_mt=26.45927,
    )
    full = blend.diagnostics["full_slag_balance"]
    components = full["slag_components_mt"]
    ib4 = (components["cao"] + components["mgo"]) / (
        components["sio2"] + components["al2o3"]
    )
    charging = compute_charging_requirements(blend, charge_mass_mt=26.45927)

    assert full["theoretical_pig_iron_mt"] == pytest.approx(14.485292475587, abs=1e-4)
    assert full["actual_pig_iron_mt"] == pytest.approx(EXCEL_ACTUAL_PI_MT, abs=1e-4)
    assert blend.slag_mt == pytest.approx(5.202128092831, abs=1e-3)
    assert blend.slag_rate_kg_per_thm == pytest.approx(359.851429169494, abs=0.05)
    assert blend.slag_basicity == pytest.approx(1.066920427646, abs=1e-5)
    assert blend.slag_ib4 == pytest.approx(0.804664838186, abs=1e-5)
    assert ib4 == pytest.approx(0.804664838186, abs=1e-5)
    assert charging["chemical_hot_metal_per_charge_mt"] == pytest.approx(
        EXCEL_ACTUAL_PI_MT, abs=1e-4
    )
    assert blend.diagnostics["dust_usage"][0]["kg_per_charge"] == pytest.approx(530.0)
    assert blend.diagnostics["dust_usage"][0]["wet_qty_mt"] == pytest.approx(0.530)


def test_all_source_fe_closure_scales_excel_mix_to_target_hm() -> None:
    ores, per_charge, fuels, fluxes, dust, settings = _golden_case()
    charge_scale = 2350.0 / EXCEL_ACTUAL_PI_MT
    daily_fluxes = [
        replace(flux, wet_qty_mt=flux.wet_qty_mt * charge_scale) for flux in fluxes
    ]
    # This is the former ore-Fe-only answer reported by BMO for the source mix.
    reference_total_mt = 4052.42
    per_charge_total = sum(per_charge.values())
    reference = {
        ore_id: qty / per_charge_total * reference_total_mt
        for ore_id, qty in per_charge.items()
    }

    scaled = scale_ore_quantities_to_hot_metal(
        ores=ores,
        reference_quantities_mt=reference,
        target_hot_metal_mt=2350.0,
        fuel_ash_inputs=fuels,
        flux_inputs=daily_fluxes,
        dust_inputs=dust,
        slag_balance_settings=settings,
        charge_mass_mt=26.45927,
    )
    blend = evaluate_blend(
        ores=ores,
        quantities_mt=scaled,
        feo_in_slag_pct=0.4,
        fuel_ash_inputs=fuels,
        flux_inputs=daily_fluxes,
        dust_inputs=dust,
        slag_balance_settings=settings,
        hot_metal_target_mt=2350.0,
        charge_mass_mt=26.45927,
    )

    assert sum(scaled.values()) == pytest.approx(4105.30, abs=0.02)
    assert blend.diagnostics["full_slag_balance"][
        "actual_pig_iron_mt"
    ] == pytest.approx(2350.0, abs=1e-6)
    assert blend.diagnostics["dust_usage"][0]["wet_qty_mt"] == pytest.approx(
        86.156, abs=0.01
    )


def test_fuel_wet_and_dry_rate_bases_apply_moisture_once() -> None:
    wet = FuelAshInput(
        fuel_id="nut_coke",
        display_name="Nut Coke",
        rate_kg_per_thm=1105.0,
        rate_basis="wet",
        moisture_pct=8.5,
    )
    dry = replace(wet, rate_kg_per_thm=1011.075, rate_basis="dry")

    assert fuel_quantities_mt(wet, 1.0) == pytest.approx((1.105, 1.011075))
    assert fuel_quantities_mt(dry, 1.0) == pytest.approx((1.105, 1.011075))
