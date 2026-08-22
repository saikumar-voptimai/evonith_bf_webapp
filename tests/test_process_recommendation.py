"""Layer 2: control settings for a blend Layer 1 has already chosen.

The blend is an input. What this layer decides is the control settings, and it
decides them from the energy balance rather than from a fitted response surface
- because every fitted alternative tried on this plant's record failed to
generalise forward.

The most important thing these tests pin is what the layer REFUSES to do:
hot blast pressure, top pressure and steam do not appear in an energy balance,
so they are passed through untouched rather than given fabricated values.
"""

from __future__ import annotations

import pytest

from utils.bmo.process_recommendation import (
    OPTIMISABLE,
    PASS_THROUGH,
    ControlSettings,
    raft_from_controls,
    recommend_controls,
)
from utils.energy_balance import EnergyBalanceInputs

PRICES = {"coke": 28.0, "nut_coke": 24.0, "pci": 18.0}


def _blend() -> EnergyBalanceInputs:
    """The reference day's blend, standing in for a Layer 1 result."""

    return EnergyBalanceInputs(
        hot_metal_mt=2275.34, slag_mt=756.29, coke_mt=767.22, nut_coke_mt=171.48,
        pci_mt=342.63, blast_volume_nm3_per_hr=111392.03, blast_temperature_c=1200.0,
        oxygen_enrichment_pct=3.81, top_gas_co_pct=24.95, top_gas_co2_pct=18.75,
        top_gas_h2_pct=2.87, top_gas_temperature_c=137.26, hm_carbon_pct=4.28,
        hm_iron_pct=94.96, hm_silicon_pct=0.41, hm_manganese_pct=0.12,
        slag_feo_pct=0.36, flux_mt=9.5, ore_mt=1400.0, sinter_mt=2100.0,
        pellet_mt=380.0, flux_loi_pct=40.0,
        fuel_vm_pct={"coke": 0.9, "nut_coke": 1.0, "pci": 19.9},
        moisture_pct={"ore": 4.0, "pellet": 2.0, "flux": 1.0, "coke": 0.4,
                      "nut_coke": 8.5, "sinter": 0.0},
        shell_loss_gj_per_hr=28.29,
    )


def _current() -> ControlSettings:
    return ControlSettings(
        blast_temperature_c=1200.0, oxygen_enrichment_pct=3.81,
        blast_volume_nm3_per_hr=111392.0, pci_kg_per_thm=150.6,
        hot_blast_pressure_bar=3.6, top_pressure_bar=1.75, steam_kg_per_hr=0.0,
    )


def _recommend(**kwargs):
    return recommend_controls(
        blend_inputs=_blend(), current=_current(), prices_rs_per_kg=PRICES, **kwargs
    )


# --- what the layer refuses to do ---------------------------------------------


def test_pressures_and_steam_are_passed_through_not_optimised():
    """They do not appear in an energy balance.

    Hot blast pressure and top pressure act through permeability and gas
    utilisation. Producing a number for them here would be fabricating a
    recommendation the physics cannot support.
    """

    r = _recommend()

    for control in PASS_THROUGH:
        assert getattr(r.settings, control) == getattr(r.current, control)
    assert set(r.diagnostics["optimised_controls"]) <= set(OPTIMISABLE) | {
        "pci_kg_per_thm"
    }
    assert "energy balance" in r.diagnostics["pass_through_note"]


def test_pci_is_held_unless_explicitly_released():
    """The operator's 'at times PCI'."""

    held = _recommend()
    assert held.settings.pci_kg_per_thm == pytest.approx(
        held.current.pci_kg_per_thm
    )
    assert "pci_kg_per_thm" not in held.diagnostics["optimised_controls"]

    released = _recommend(optimise_pci=True)
    assert "pci_kg_per_thm" in released.diagnostics["optimised_controls"]


# --- the recommendation is actionable ------------------------------------------


def test_recommendation_respects_the_move_limits():
    """An instruction the operator cannot act on next shift is not useful."""

    r = _recommend()
    deltas = r.deltas()

    assert abs(deltas["blast_temperature_c"]) <= 30.0 + 1e-6
    assert abs(deltas["oxygen_enrichment_pct"]) <= 0.5 + 1e-6
    assert abs(deltas["blast_volume_nm3_per_hr"]) <= 4000.0 + 1e-6


def test_recommendation_does_not_increase_fuel_cost():
    r = _recommend()

    assert r.fuel_cost_rs_per_thm <= r.current_fuel_cost_rs_per_thm + 1e-6
    assert r.fuel_cost_saving_rs_per_thm >= 0.0


def test_coke_saving_matches_the_derived_coefficients():
    """Cross-check against Phase 2, so the two layers cannot drift apart.

    Blast temp at -9.93 kg/100 C and O2 at -0.53 kg/% should account for the
    whole coke movement, since blast volume is near neutral.
    """

    r = _recommend()
    d = r.deltas()
    predicted = (
        d["blast_temperature_c"] / 100.0 * -9.93
        + d["oxygen_enrichment_pct"] * -0.53
    )
    actual = r.coke_rate_kg_per_thm - r.current_coke_rate_kg_per_thm

    assert actual == pytest.approx(predicted, abs=0.5)


# --- honesty about limits -------------------------------------------------------


def test_leaving_the_observed_envelope_raises_a_warning():
    """The furnace has only demonstrated 1,138.9-1,229.3 C."""

    r = _recommend()

    if r.settings.blast_temperature_c > 1229.3:
        assert any("extrapolation" in w for w in r.warnings)


def test_raft_is_advisory_and_never_blocks():
    """Forward R2 is 0.11 with a seasonal bias up to 46 C, so it cannot be a
    hard constraint - it would block valid settings in some months and pass
    invalid ones in others."""

    r = _recommend()

    assert r.raft_c is not None
    assert r.diagnostics["raft_is_advisory"] is True
    # A RAFT outside the band produces a warning, not a refusal.
    assert r.settings is not None


def test_raft_correlation_uses_the_calibrated_coefficients():
    """Intercept 1555.4 and 0.810 per C match the textbook 1559 and 0.839."""

    hot = raft_from_controls(
        ControlSettings(1250.0, 4.0, 110000.0, 150.0)
    )
    cool = raft_from_controls(
        ControlSettings(1150.0, 4.0, 110000.0, 150.0)
    )

    assert (hot - cool) / 100.0 == pytest.approx(0.810, rel=0.02)
    # More PCI cools the raceway; more oxygen heats it.
    assert raft_from_controls(ControlSettings(1200.0, 4.0, 110000.0, 200.0)) < \
        raft_from_controls(ControlSettings(1200.0, 4.0, 110000.0, 150.0))
    assert raft_from_controls(ControlSettings(1200.0, 5.0, 110000.0, 150.0)) > \
        raft_from_controls(ControlSettings(1200.0, 4.0, 110000.0, 150.0))


def test_current_settings_are_reported_for_comparison():
    """The operator needs to see what changed, not just the destination."""

    r = _recommend()

    assert r.current_coke_rate_kg_per_thm > 0.0
    assert r.current_fuel_cost_rs_per_thm > 0.0
    assert set(r.deltas()) == set(r.settings.as_dict())


# --- the Layer 1 -> Layer 2 seam ------------------------------------------------


def _fake_blend():
    from types import SimpleNamespace

    return SimpleNamespace(
        slag_mt=756.29,
        quantities_mt={"sinter_a": 2100.0, "clo_a": 1400.0, "pellet_a": 380.0},
        diagnostics={
            "fuel_rate_estimate": {
                "coke_rate_kg_thm": 337.2,
                "nut_coke_rate_kg_thm": 75.4,
                "pci_rate_kg_thm": 150.6,
            }
        },
    )


def _fake_ores():
    from utils.bmo.types import OreChemistry, OreInput

    def make(ore_id, name, moisture):
        return OreInput(
            ore_id=ore_id, display_name=name, stock_mt=9e4, price_rs_per_mt=7000.0,
            min_share_pct=0.0, max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=60.0, moisture_pct=moisture),
            metadata={"material_key": ore_id},
        )

    return [
        make("sinter_a", "SINTER (SP-02)", 2.0),
        make("clo_a", "GEOMIN CLO", 5.0),
        make("pellet_a", "LLOYDS PELLET", 2.0),
    ]


def _snapshot():
    return {
        "hot_blast_vol_nm3h": 111392.0, "hot_blast_temp": 1200.0,
        "oxygen_enrichment_pct": 3.81, "co_pct": 24.95, "co2_pct": 18.75,
        "h2_pct": 2.87, "top_temp_avg": 137.26, "hot_blast_press": 3.6,
        "top_press_avg": 1.75, "steam_injection": 0.0,
    }


def test_bridge_classifies_burden_into_the_balance_categories():
    """Sinter, pellet and ore are separate terms in the moisture calculation."""

    from utils.bmo.process_recommendation import blend_to_energy_inputs

    ei = blend_to_energy_inputs(
        _fake_blend(), hot_metal_mt=2275.34, ores=_fake_ores(),
        fuel_rates_kg_per_thm=_fake_blend().diagnostics["fuel_rate_estimate"],
        hm_chemistry={"carbon_pct": 4.28, "iron_pct": 94.96},
        process_snapshot=_snapshot(), flux_mt=9.5,
    )

    assert ei.sinter_mt == pytest.approx(2100.0)
    assert ei.ore_mt == pytest.approx(1400.0)
    assert ei.pellet_mt == pytest.approx(380.0)
    # Moisture is tonnage-weighted per category, not a blanket average.
    assert ei.moisture_pct["ore"] == pytest.approx(5.0)
    assert ei.moisture_pct["sinter"] == pytest.approx(2.0)


def test_bridge_converts_fuel_rates_back_to_daily_tonnes():
    from utils.bmo.process_recommendation import blend_to_energy_inputs

    hm = 2275.34
    ei = blend_to_energy_inputs(
        _fake_blend(), hot_metal_mt=hm, ores=_fake_ores(),
        fuel_rates_kg_per_thm=_fake_blend().diagnostics["fuel_rate_estimate"],
        hm_chemistry={"carbon_pct": 4.28, "iron_pct": 94.96},
        process_snapshot=_snapshot(),
    )

    assert ei.coke_mt == pytest.approx(337.2 * hm / 1000.0)
    assert ei.pci_mt == pytest.approx(150.6 * hm / 1000.0)
    # And the slag comes straight from Layer 1, unmodified.
    assert ei.slag_mt == pytest.approx(756.29)


def test_a_category_with_no_material_contributes_no_moisture():
    """Otherwise an absent category would inject a spurious average."""

    from utils.bmo.process_recommendation import blend_to_energy_inputs

    blend = _fake_blend()
    blend.quantities_mt = {"clo_a": 1400.0}

    ei = blend_to_energy_inputs(
        blend, hot_metal_mt=2275.34, ores=_fake_ores(),
        fuel_rates_kg_per_thm=blend.diagnostics["fuel_rate_estimate"],
        hm_chemistry={"carbon_pct": 4.28, "iron_pct": 94.96},
        process_snapshot=_snapshot(),
    )

    assert ei.sinter_mt == 0.0
    assert ei.moisture_pct["sinter"] == 0.0
    assert ei.moisture_pct["ore"] == pytest.approx(5.0)


def test_full_layer1_to_layer2_handoff():
    """A Layer 1 blend goes in, control settings come out."""

    from utils.bmo.process_recommendation import (
        ControlSettings, blend_to_energy_inputs, recommend_controls,
    )

    blend = _fake_blend()
    ei = blend_to_energy_inputs(
        blend, hot_metal_mt=2275.34, ores=_fake_ores(),
        fuel_rates_kg_per_thm=blend.diagnostics["fuel_rate_estimate"],
        hm_chemistry={"carbon_pct": 4.28, "iron_pct": 94.96, "silicon_pct": 0.41},
        process_snapshot=_snapshot(), flux_mt=9.5, shell_loss_gj_per_hr=28.29,
    )
    r = recommend_controls(
        blend_inputs=ei,
        current=ControlSettings(1200.0, 3.81, 111392.0, 150.6, 3.6, 1.75, 0.0),
        prices_rs_per_kg=PRICES,
    )

    assert r.coke_rate_kg_per_thm < r.current_coke_rate_kg_per_thm
    assert r.fuel_cost_saving_rs_per_thm > 0.0
    assert r.settings.top_pressure_bar == pytest.approx(1.75)
