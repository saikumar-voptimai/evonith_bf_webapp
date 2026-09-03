"""Tests for the physics-based coke-rate correction.

The deployed fuel model is blind to the burden (doubling ore per THM moves it
0.012%) and the plant history has no recoverable slag-to-fuel signal, so this
layer supplies the entire blend sensitivity from first principles. These tests
pin the coefficients, the guardrails, and above all the non-regression
guarantee: the correction is exactly zero when the blend reproduces the
reference operating point.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from utils.bmo.coke_correction import (
    TERM_BURDEN_OXYGEN,
    TERM_FLUX_CALCINATION,
    TERM_HOT_METAL_SI,
    TERM_SLAG_HEAT,
    CokeCorrectionDrivers,
    CokeCorrectionReference,
    build_linear_coke_correction_cost_coeffs,
    build_reference,
    compute_burden_oxygen_kg_per_thm,
    compute_coke_correction,
    compute_flux_co2_kg_per_thm,
    load_coke_correction_settings,
    soft_saturate,
)
from utils.bmo.fuel_rates import (
    ASSUMED_FUEL_PRICES_RS_PER_KG,
    EstimatedFuelRates,
    with_coke_delta,
)
from utils.bmo.types import FluxInput, OreChemistry, OreInput

_ANCHOR = 300.0
_NUT = 70.0
_PCI = 170.0


def _cfg(**overrides):
    """Build a ``bmo`` config mapping with every term on and tapers disabled.

    Tapers off by default so a coefficient test measures the coefficient rather
    than the compression curve; individual tests turn them back on.
    """

    terms = {
        "slag_heat": {
            "enabled": True,
            "kg_coke_per_100kg_slag": 30.0,
            "max_abs_kg_thm": 45.0,
            "reference_source": "observed_dpr",
            "reference_fixed_kg_per_thm": 320.0,
        },
        "flux_calcination": {
            "enabled": True,
            "kg_coke_per_100kg_co2": 30.0,
            "max_abs_kg_thm": 25.0,
            "reference_source": "model_current",
        },
        "burden_oxygen": {
            "enabled": True,
            "kg_coke_per_kg_oxygen": 0.26,
            "max_abs_kg_thm": 20.0,
            "reference_source": "model_current",
        },
        "hot_metal_si": {
            "enabled": True,
            "kg_coke_per_0p1pct_si": 5.0,
            "max_abs_kg_thm": 15.0,
            "reference_source": "model_current",
        },
    }
    for term_id, patch in (overrides.pop("terms", None) or {}).items():
        terms.setdefault(term_id, {}).update(patch)

    block = {
        "enabled": True,
        "apply_to_objective": True,
        "apply_to_manual_blend": False,
        "guardrails": {
            "max_abs_correction_kg_thm": 60.0,
            "taper_start_fraction": 0.6,
            "coke_rate_band_kg_thm": [280.0, 420.0],
            "total_fuel_rate_band_kg_thm": [480.0, 680.0],
        },
        "terms": terms,
    }
    block.update(overrides)
    return {"coke_rate_correction": block}


def _reference(**overrides) -> CokeCorrectionReference:
    values = {
        "slag_rate_kg_per_thm": 320.0,
        "flux_co2_kg_per_thm": 4.0,
        "burden_oxygen_kg_per_thm": 380.0,
        "hot_metal_si_pct": 0.45,
    }
    values.update(overrides)
    return CokeCorrectionReference(**values)


def _drivers(**overrides) -> CokeCorrectionDrivers:
    values = {
        "slag_rate_kg_per_thm": 320.0,
        "flux_co2_kg_per_thm": 4.0,
        "burden_oxygen_kg_per_thm": 380.0,
        "hot_metal_si_pct": 0.45,
    }
    values.update(overrides)
    return CokeCorrectionDrivers(**values)


def _correct(drivers, *, settings=None, reference=None, anchor=_ANCHOR):
    return compute_coke_correction(
        anchor_coke_rate_kg_thm=anchor,
        anchor_nut_coke_rate_kg_thm=_NUT,
        anchor_pci_rate_kg_thm=_PCI,
        drivers=drivers,
        reference=reference if reference is not None else _reference(),
        settings=settings
        if settings is not None
        else load_coke_correction_settings(_cfg()),
    )


def _ore(
    ore_id: str,
    *,
    fe_t: float = 62.0,
    feo: float = 0.0,
    moisture: float = 0.0,
    price: float = 1000.0,
    sio2: float = 5.0,
) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=ore_id.upper(),
        stock_mt=5000.0,
        price_rs_per_mt=price,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(
            fe_t_pct=fe_t, moisture_pct=moisture, feo_pct=feo, sio2_pct=sio2
        ),
    )


def _flux(flux_id: str, *, loi: float, moisture: float = 0.0) -> FluxInput:
    return FluxInput(
        flux_id=flux_id,
        display_name=flux_id.upper(),
        enabled=True,
        wet_qty_mt=100.0,
        moisture_pct=moisture,
        cao_pct=52.0,
        loi_pct=loi,
        price_rs_per_mt=1800.0,
        stock_mt=1000.0,
        optimizable=True,
    )


# --------------------------------------------------------------------------
# The non-regression guarantee
# --------------------------------------------------------------------------


def test_correction_is_exactly_zero_at_the_reference_point():
    """Every term on, blend == reference: the correction must be bit-exact zero.

    This is the guarantee that nothing regresses on a normal run.
    """

    result = _correct(_drivers())

    assert result.enabled is True
    assert result.applied_delta_kg_thm == 0.0
    assert result.sum_raw_kg_thm == 0.0
    assert result.corrected_coke_rate_kg_thm == _ANCHOR
    assert result.corrected_total_fuel_rate_kg_thm == _ANCHOR + _NUT + _PCI
    assert result.band_bindings == []
    assert all(term.delta_kg_thm == 0.0 for term in result.terms)


def test_absent_config_block_is_a_no_op():
    settings = load_coke_correction_settings({})

    assert settings.enabled is False
    assert settings.apply_to_objective is False

    result = _correct(_drivers(slag_rate_kg_per_thm=500.0), settings=settings)

    assert result.enabled is False
    assert result.applied_delta_kg_thm == 0.0
    assert result.corrected_coke_rate_kg_thm == _ANCHOR
    assert all(term.disabled_reason for term in result.terms)


def test_none_config_is_a_no_op():
    assert load_coke_correction_settings(None).enabled is False


# --------------------------------------------------------------------------
# The shipped configuration
#
# Every other test in this file builds its own config dict, so none of them
# notice if setting_bmo.yml changes. These three read the real file. An earlier
# version of the slag coefficient was lost in a merge precisely because nothing
# asserted it.
# --------------------------------------------------------------------------


def _shipped_settings():
    import yaml

    path = Path(__file__).resolve().parents[1] / "src" / "config" / "setting_bmo.yml"
    return load_coke_correction_settings(
        yaml.safe_load(path.read_text(encoding="utf-8"))["bmo"]
    )


def test_shipped_slag_coefficient_is_the_empirically_anchored_value():
    """Pins the one coefficient this plant's data can actually measure.

    22 kg coke per 100 kg slag comes from a 222-day daily mass balance: six
    specifications spanning +18.4 to +26.9 (all p<0.001), bootstrap median
    +20.9, and an IV estimate of +20.0 once the shared coke-ash error is purged.
    See docs/bmo_fuel_slag_si_findings.md section 2.

    Changing this number is a decision about plant physics, not a tuning knob.
    If you are here because this test failed, update the derivation in
    setting_bmo.yml and the findings document alongside it.
    """

    slag = _shipped_settings().term(TERM_SLAG_HEAT)

    assert slag.enabled is True
    assert slag.k_config_value == pytest.approx(22.0)
    assert slag.k == pytest.approx(0.22)
    # Inside the range the 222-day mass balance supports.
    assert 11.0 <= slag.k_config_value <= 30.0


def test_shipped_config_is_enabled_and_priced_into_the_objective():
    settings = _shipped_settings()

    assert settings.enabled is True
    assert settings.apply_to_objective is True
    # The manual blend is the realised-cost reference and must stay untouched.
    assert settings.apply_to_manual_blend is False
    assert settings.term(TERM_FLUX_CALCINATION).enabled is True
    assert settings.term(TERM_HOT_METAL_SI).enabled is True
    # Modest and unvalidated; stays off until the two main terms are trusted.
    assert settings.term(TERM_BURDEN_OXYGEN).enabled is False


def test_shipped_guardrails_bound_the_reported_rates():
    settings = _shipped_settings()

    assert settings.max_abs_correction_kg_thm == pytest.approx(60.0)
    low, high = settings.coke_rate_band_kg_thm
    assert low < 300.0 < high
    fuel_low, fuel_high = settings.total_fuel_rate_band_kg_thm
    # The plant runs ~540 kg/THM total fuel; the band must contain that.
    assert fuel_low < 540.0 < fuel_high


# --------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------


def test_slag_term_coefficient_and_sign():
    """+100 kg/THM slag over reference costs exactly +30 kg/THM of coke."""

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = _correct(_drivers(slag_rate_kg_per_thm=420.0), settings=settings)

    assert result.applied_delta_kg_thm == pytest.approx(30.0)
    assert result.corrected_coke_rate_kg_thm == pytest.approx(330.0)

    slag_term = next(t for t in result.terms if t.term_id == TERM_SLAG_HEAT)
    assert slag_term.enabled is True
    assert slag_term.delta_kg_thm == pytest.approx(30.0)
    assert slag_term.x_blend == pytest.approx(420.0)
    assert slag_term.x_reference == pytest.approx(320.0)


def test_slag_term_is_symmetric_for_a_leaner_slag_burden():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = _correct(_drivers(slag_rate_kg_per_thm=270.0), settings=settings)

    assert result.applied_delta_kg_thm == pytest.approx(-15.0)


def test_si_term_coefficient():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"enabled": False},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
            }
        )
    )
    result = _correct(_drivers(hot_metal_si_pct=0.55), settings=settings)

    assert result.applied_delta_kg_thm == pytest.approx(5.0)


def test_si_term_contributes_near_zero_with_the_blend_flat_model():
    """The Si model swings only 0.371 -> 0.384 across a 2x sinter change.

    The term is shipped because the physics is right and it goes live the moment
    a blend-sensitive Si model is deployed. If someone later "fixes" its
    apparent uselessness by inflating the coefficient, this test fails and
    forces them to fix the model instead.
    """

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"enabled": False},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
            }
        )
    )
    result = _correct(
        _drivers(hot_metal_si_pct=0.384),
        settings=settings,
        reference=_reference(hot_metal_si_pct=0.371),
    )

    assert abs(result.applied_delta_kg_thm) < 1.0


def test_flux_term_uses_loi_not_total_flux_mass():
    """Equal flux mass, different LOI: only the carbonate moves the correction.

    This encodes the no-double-counting rule. The CaO residue of both rows
    already reaches the correction through slag mass; this term must charge only
    the CO2 that leaves as gas.
    """

    carbonate_co2 = compute_flux_co2_kg_per_thm(
        flux_inputs=[_flux("limestone", loi=40.8)], hot_metal_mt=2350.0
    )
    inert_co2 = compute_flux_co2_kg_per_thm(
        flux_inputs=[_flux("quartz", loi=0.0)], hot_metal_mt=2350.0
    )

    assert inert_co2 == 0.0
    # 100 wet MT at 40.8% LOI over 2,350 MT HM.
    assert carbonate_co2 == pytest.approx(100.0 * 0.408 * 1000.0 / 2350.0)

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    carbonate = _correct(
        _drivers(flux_co2_kg_per_thm=carbonate_co2),
        settings=settings,
        reference=_reference(flux_co2_kg_per_thm=0.0),
    )
    inert = _correct(
        _drivers(flux_co2_kg_per_thm=inert_co2),
        settings=settings,
        reference=_reference(flux_co2_kg_per_thm=0.0),
    )

    assert inert.applied_delta_kg_thm == 0.0
    assert carbonate.applied_delta_kg_thm == pytest.approx(0.30 * carbonate_co2)


def test_flux_co2_ignores_disabled_rows_and_removes_moisture():
    enabled = _flux("limestone", loi=40.0, moisture=5.0)
    disabled = _flux("dolomite", loi=45.0)
    disabled.enabled = False

    co2 = compute_flux_co2_kg_per_thm(
        flux_inputs=[enabled, disabled], hot_metal_mt=1000.0
    )

    assert co2 == pytest.approx(100.0 * 0.95 * 0.40 * 1000.0 / 1000.0)


def test_burden_oxygen_sinter_to_pellet_swing_is_modest():
    """A full sinter-to-pellet swing is only about 3.5 kg/THM of coke.

    Guards the unit conversion: if the driver ever silently changes scale this
    lands nowhere near its documented magnitude.
    """

    sinter = _ore("sinter", fe_t=56.0, feo=8.0)
    pellet = _ore("pellet", fe_t=64.0, feo=1.0)

    # Same Fe delivered by each, so only the oxide form differs.
    sinter_mt = 1000.0 * 0.62 / 0.56
    pellet_mt = 1000.0 * 0.62 / 0.64

    sinter_o = compute_burden_oxygen_kg_per_thm(
        ores=[sinter], quantities_mt={"sinter": sinter_mt}, hot_metal_mt=1000.0
    )
    pellet_o = compute_burden_oxygen_kg_per_thm(
        ores=[pellet], quantities_mt={"pellet": pellet_mt}, hot_metal_mt=1000.0
    )

    delta_o = pellet_o - sinter_o
    assert 5.0 < delta_o < 25.0, f"oxygen swing {delta_o:.1f} kg/THM out of scale"
    assert 0.26 * delta_o < 7.0


def test_burden_oxygen_fe2o3_carries_more_oxygen_than_feo():
    hematite = _ore("hematite", fe_t=60.0, feo=0.0)
    wustite = _ore("wustite", fe_t=60.0, feo=77.0)

    hematite_o = compute_burden_oxygen_kg_per_thm(
        ores=[hematite], quantities_mt={"hematite": 1000.0}, hot_metal_mt=1000.0
    )
    wustite_o = compute_burden_oxygen_kg_per_thm(
        ores=[wustite], quantities_mt={"wustite": 1000.0}, hot_metal_mt=1000.0
    )

    assert hematite_o > wustite_o
    assert hematite_o == pytest.approx(1000.0 * 0.60 * (48.0 / 111.69), rel=1e-3)


def test_burden_oxygen_caps_feo_borne_iron_at_total_iron():
    """A stale FeO reading must not manufacture negative Fe2O3-borne iron."""

    absurd = _ore("absurd", fe_t=50.0, feo=99.0)
    oxygen = compute_burden_oxygen_kg_per_thm(
        ores=[absurd], quantities_mt={"absurd": 1000.0}, hot_metal_mt=1000.0
    )

    assert oxygen == pytest.approx(1000.0 * 0.50 * (16.0 / 55.845), rel=1e-3)


def test_terms_are_additive():
    result = _correct(
        _drivers(slag_rate_kg_per_thm=350.0, hot_metal_si_pct=0.47),
    )

    # 30 kg/THM slag at 0.30, plus 0.02 %Si at 50 per %.
    assert result.applied_delta_kg_thm == pytest.approx(9.0 + 1.0)


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


def test_taper_is_monotone_and_bounded():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"envelope_halfwidth_kg_per_thm": 60.0},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )

    previous = -math.inf
    for slag in np.linspace(-600.0, 1200.0, 400):
        result = _correct(
            _drivers(slag_rate_kg_per_thm=float(slag)), settings=settings
        )
        delta = result.applied_delta_kg_thm
        assert delta >= previous - 1e-9, "correction must be monotone in slag rate"
        assert abs(delta) <= settings.max_abs_correction_kg_thm + 1e-9
        previous = delta


def test_taper_leaves_the_normal_operating_range_untouched():
    """A 900 MT/day slag campaign must still get the full physics, not a taper."""

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"envelope_halfwidth_kg_per_thm": 60.0},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    # 900 MT slag over 2,350 MT HM is ~383 kg/THM, +63 against a 320 reference.
    result = _correct(_drivers(slag_rate_kg_per_thm=383.0), settings=settings)

    assert result.taper_active is False
    assert result.applied_delta_kg_thm == pytest.approx(0.30 * 63.0)


def test_soft_saturate_is_continuous_at_the_start_point():
    assert soft_saturate(36.0, 36.0, 60.0) == pytest.approx(36.0)
    assert soft_saturate(36.001, 36.0, 60.0) == pytest.approx(36.001, abs=1e-3)
    assert soft_saturate(1e9, 36.0, 60.0) == pytest.approx(60.0)
    assert soft_saturate(-1e9, 36.0, 60.0) == pytest.approx(-60.0)


def test_total_clamp_binds_and_is_reported():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = _correct(_drivers(slag_rate_kg_per_thm=1500.0), settings=settings)

    assert result.total_clamp_binding is True
    assert abs(result.applied_delta_kg_thm) <= 60.0
    assert any("capped" in warning for warning in result.warnings)


def test_per_term_clamp_binds_and_is_reported():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"max_abs_kg_thm": 10.0},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = _correct(_drivers(slag_rate_kg_per_thm=520.0), settings=settings)

    slag_term = next(t for t in result.terms if t.term_id == TERM_SLAG_HEAT)
    assert slag_term.term_clamp_binding is True
    assert abs(slag_term.delta_kg_thm) <= 10.0


def test_a_band_never_manufactures_a_correction_from_out_of_band_anchor():
    """Regression: an out-of-band anchor once turned -7.8 into +74.8.

    A PCI tag reading 294.8 kg/THM crushed the back-solved coke rate to 205.2,
    below the 280 band floor. The band then clamped UP to 280 and the 74.8 kg/THM
    gap was reported to the operator as a physics correction. A guardrail must
    only ever cap what the physics asked for; it must never move a rate on its
    own, and it cannot repair a bad anchor.
    """

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                # Reproduce the shipped plant coefficient used in the incident.
                "slag_heat": {"kg_coke_per_100kg_slag": 22.0},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = compute_coke_correction(
        anchor_coke_rate_kg_thm=205.2,          # below the 280 band floor
        anchor_nut_coke_rate_kg_thm=70.0,
        anchor_pci_rate_kg_thm=294.8,           # the bad tag that caused it
        drivers=_drivers(slag_rate_kg_per_thm=298.5),
        reference=_reference(slag_rate_kg_per_thm=334.0),
        settings=settings,
    )

    physics = 0.22 * (298.5 - 334.0)
    assert result.sum_after_term_clamps_kg_thm == pytest.approx(physics, abs=1e-6)
    # The applied delta must equal the physics, not the distance to the band.
    assert result.applied_delta_kg_thm == pytest.approx(physics, abs=1e-6)
    assert result.applied_delta_kg_thm < 0.0
    assert result.corrected_coke_rate_kg_thm == pytest.approx(205.2 + physics)
    # And the operator is told the anchor itself is the problem.
    assert any("outside the expected" in w for w in result.warnings)


def test_a_plausible_anchor_still_gets_the_band_as_a_guardrail():
    """The stand-down must not disarm the band for normal anchors."""

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = compute_coke_correction(
        anchor_coke_rate_kg_thm=410.0,          # inside [280, 420]
        anchor_nut_coke_rate_kg_thm=70.0,
        anchor_pci_rate_kg_thm=170.0,
        drivers=_drivers(slag_rate_kg_per_thm=3000.0),
        reference=_reference(slag_rate_kg_per_thm=334.0),
        settings=settings,
    )

    assert "coke_rate" in result.band_bindings
    assert result.corrected_coke_rate_kg_thm == pytest.approx(420.0)
    assert result.applied_delta_kg_thm == pytest.approx(10.0)


def test_total_fuel_band_never_moves_an_out_of_band_anchor_on_its_own():
    """The total-fuel band obeys the same correction-only invariant."""

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"enabled": False},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = compute_coke_correction(
        anchor_coke_rate_kg_thm=300.0,
        anchor_nut_coke_rate_kg_thm=70.0,
        anchor_pci_rate_kg_thm=400.0,  # total 770, above the 680 band
        drivers=_drivers(),
        reference=_reference(),
        settings=settings,
    )

    assert result.sum_after_term_clamps_kg_thm == pytest.approx(0.0)
    assert result.applied_delta_kg_thm == pytest.approx(0.0)
    assert result.corrected_coke_rate_kg_thm == pytest.approx(300.0)
    assert result.corrected_total_fuel_rate_kg_thm == pytest.approx(770.0)
    assert result.band_bindings == []
    assert any("uncorrected total fuel rate" in w for w in result.warnings)


def test_output_band_binding_is_surfaced_not_silent():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = _correct(
        _drivers(slag_rate_kg_per_thm=1500.0), settings=settings, anchor=400.0
    )

    assert "coke_rate" in result.band_bindings
    assert result.corrected_coke_rate_kg_thm == pytest.approx(420.0)
    assert any("correction was capped" in warning for warning in result.warnings)


def test_total_fuel_band_binding_is_surfaced():
    settings = load_coke_correction_settings(
        _cfg(
            guardrails={
                "max_abs_correction_kg_thm": 60.0,
                "taper_start_fraction": 0.6,
                "coke_rate_band_kg_thm": [100.0, 900.0],
                "total_fuel_rate_band_kg_thm": [480.0, 560.0],
            },
            terms={
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            },
        )
    )
    result = _correct(_drivers(slag_rate_kg_per_thm=1500.0), settings=settings)

    assert "total_fuel_rate" in result.band_bindings
    assert result.corrected_total_fuel_rate_kg_thm == pytest.approx(560.0)


@pytest.mark.parametrize(
    (
        "case",
        "anchor",
        "nut",
        "pci",
        "slag",
        "expected_applied",
        "expected_coke",
        "expected_bindings",
        "warning_fragment",
    ),
    [
        (
            "normal_reference",
            300.0,
            70.0,
            170.0,
            334.0,
            0.0,
            300.0,
            (),
            None,
        ),
        (
            "reported_high_pci_incident",
            205.2,
            70.0,
            294.8,
            298.5,
            -7.81,
            197.39,
            (),
            "model-derived coke rate",
        ),
        (
            "low_coke_anchor_zero_physics",
            205.2,
            70.0,
            294.8,
            334.0,
            0.0,
            205.2,
            (),
            "model-derived coke rate",
        ),
        (
            "high_coke_anchor_positive_physics",
            430.0,
            70.0,
            100.0,
            384.0,
            11.0,
            441.0,
            (),
            "model-derived coke rate",
        ),
        (
            "plausible_coke_crosses_lower_edge",
            285.0,
            70.0,
            170.0,
            284.0,
            -5.0,
            280.0,
            ("coke_rate",),
            "correction was capped",
        ),
        (
            "plausible_coke_crosses_upper_edge",
            415.0,
            70.0,
            100.0,
            384.0,
            5.0,
            420.0,
            ("coke_rate",),
            "correction was capped",
        ),
        (
            "high_total_fuel_anchor_zero_physics",
            300.0,
            70.0,
            400.0,
            334.0,
            0.0,
            300.0,
            (),
            "uncorrected total fuel rate",
        ),
        (
            "low_total_fuel_anchor_zero_physics",
            280.0,
            0.0,
            0.0,
            334.0,
            0.0,
            280.0,
            (),
            "uncorrected total fuel rate",
        ),
        (
            "plausible_total_fuel_crosses_upper_edge",
            400.0,
            70.0,
            200.0,
            384.0,
            10.0,
            410.0,
            ("total_fuel_rate",),
            "total fuel",
        ),
        (
            "plausible_total_fuel_crosses_lower_edge",
            300.0,
            70.0,
            120.0,
            284.0,
            -10.0,
            290.0,
            ("total_fuel_rate",),
            "total fuel",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_guardrails_across_ten_operating_conditions(
    case,
    anchor,
    nut,
    pci,
    slag,
    expected_applied,
    expected_coke,
    expected_bindings,
    warning_fragment,
):
    """Ten boundary conditions pin the correction-only band invariant."""

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"kg_coke_per_100kg_slag": 22.0},
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    result = compute_coke_correction(
        anchor_coke_rate_kg_thm=anchor,
        anchor_nut_coke_rate_kg_thm=nut,
        anchor_pci_rate_kg_thm=pci,
        drivers=_drivers(slag_rate_kg_per_thm=slag),
        reference=_reference(slag_rate_kg_per_thm=334.0),
        settings=settings,
    )

    assert result.applied_delta_kg_thm == pytest.approx(expected_applied)
    assert result.corrected_coke_rate_kg_thm == pytest.approx(expected_coke)
    assert set(result.band_bindings) == set(expected_bindings)
    assert result.corrected_coke_rate_kg_thm == pytest.approx(
        anchor + result.applied_delta_kg_thm
    )
    assert result.corrected_total_fuel_rate_kg_thm == pytest.approx(
        result.corrected_coke_rate_kg_thm + nut + pci
    )

    coke_low, coke_high = settings.coke_rate_band_kg_thm
    if not coke_low <= anchor <= coke_high:
        assert "coke_rate" not in result.band_bindings, case
    fuel_low, fuel_high = settings.total_fuel_rate_band_kg_thm
    anchor_total = anchor + nut + pci
    if not fuel_low <= anchor_total <= fuel_high:
        assert "total_fuel_rate" not in result.band_bindings, case

    if warning_fragment is None:
        assert result.warnings == []
    else:
        assert any(warning_fragment in warning for warning in result.warnings), case


def test_correction_stays_in_band_across_an_extreme_sweep():
    settings = load_coke_correction_settings(_cfg())
    low, high = settings.coke_rate_band_kg_thm

    for slag in np.linspace(0.0, 2000.0, 120):
        for si in (0.0, 0.5, 2.0):
            result = _correct(
                _drivers(slag_rate_kg_per_thm=float(slag), hot_metal_si_pct=float(si))
            )
            assert math.isfinite(result.applied_delta_kg_thm)
            assert abs(result.applied_delta_kg_thm) <= 60.0 + 1e-9
            assert low - 1e-9 <= result.corrected_coke_rate_kg_thm <= high + 1e-9


# --------------------------------------------------------------------------
# Provenance and degraded inputs
# --------------------------------------------------------------------------


def test_disabled_term_reports_reason_not_silence():
    settings = load_coke_correction_settings(_cfg())
    result = _correct(
        _drivers(),
        settings=settings,
        reference=_reference(burden_oxygen_kg_per_thm=None),
    )

    oxygen = next(t for t in result.terms if t.term_id == TERM_BURDEN_OXYGEN)
    assert oxygen.enabled is False
    assert oxygen.disabled_reason == "no current-burden reference available"


def test_missing_anchor_yields_an_inactive_result_not_a_crash():
    result = _correct(_drivers(slag_rate_kg_per_thm=500.0), anchor=float("nan"))

    assert result.enabled is False
    assert result.applied_delta_kg_thm == 0.0


def test_build_reference_prefers_observed_slag_and_records_provenance():
    settings = load_coke_correction_settings(_cfg())
    reference = build_reference(
        settings=settings,
        observed_slag_rate_kg_per_thm=318.0,
        current_drivers=_drivers(slag_rate_kg_per_thm=325.0),
    )

    assert reference.slag_rate_kg_per_thm == pytest.approx(318.0)
    assert "observed_dpr" in reference.sources[TERM_SLAG_HEAT]
    assert reference.slag_basis_gap_kg_per_thm == pytest.approx(7.0)
    assert reference.warnings == []


def test_build_reference_warns_when_the_slag_basis_gap_is_large():
    settings = load_coke_correction_settings(_cfg())
    reference = build_reference(
        settings=settings,
        observed_slag_rate_kg_per_thm=300.0,
        current_drivers=_drivers(slag_rate_kg_per_thm=360.0),
    )

    assert reference.slag_basis_gap_kg_per_thm == pytest.approx(60.0)
    assert any(
        "slag_correction_factor" in warning for warning in reference.warnings
    )


def test_build_reference_falls_back_to_modelled_slag_with_a_warning():
    settings = load_coke_correction_settings(_cfg())
    reference = build_reference(
        settings=settings,
        observed_slag_rate_kg_per_thm=None,
        current_drivers=_drivers(slag_rate_kg_per_thm=333.0),
    )

    assert reference.slag_rate_kg_per_thm == pytest.approx(333.0)
    assert any("Observed DPR slag rate" in w for w in reference.warnings)


def test_reference_warnings_reach_the_correction_result():
    settings = load_coke_correction_settings(_cfg())
    reference = build_reference(
        settings=settings,
        observed_slag_rate_kg_per_thm=300.0,
        current_drivers=_drivers(slag_rate_kg_per_thm=360.0),
    )
    result = _correct(_drivers(), settings=settings, reference=reference)

    assert any("slag_correction_factor" in w for w in result.warnings)


# --------------------------------------------------------------------------
# with_coke_delta
# --------------------------------------------------------------------------


def _rates() -> EstimatedFuelRates:
    return EstimatedFuelRates(
        pci_rate_kg_thm=170.0,
        nut_coke_rate_kg_thm=70.0,
        total_coke_rate_kg_thm=370.0,
        coke_rate_kg_thm=300.0,
        total_fuel_rate_kg_thm=540.0,
        pci_source="test",
        nut_coke_source="test",
    )


def test_with_coke_delta_keeps_totals_in_step():
    updated = with_coke_delta(_rates(), 20.0)

    assert updated.coke_rate_kg_thm == pytest.approx(320.0)
    assert updated.total_coke_rate_kg_thm == pytest.approx(390.0)
    assert updated.total_fuel_rate_kg_thm == pytest.approx(560.0)
    # Nut coke and PCI are operator run inputs and must not move.
    assert updated.nut_coke_rate_kg_thm == pytest.approx(70.0)
    assert updated.pci_rate_kg_thm == pytest.approx(170.0)
    assert updated.nut_coke_source == "test"


def test_with_coke_delta_floors_coke_at_zero():
    updated = with_coke_delta(_rates(), -500.0)

    assert updated.coke_rate_kg_thm == 0.0
    assert updated.total_fuel_rate_kg_thm == pytest.approx(240.0)


# --------------------------------------------------------------------------
# LP linearisation
# --------------------------------------------------------------------------


def test_lp_coefficients_are_zero_when_the_correction_is_off_objective():
    settings = load_coke_correction_settings(_cfg(apply_to_objective=False))
    coeffs = build_linear_coke_correction_cost_coeffs(
        ores=[_ore("a"), _ore("b")],
        variable_fluxes=[_flux("limestone", loi=40.8)],
        settings=settings,
        slag_coeff=np.array([0.1, 0.2, 0.5]),
        hot_metal_target_mt=2350.0,
    )

    assert np.allclose(coeffs, 0.0)


def test_lp_slag_coefficient_prices_marginal_slag_at_the_baseline_coke_price():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "flux_calcination": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    slag_coeff = np.array([0.10, 0.25, 0.50])
    coeffs = build_linear_coke_correction_cost_coeffs(
        ores=[_ore("a"), _ore("b")],
        variable_fluxes=[_flux("limestone", loi=40.8)],
        settings=settings,
        slag_coeff=slag_coeff,
        hot_metal_target_mt=2350.0,
    )

    price = ASSUMED_FUEL_PRICES_RS_PER_KG["coke"]
    assert np.allclose(coeffs, 1000.0 * 0.30 * price * slag_coeff)
    # An ore making 0.10 MT of slag per wet MT picks up 840 Rs/MT of fuel cost,
    # which is real against ore prices in the 1,000-8,000 Rs/MT band.
    assert coeffs[0] == pytest.approx(840.0)


def test_lp_flux_calcination_coefficient_reprices_limestone_materially():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"enabled": False},
                "burden_oxygen": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    coeffs = build_linear_coke_correction_cost_coeffs(
        ores=[_ore("a")],
        variable_fluxes=[_flux("limestone", loi=40.8), _flux("quartz", loi=0.0)],
        settings=settings,
        slag_coeff=np.array([0.1, 0.5, 0.4]),
        hot_metal_target_mt=2350.0,
    )

    assert coeffs[0] == 0.0  # ore chemistry carries no LOI
    assert coeffs[2] == 0.0  # quartz releases no CO2
    # 1,000 kg x 40.8% CO2 x 0.30 kg coke/kg CO2 x 28 Rs/kg.
    assert coeffs[1] == pytest.approx(1000.0 * 0.408 * 0.30 * 28.0)
    assert coeffs[1] > 3000.0


def test_lp_burden_oxygen_coefficient_favours_feo_bearing_burden():
    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"enabled": False},
                "flux_calcination": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    coeffs = build_linear_coke_correction_cost_coeffs(
        ores=[_ore("pellet", fe_t=64.0, feo=1.0), _ore("sinter", fe_t=56.0, feo=8.0)],
        variable_fluxes=None,
        settings=settings,
        slag_coeff=np.array([0.1, 0.2]),
        hot_metal_target_mt=2350.0,
    )

    assert coeffs[0] > coeffs[1] > 0.0


def test_lp_coefficients_match_the_nonlinear_gradient():
    """The LP must optimise the same physics the UI reports.

    Perturbs one wet MT of each column, recomputes the nonlinear per-THM
    correction cost by hand, and checks the finite difference equals the LP
    coefficient. A units slip in either path shows up immediately.
    """

    settings = load_coke_correction_settings(
        _cfg(
            terms={
                "slag_heat": {"enabled": False},
                "hot_metal_si": {"enabled": False},
            }
        )
    )
    ores = [_ore("pellet", fe_t=64.0, feo=1.0, moisture=2.0)]
    fluxes = [_flux("limestone", loi=40.8, moisture=3.0)]
    hot_metal_mt = 2350.0
    price = ASSUMED_FUEL_PRICES_RS_PER_KG["coke"]

    coeffs = build_linear_coke_correction_cost_coeffs(
        ores=ores,
        variable_fluxes=fluxes,
        settings=settings,
        slag_coeff=np.zeros(2),
        hot_metal_target_mt=hot_metal_mt,
    )

    def total_cost(ore_mt: float, flux_mt: float) -> float:
        oxygen = compute_burden_oxygen_kg_per_thm(
            ores=ores, quantities_mt={"pellet": ore_mt}, hot_metal_mt=hot_metal_mt
        )
        flux_row = _flux("limestone", loi=40.8, moisture=3.0)
        flux_row.wet_qty_mt = flux_mt
        co2 = compute_flux_co2_kg_per_thm(
            flux_inputs=[flux_row], hot_metal_mt=hot_metal_mt
        )
        delta_coke = 0.26 * oxygen + 0.30 * co2
        return hot_metal_mt * price * delta_coke

    base = total_cost(1000.0, 100.0)
    assert (total_cost(1001.0, 100.0) - base) == pytest.approx(coeffs[0], rel=1e-6)
    assert (total_cost(1000.0, 101.0) - base) == pytest.approx(coeffs[1], rel=1e-6)


def test_lp_coefficients_are_zero_without_a_hot_metal_basis():
    settings = load_coke_correction_settings(_cfg())
    coeffs = build_linear_coke_correction_cost_coeffs(
        ores=[_ore("a")],
        variable_fluxes=None,
        settings=settings,
        slag_coeff=np.array([0.3]),
        hot_metal_target_mt=0.0,
    )

    assert np.allclose(coeffs, 0.0)
