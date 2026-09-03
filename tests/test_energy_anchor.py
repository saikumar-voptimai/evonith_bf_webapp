"""The energy-balance coke anchor that sets the BMO fuel cost level.

The anchor replaces a number the operator can see (the Fuel Ash coke rate) with
one they cannot check by eye. That makes its failure modes the whole story: if
it ever returns a confident wrong number instead of declining, the fuel cost is
wrong by 28 Rs for every kg/THM and nothing on the page says so.

These tests pin the declining.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from utils.bmo.coke_calibration import NO_CALIBRATION, CokeCalibration
from utils.bmo.energy_anchor import PLAUSIBLE_COKE_RATE_KG_THM, solve_energy_anchor
from utils.bmo.fuel_rates import EstimatedFuelRates, with_coke_rate

GOOD_CALIBRATION = CokeCalibration(
    offset_kg_per_thm=19.7, sample_days=90, residual_sd_kg_per_thm=12.0,
    window_days=90, fitted_on="2026-08-30",
)


def _args(**overrides):
    base = dict(
        quantities_mt={"ore_a": 100.0},
        ores=[],
        slag_mt=90.0,
        hot_metal_mt=300.0,
        fuel_rates_kg_per_thm={
            "coke_rate_kg_thm": 330.0,
            "nut_coke_rate_kg_thm": 70.0,
            "pci_rate_kg_thm": 150.0,
        },
        hm_chemistry={"carbon_pct": 4.3, "silicon_pct": 0.5},
        process_snapshot={"hot_blast_vol_nm3h": 120_000.0},
        calibration=GOOD_CALIBRATION,
    )
    base.update(overrides)
    return base


def _solving(value: float):
    """Patch the balance to return ``value``, leaving everything else real."""

    return patch(
        "utils.energy_balance.solve.solve_coke_rate_kg_per_thm",
        return_value=value,
    )


# --- the happy path ------------------------------------------------------------


def test_it_subtracts_the_offset_from_the_raw_balance():
    with _solving(350.0), patch(
        "utils.bmo.process_recommendation.blend_to_energy_inputs", return_value=object()
    ):
        anchor = solve_energy_anchor(**_args())

    assert anchor.usable
    assert anchor.raw_coke_rate_kg_thm == pytest.approx(350.0)
    assert anchor.coke_rate_kg_thm == pytest.approx(330.3)
    assert anchor.offset_kg_per_thm == pytest.approx(19.7)


# --- every way it must decline -------------------------------------------------


def test_no_live_blast_tags_declines():
    """Without a blast volume there is no balance to solve, only a guess."""

    anchor = solve_energy_anchor(**_args(process_snapshot={}))

    assert not anchor.usable
    assert any("blast" in note for note in anchor.notes)


def test_zero_hot_metal_basis_declines():
    anchor = solve_energy_anchor(**_args(hot_metal_mt=0.0))

    assert not anchor.usable


def test_a_solver_failure_declines_rather_than_propagating():
    """A blend evaluation must never be broken by the anchor failing."""

    with patch(
        "utils.bmo.process_recommendation.blend_to_energy_inputs",
        side_effect=ValueError("bad inputs"),
    ):
        anchor = solve_energy_anchor(**_args())

    assert not anchor.usable
    assert any("did not solve" in note for note in anchor.notes)


@pytest.mark.parametrize("diverged", [PLAUSIBLE_COKE_RATE_KG_THM[0] - 1.0,
                                      PLAUSIBLE_COKE_RATE_KG_THM[1] + 1.0])
def test_an_implausible_solve_declines(diverged):
    """A balance that lands at 40 or 4,000 kg/THM has diverged, not solved.

    Without this the offset would be applied to nonsense and the result would
    still look like a coke rate.
    """

    with _solving(diverged), patch(
        "utils.bmo.process_recommendation.blend_to_energy_inputs", return_value=object()
    ):
        anchor = solve_energy_anchor(**_args())

    assert not anchor.usable
    assert any("plausible" in note for note in anchor.notes)


def test_without_a_calibration_it_declines_rather_than_shipping_the_raw_figure():
    """The raw balance runs ~20 kg/THM high — about 550 Rs/THM of fuel cost.

    Reporting it uncorrected would be worse than falling back to the observed
    rate, which at least is a measurement.
    """

    with _solving(350.0), patch(
        "utils.bmo.process_recommendation.blend_to_energy_inputs", return_value=object()
    ):
        anchor = solve_energy_anchor(**_args(calibration=NO_CALIBRATION))

    assert not anchor.usable
    assert any("not used as the cost anchor" in note for note in anchor.notes)


def test_a_stale_calibration_is_still_used_but_says_so():
    """Stale is worse than fresh and far better than nothing.

    Declining here would drop the page back to the observed anchor over a
    calibration that is merely a fortnight old, which is a bigger error than the
    drift it is worried about.
    """

    stale = CokeCalibration(
        offset_kg_per_thm=19.7, sample_days=90, residual_sd_kg_per_thm=12.0,
        window_days=90, fitted_on="2026-01-01",
    )
    with _solving(350.0), patch(
        "utils.bmo.process_recommendation.blend_to_energy_inputs", return_value=object()
    ):
        anchor = solve_energy_anchor(**_args(calibration=stale))

    assert anchor.usable
    assert any("days old" in note for note in anchor.notes)


# --- the rate substitution it feeds --------------------------------------------


def test_replacing_the_coke_rate_leaves_the_operator_set_fuels_alone():
    """Nut coke and PCI are run inputs; only coke is furnace demand."""

    rates = EstimatedFuelRates(
        pci_rate_kg_thm=150.0, nut_coke_rate_kg_thm=70.0,
        total_coke_rate_kg_thm=400.0, coke_rate_kg_thm=330.0,
        total_fuel_rate_kg_thm=550.0, pci_source="x", nut_coke_source="y",
    )

    moved = with_coke_rate(rates, 288.0)

    assert moved.coke_rate_kg_thm == pytest.approx(288.0)
    assert moved.nut_coke_rate_kg_thm == pytest.approx(70.0)
    assert moved.pci_rate_kg_thm == pytest.approx(150.0)
    assert moved.total_coke_rate_kg_thm == pytest.approx(358.0)
    assert moved.total_fuel_rate_kg_thm == pytest.approx(508.0)


def test_a_negative_coke_rate_is_clamped_not_propagated():
    rates = EstimatedFuelRates(
        pci_rate_kg_thm=150.0, nut_coke_rate_kg_thm=70.0,
        total_coke_rate_kg_thm=400.0, coke_rate_kg_thm=330.0,
        total_fuel_rate_kg_thm=550.0, pci_source="x", nut_coke_source="y",
    )

    assert with_coke_rate(rates, -5.0).coke_rate_kg_thm == 0.0
