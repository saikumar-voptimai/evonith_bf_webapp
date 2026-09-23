"""Coke rate across the full PCI range: model inside, anchored outside.

The optimal-coke-rate model is fitted on 135-208 kg/tHM of PCI, with half the
record between 175 and 190. A UI number box accepts 0 and 250, and a tree model
asked to extrapolate there will answer confidently and arbitrarily.

So outside 150-210 the answer is interpolated to the process team's operating
anchors instead - 460 kg/tHM of coke at zero PCI, 275 at 250 - and the thing
that must hold is that the two pieces MEET. An operator seeing one answer at
149 and a different one at 151 would rightly stop trusting the panel.
"""

from __future__ import annotations

import pytest

from utils.bmo.pci_anchoring import (
    ANCHOR_HIGH_COKE, ANCHOR_HIGH_PCI, ANCHOR_LOW_COKE, ANCHOR_LOW_PCI,
    MODEL_PCI_MAX, MODEL_PCI_MIN, NUT_COKE_KG_THM,
    anchored_coke_rate, substitution_ratio,
)


def model(pci: float) -> float:
    """Stand-in for the optimal-coke-rate model, plausible over its band."""

    return 460.0 - 0.78 * float(pci)


def coke(pci: float) -> float:
    return anchored_coke_rate(pci, model).coke_kg_per_thm


# --- the joins must not step ------------------------------------------------------


@pytest.mark.parametrize("edge", [MODEL_PCI_MIN, MODEL_PCI_MAX])
def test_the_curve_is_continuous_at_the_band_edges(edge):
    """The whole design rests on this: interpolation ends on the model's value."""

    inside = coke(edge)
    just_below = coke(edge - 0.25)
    just_above = coke(edge + 0.25)

    assert inside == pytest.approx(model(edge), abs=1e-9)
    assert just_below == pytest.approx(inside, abs=0.5)
    assert just_above == pytest.approx(inside, abs=0.5)


def test_the_model_is_used_inside_its_band_and_not_outside():
    assert anchored_coke_rate(180.0, model).basis == "model"
    assert anchored_coke_rate(MODEL_PCI_MIN, model).basis == "model"
    assert anchored_coke_rate(MODEL_PCI_MAX, model).basis == "model"
    assert anchored_coke_rate(120.0, model).basis == "interpolated_low"
    assert anchored_coke_rate(230.0, model).basis == "interpolated_high"


def test_a_caller_can_tell_a_modelled_number_from_an_interpolated_one():
    """Otherwise the UI cannot caveat the figures that deserve caveating."""

    assert anchored_coke_rate(180.0, model).is_model
    assert not anchored_coke_rate(100.0, model).is_model
    assert anchored_coke_rate(100.0, model).note


# --- the anchors ------------------------------------------------------------------


def test_zero_pci_lands_on_the_all_coke_anchor():
    result = anchored_coke_rate(ANCHOR_LOW_PCI, model)

    assert result.coke_kg_per_thm == pytest.approx(ANCHOR_LOW_COKE)
    assert result.nut_coke_kg_per_thm == pytest.approx(NUT_COKE_KG_THM)
    assert result.total_fuel_kg_per_thm == pytest.approx(530.0)


def test_max_pci_lands_on_the_high_injection_anchor():
    result = anchored_coke_rate(ANCHOR_HIGH_PCI, model)

    assert result.coke_kg_per_thm == pytest.approx(ANCHOR_HIGH_COKE)
    assert result.total_fuel_kg_per_thm == pytest.approx(595.0)


def test_nut_coke_is_fixed_and_does_not_move_with_pci():
    rates = {anchored_coke_rate(p, model).nut_coke_kg_per_thm
             for p in (0, 50, 150, 180, 210, 250)}

    assert rates == {NUT_COKE_KG_THM}


# --- shape over the whole range ---------------------------------------------------


def test_coke_falls_monotonically_across_the_entire_range():
    """Including across both joins, which is where a spline could overshoot."""

    values = [coke(p) for p in range(0, 251, 5)]

    for earlier, later in zip(values, values[1:]):
        assert later <= earlier + 1e-9


def test_total_fuel_stays_inside_the_expected_envelope():
    """The process team expects 530-595 kg/tHM across 0-250 PCI."""

    totals = [anchored_coke_rate(p, model).total_fuel_kg_per_thm
              for p in range(0, 251, 5)]

    assert min(totals) == pytest.approx(530.0, abs=1.0)
    assert max(totals) == pytest.approx(595.0, abs=1.0)


def test_interpolation_never_overshoots_its_endpoints():
    """Linear was chosen over a spline precisely to guarantee this."""

    edge_low = model(MODEL_PCI_MIN)
    for pci in range(0, int(MODEL_PCI_MIN), 5):
        assert edge_low - 1e-9 <= coke(pci) <= ANCHOR_LOW_COKE + 1e-9

    edge_high = model(MODEL_PCI_MAX)
    for pci in range(int(MODEL_PCI_MAX) + 1, 251, 5):
        assert ANCHOR_HIGH_COKE - 1e-9 <= coke(pci) <= edge_high + 1e-9


# --- outside the physical range ---------------------------------------------------


def test_impossible_pci_is_clamped_and_says_so():
    """A number box accepts anything. It must not be answered seriously."""

    low = anchored_coke_rate(-50.0, model)
    high = anchored_coke_rate(400.0, model)

    assert low.basis == "clamped" and low.note
    assert high.basis == "clamped" and high.note
    assert low.coke_kg_per_thm == pytest.approx(ANCHOR_LOW_COKE)
    assert high.coke_kg_per_thm == pytest.approx(ANCHOR_HIGH_COKE)


def test_the_model_is_never_called_outside_its_band():
    """Extrapolating a tree model is the failure this module exists to prevent."""

    seen: list[float] = []

    def spy(pci: float) -> float:
        seen.append(pci)
        return model(pci)

    for pci in (0, 25, 100, 149, 150, 180, 210, 211, 240, 250, 400):
        anchored_coke_rate(pci, spy)

    assert seen, "the model should still be consulted at the band edges"
    assert min(seen) >= MODEL_PCI_MIN - 1e-9
    assert max(seen) <= MODEL_PCI_MAX + 1e-9


# --- the ratio is reported, not assumed -------------------------------------------


def test_the_substitution_ratio_is_local_not_a_single_constant():
    """Plant history gives -0.83 at 168 falling to -0.74 at 197. A constant
    cannot express that, which is why this is a function of PCI."""

    inside = substitution_ratio(180.0, model)
    above = substitution_ratio(230.0, model)

    assert inside == pytest.approx(-0.78, abs=0.02)
    # Past the band the curve bends toward the anchor, so the ratio changes.
    assert above != pytest.approx(inside, abs=0.05)
    assert above < 0.0
