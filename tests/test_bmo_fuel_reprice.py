"""Tests for re-pricing the predicted fuel cost at the operator's current prices.

The fuel model predicts unit fuel cost at baseline prices (see
``ASSUMED_FUEL_PRICES_RS_PER_KG``). Optimized (LP/DE) blends decompose that
model-predicted cost back into physical rates at the baseline prices and re-price
them at the operator's current per-MT prices (``fuel_rate_basis="model_cost"``,
the default) — keeping the displayed cost blend-sensitive. The manual blend uses
``fuel_rate_basis="inputs"``: the actual current rates from the Fuel Ash table
(seeded from the latest static-dataset row), re-priced at current prices. This is
display-only: the optimizer-facing
``objective_rs_per_thm`` / ``fuel_cost_per_thm_rs`` keep the baseline-price value,
while the re-priced cost is stored in ``blend.diagnostics``.
"""

from __future__ import annotations

import pytest

from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.fuel_rates import (
    ASSUMED_FUEL_PRICES_RS_PER_KG,
    estimate_fuel_rates_from_cost,
    estimate_fuel_rates_from_inputs,
)
from utils.bmo.types import FuelAshInput, OreChemistry, OreInput

# Recent PCI + nut coke rates so the decomposition is fully determined.
_CTX = {"PCI_KG/THM": 150.0, "NUT COKE RATE KG/THM": 20.0}


class _FakePrediction:
    def __init__(self, value: float) -> None:
        self.value = value
        self.details: dict = {}
        self.used_fallback = False


class _FakeModelService:
    """Constant baseline-price fuel cost so the test needs no model/DB."""

    def __init__(self, value: float = 12_900.0) -> None:
        self.value = value

    def predict(self, feature_payload, history_df):  # noqa: ANN001
        return _FakePrediction(self.value)


def _ore(ore_id: str, *, sio2: float, cao: float) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=ore_id.upper(),
        stock_mt=5000.0,
        price_rs_per_mt=1000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=62.0, moisture_pct=3.0, sio2_pct=sio2, cao_pct=cao),
    )


def _fuel_ash(
    coke_mt: float,
    nut_mt: float,
    pci_mt: float,
    *,
    coke_rate: float = 340.0,
    nut_rate: float = 20.0,
    pci_rate: float = 150.0,
) -> list[FuelAshInput]:
    return [
        FuelAshInput(fuel_id="coke", display_name="Coke", price_rs_per_mt=coke_mt,
                     rate_kg_per_thm=coke_rate, ash_pct=11.0, sio2_pct=55.0),
        FuelAshInput(fuel_id="nut_coke", display_name="Nut Coke", price_rs_per_mt=nut_mt,
                     rate_kg_per_thm=nut_rate, ash_pct=11.0, sio2_pct=55.0),
        FuelAshInput(fuel_id="pci", display_name="PCI", price_rs_per_mt=pci_mt,
                     rate_kg_per_thm=pci_rate, ash_pct=9.0, sio2_pct=49.0),
    ]


def _blend(
    fuel_ash: list[FuelAshInput],
    *,
    model_value: float = 12_900.0,
    fuel_rate_basis: str = "model_cost",
):
    return evaluate_blend_with_fuel_prediction(
        ores=[_ore("ore_a", sio2=8.0, cao=1.0), _ore("ore_b", sio2=7.0, cao=1.5)],
        quantities_mt={"ore_a": 100.0, "ore_b": 100.0},
        feo_in_slag_pct=0.0,
        model_service=_FakeModelService(model_value),
        process_context=_CTX,
        history_df=None,
        fuel_ash_inputs=fuel_ash,
        hot_metal_target_mt=100.0,
        fuel_rate_basis=fuel_rate_basis,
    )


def test_decomposition_round_trips_at_assumed_prices():
    # Recovered rates, re-costed at the assumed prices, must reproduce the cost.
    rates = estimate_fuel_rates_from_cost(fuel_cost_per_thm_rs=12_900.0, process_context=_CTX)
    assert rates is not None
    assert rates.pci_rate_kg_thm == pytest.approx(150.0)
    assert rates.nut_coke_rate_kg_thm == pytest.approx(20.0)
    recomputed = (
        rates.coke_rate_kg_thm * ASSUMED_FUEL_PRICES_RS_PER_KG["coke"]
        + rates.nut_coke_rate_kg_thm * ASSUMED_FUEL_PRICES_RS_PER_KG["nut_coke"]
        + rates.pci_rate_kg_thm * ASSUMED_FUEL_PRICES_RS_PER_KG["pci"]
    )
    assert recomputed == pytest.approx(12_900.0)


def test_fuel_ash_input_rates_are_read_verbatim():
    rates = estimate_fuel_rates_from_inputs(
        _fuel_ash(
            28_000.0,
            24_000.0,
            15_000.0,
            coke_rate=318.0,
            nut_rate=37.0,
            pci_rate=71.0,
        )
    )

    assert rates is not None
    assert rates.coke_rate_kg_thm == pytest.approx(318.0)
    assert rates.nut_coke_rate_kg_thm == pytest.approx(37.0)
    assert rates.pci_rate_kg_thm == pytest.approx(71.0)


def test_reprice_stored_in_diagnostics_and_optimizer_untouched():
    # Coke 30/nut 24/PCI 20 Rs/kg (nut unchanged, coke + PCI raised vs 28/24/18).
    blend = _blend(_fuel_ash(30_000.0, 24_000.0, 20_000.0))
    rates = blend.diagnostics["fuel_rate_estimate"]
    expected = (
        rates["coke_rate_kg_thm"] * 30.0
        + rates["nut_coke_rate_kg_thm"] * 24.0
        + rates["pci_rate_kg_thm"] * 20.0
    )
    assert blend.diagnostics["adjusted_fuel_cost_per_thm_rs"] == pytest.approx(expected)
    # Raising coke + PCI prices lifts the displayed cost above the baseline.
    assert blend.diagnostics["adjusted_fuel_cost_per_thm_rs"] > 12_900.0
    # Optimizer-facing fields keep the baseline-price model cost (display-only).
    assert blend.fuel_cost_per_thm_rs == pytest.approx(12_900.0)
    ore_cost = blend.objective_rs_per_thm - blend.fuel_cost_per_thm_rs
    assert blend.diagnostics["adjusted_objective_rs_per_thm"] == pytest.approx(
        ore_cost + blend.diagnostics["adjusted_fuel_cost_per_thm_rs"]
    )


def test_default_basis_converts_model_cost_and_stays_blend_sensitive():
    # Optimized blends (default basis): rates are back-solved from the model's
    # predicted cost at baseline prices, so a different prediction changes the
    # re-priced display even with identical Fuel Ash inputs.
    fuel_ash = _fuel_ash(30_000.0, 24_000.0, 20_000.0)
    blend_a = _blend(fuel_ash, model_value=12_900.0)
    blend_b = _blend(fuel_ash, model_value=13_793.0)

    assert blend_a.diagnostics["fuel_rate_estimate_source"] == "model_cost_residual"
    assert blend_b.diagnostics["fuel_rate_estimate_source"] == "model_cost_residual"
    # PCI + nut coke rates come from recent context; coke is the cost residual.
    rates_a = blend_a.diagnostics["fuel_rate_estimate"]
    assert rates_a["pci_rate_kg_thm"] == pytest.approx(150.0)
    assert rates_a["nut_coke_rate_kg_thm"] == pytest.approx(20.0)
    assert (
        blend_b.diagnostics["adjusted_fuel_cost_per_thm_rs"]
        > blend_a.diagnostics["adjusted_fuel_cost_per_thm_rs"]
    )


def test_inputs_basis_uses_actual_current_rates_not_model_prediction():
    # Manual blend: realised cost from the actual current rates in the Fuel Ash
    # table, regardless of what the model predicts for the blend.
    blend = _blend(
        _fuel_ash(
            28_000.0,
            24_000.0,
            15_000.0,
            coke_rate=318.0,
            nut_rate=37.0,
            pci_rate=71.0,
        ),
        model_value=13_793.0,
        fuel_rate_basis="inputs",
    )

    expected = 318.0 * 28.0 + 37.0 * 24.0 + 71.0 * 15.0
    assert blend.fuel_cost_per_thm_rs == pytest.approx(13_793.0)
    assert blend.diagnostics["fuel_rate_estimate_source"] == "fuel_ash_inputs"
    assert blend.diagnostics["adjusted_fuel_cost_per_thm_rs"] == pytest.approx(expected)
    assert blend.diagnostics["adjusted_fuel_cost_per_thm_rs"] < 13_793.0


def test_missing_prices_fall_back_to_assumed_no_change():
    # All prices left at 0 -> assumed prices. With the default model-cost basis
    # and baseline prices, the re-priced cost round-trips to the prediction.
    blend = _blend(_fuel_ash(0.0, 0.0, 0.0))
    assert blend.diagnostics["adjusted_fuel_cost_per_thm_rs"] == pytest.approx(
        12_900.0
    )
    assert (
        blend.diagnostics["current_fuel_prices_rs_per_kg"]
        == ASSUMED_FUEL_PRICES_RS_PER_KG
    )


def test_missing_prices_inputs_basis_uses_editor_rates_at_assumed_prices():
    blend = _blend(_fuel_ash(0.0, 0.0, 0.0), fuel_rate_basis="inputs")
    expected = 340.0 * 28.0 + 20.0 * 24.0 + 150.0 * 18.0
    assert blend.diagnostics["adjusted_fuel_cost_per_thm_rs"] == pytest.approx(expected)
    assert (
        blend.diagnostics["current_fuel_prices_rs_per_kg"]
        == ASSUMED_FUEL_PRICES_RS_PER_KG
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
