"""Blend selection and fuel estimation are two stages, and stage 2 is read-only.

Stage 1 - the LP picks the cheapest ore + flux basket that satisfies the slag
rate, basicity, T-basicity, Al2O3, MgO and MgO/Al2O3 limits. Slag comes from the
blend plus the fuel-ash rates the operator entered. Nothing else is in the
objective.

Stage 2 - the ML model predicts fuel unit cost for that blend, a coke rate is
back-solved from it against the table's prices, and the physics correction is
applied. This is REPORTING. It must not move the slag the LP agreed to.

The old behaviour re-ran the slag balance with the corrected coke rate, so the
displayed basicity was not the basicity the LP had solved for - up to ~0.04 on
B2. DE still opts into that feedback because its objective genuinely prices
corrected fuel; the LP display path does not.
"""

from __future__ import annotations

import pytest

from utils.bmo.calculations import evaluate_blend
from utils.bmo.coke_correction import load_coke_correction_settings
from utils.bmo.fuel_rates import estimate_fuel_rates_from_cost
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.types import FuelAshInput, OreChemistry, OreInput

_HM_MT = 100.0


def _correction_settings(k_slag: float = 30.0):
    """Settings with a live slag term, so pricing them actually moves the LP.

    A bare ``CokeCorrectionSettings()`` has every term disabled and would make
    these tests pass without proving anything.
    """

    return load_coke_correction_settings(
        {
            "coke_rate_correction": {
                "enabled": True,
                "apply_to_objective": True,
                "guardrails": {
                    "max_abs_correction_kg_thm": 60.0,
                    "taper_start_fraction": 0.6,
                    "coke_rate_band_kg_thm": [200.0, 500.0],
                    "total_fuel_rate_band_kg_thm": [400.0, 800.0],
                },
                "terms": {
                    "slag_heat": {
                        "enabled": True,
                        "kg_coke_per_100kg_slag": k_slag,
                        "max_abs_kg_thm": 45.0,
                        "reference_source": "model_current",
                    }
                },
            }
        }
    )


def _ore(ore_id: str, *, fe_t: float, sio2: float, price: float) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=ore_id.upper(),
        stock_mt=10_000.0,
        price_rs_per_mt=price,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(
            fe_t_pct=fe_t, sio2_pct=sio2, al2o3_pct=2.0, cao_pct=4.0, mgo_pct=1.5
        ),
    )


def _fuels(coke_rate: float = 300.0) -> list[FuelAshInput]:
    return [
        FuelAshInput(
            fuel_id="coke", display_name="Coke", enabled=True,
            rate_kg_per_thm=coke_rate, price_rs_per_mt=28_000.0,
            ash_pct=12.0, sio2_pct=57.0, al2o3_pct=26.0, cao_pct=2.7, mgo_pct=1.0,
        ),
        FuelAshInput(
            fuel_id="nut_coke", display_name="Nut Coke", enabled=True,
            rate_kg_per_thm=76.44, price_rs_per_mt=24_000.0,
            ash_pct=12.5, sio2_pct=57.0, al2o3_pct=27.0, cao_pct=2.7, mgo_pct=1.0,
        ),
        FuelAshInput(
            fuel_id="pci", display_name="PCI", enabled=True,
            rate_kg_per_thm=180.0, price_rs_per_mt=18_000.0,
            ash_pct=8.6, sio2_pct=50.0, al2o3_pct=28.0, cao_pct=5.2, mgo_pct=1.9,
        ),
    ]


def _lp(**kwargs):
    ores = [
        _ore("lean", fe_t=52.0, sio2=9.0, price=900.0),
        _ore("rich", fe_t=64.0, sio2=3.0, price=1600.0),
    ]
    call = dict(
        target_production_mt=60.0,
        target_slag_qty_mt=400.0,
        feo_in_slag_pct=0.0,
        hot_metal_target_mt=_HM_MT,
    )
    call.update(kwargs)
    return run_lp_baseline(ores, **call)


# --- Stage 1: the LP objective is ore + flux purchase cost, nothing else -----


def test_lp_objective_is_ore_cost_only_by_default() -> None:
    """A coke-correction setting must not change the answer unless asked for.

    The LP's job is the cheapest basket that satisfies the slag limits. Slag is
    governed by the hard constraints, not by a fuel term in the cost vector.
    """

    plain, errors_plain = _lp()
    with_settings, errors_settings = _lp(
        coke_correction_settings=_correction_settings()
    )

    assert errors_plain == [] and errors_settings == []
    assert plain is not None and with_settings is not None
    assert with_settings.quantities_mt == pytest.approx(plain.quantities_mt)
    assert with_settings.ore_cost_total_rs == pytest.approx(plain.ore_cost_total_rs)


def test_lp_does_not_record_priced_correction_terms_by_default() -> None:
    plain, _ = _lp(coke_correction_settings=_correction_settings())

    assert plain is not None
    assert "lp_coke_correction_linear_terms" not in plain.diagnostics


def test_correction_pricing_is_still_available_for_de_seeding() -> None:
    """The mechanism is gated, not deleted - run_nonlinear_optimizer needs it."""

    settings = _correction_settings()
    off, _ = _lp(coke_correction_settings=settings)
    on, errors = _lp(coke_correction_settings=settings, price_coke_correction=True)

    assert errors == []
    assert on is not None and off is not None
    # Pricing slag into the objective pushes the LP toward the richer ore.
    assert on.fe_t_pct > off.fe_t_pct
    assert "lp_coke_correction_linear_terms" in on.diagnostics


# --- Stage 2 must not disturb stage 1 ----------------------------------------


def test_coke_rate_zero_removes_only_its_slag_contribution() -> None:
    """Deliberately zeroing coke in the Fuel Ash table is a slag-only lever."""

    ore = _ore("burden", fe_t=56.0, sio2=5.6, price=7000.0)
    quantities = {"burden": 3900.0}

    with_coke = evaluate_blend(
        ores=[ore], quantities_mt=quantities, feo_in_slag_pct=0.4,
        fuel_cost_per_thm_rs=0.0, fuel_ash_inputs=_fuels(coke_rate=300.0),
        hot_metal_target_mt=2350.0,
    )
    without_coke = evaluate_blend(
        ores=[ore], quantities_mt=quantities, feo_in_slag_pct=0.4,
        fuel_cost_per_thm_rs=0.0, fuel_ash_inputs=_fuels(coke_rate=0.0),
        hot_metal_target_mt=2350.0,
    )

    # The drop is exactly the SLAG-FORMING oxides in the coke ash that is no
    # longer charged - not the whole ash. Ash that is not one of the tracked
    # oxides never reported to slag in the first place.
    coke_ash_mt = 300.0 * 2350.0 / 1000.0 * 0.12
    slag_forming_fraction = (57.0 + 26.0 + 2.7 + 1.0) / 100.0
    assert without_coke.slag_mt < with_coke.slag_mt
    assert with_coke.slag_mt - without_coke.slag_mt == pytest.approx(
        coke_ash_mt * slag_forming_fraction
    )
    # The blend itself is untouched - this is a fuel-ash lever, not an ore lever.
    assert without_coke.quantities_mt == pytest.approx(with_coke.quantities_mt)
    assert without_coke.fe_production_mt == pytest.approx(with_coke.fe_production_mt)


def test_coke_rate_zero_does_not_move_the_back_solved_fuel_rates() -> None:
    """Coke rate is the residual of the cost decomposition, never an input to it.

    So the operator can zero the coke row to strip its ash from the slag without
    disturbing a single number on the fuel side.
    """

    common = dict(
        fuel_cost_per_thm_rs=13_364.0, process_context={}, history_df=None
    )
    normal = estimate_fuel_rates_from_cost(fuel_ash_inputs=_fuels(300.0), **common)
    zeroed = estimate_fuel_rates_from_cost(fuel_ash_inputs=_fuels(0.0), **common)

    assert normal is not None and zeroed is not None
    assert zeroed.coke_rate_kg_thm == pytest.approx(normal.coke_rate_kg_thm)
    assert zeroed.nut_coke_rate_kg_thm == pytest.approx(normal.nut_coke_rate_kg_thm)
    assert zeroed.pci_rate_kg_thm == pytest.approx(normal.pci_rate_kg_thm)
    assert zeroed.total_fuel_rate_kg_thm == pytest.approx(
        normal.total_fuel_rate_kg_thm
    )


def test_nut_coke_and_pci_rates_come_from_the_table() -> None:
    """Both are read straight off the Fuel Ash rows, so slag and the cost
    decomposition can never be looking at different numbers."""

    rates = estimate_fuel_rates_from_cost(
        fuel_cost_per_thm_rs=13_364.0,
        fuel_ash_inputs=_fuels(),
        process_context={},
        history_df=None,
    )

    assert rates is not None
    assert rates.nut_coke_rate_kg_thm == pytest.approx(76.44)
    assert rates.pci_rate_kg_thm == pytest.approx(180.0)
    assert rates.nut_coke_source == "fuel_ash_inputs.nut_coke.rate_kg_per_thm"
    assert rates.pci_source == "fuel_ash_inputs.pci.rate_kg_per_thm"


def test_editing_a_fuel_price_never_moves_the_slag() -> None:
    """Prices belong to stage 2 only."""

    ore = _ore("burden", fe_t=56.0, sio2=5.6, price=7000.0)
    cheap = _fuels()
    dear = [
        FuelAshInput(**{**f.__dict__, "price_rs_per_mt": f.price_rs_per_mt * 2.0})
        for f in _fuels()
    ]

    a = evaluate_blend(
        ores=[ore], quantities_mt={"burden": 3900.0}, feo_in_slag_pct=0.4,
        fuel_cost_per_thm_rs=0.0, fuel_ash_inputs=cheap, hot_metal_target_mt=2350.0,
    )
    b = evaluate_blend(
        ores=[ore], quantities_mt={"burden": 3900.0}, feo_in_slag_pct=0.4,
        fuel_cost_per_thm_rs=0.0, fuel_ash_inputs=dear, hot_metal_target_mt=2350.0,
    )

    assert a.slag_mt == pytest.approx(b.slag_mt)
    assert a.slag_basicity == pytest.approx(b.slag_basicity)
    assert a.slag_al2o3_pct == pytest.approx(b.slag_al2o3_pct)


# --- The displayed blend must BE the LP's blend -------------------------------


class _Prediction:
    value = 13_364.0
    details: dict = {}
    used_fallback = False
    model_loaded = True
    scaler_loaded = True
    missing_features: list = []
    imputed_features: list = []


class _ModelService:
    def predict(self, feature_payload, history_df):
        return _Prediction()


def test_displayed_basicity_equals_the_basicity_the_lp_solved():
    """The bug this guards: LP lands on the bound, display reports a violation.

    The LP drives basicity exactly onto its ceiling. If the display path
    re-evaluates slag on any other fuel basis, the exact value moves off the
    bound and the page reports a constraint violation for a solution that never
    violated one. Observed live as "1.151 > 1.150" and "1.376 > 1.364".
    """

    from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction

    ores = [
        _ore("lean", fe_t=52.0, sio2=9.0, price=900.0),
        _ore("rich", fe_t=64.0, sio2=3.0, price=1600.0),
    ]
    fuels = _fuels()
    b2_max = 0.60

    lp, errors = run_lp_baseline(
        ores,
        target_production_mt=60.0,
        target_slag_qty_mt=1.0e6,
        feo_in_slag_pct=0.0,
        target_slag_basicity_max=b2_max,
        fuel_ash_inputs=fuels,
        hot_metal_target_mt=_HM_MT,
    )
    assert errors == [] and lp is not None and lp.feasible

    shown = evaluate_blend_with_fuel_prediction(
        ores=ores,
        quantities_mt=lp.quantities_mt,
        feo_in_slag_pct=0.0,
        model_service=_ModelService(),
        process_context={},
        history_df=None,
        fuel_ash_inputs=fuels,
        hot_metal_target_mt=_HM_MT,
    )

    assert shown.slag_basicity == pytest.approx(lp.slag_basicity)
    assert shown.slag_t_basicity == pytest.approx(lp.slag_t_basicity)
    assert shown.slag_mt == pytest.approx(lp.slag_mt)
    assert shown.slag_al2o3_pct == pytest.approx(lp.slag_al2o3_pct)
    # And therefore re-validating the displayed blend finds nothing wrong.
    from utils.bmo.constraints import check_blend_constraints

    assert check_blend_constraints(
        shown, ores,
        target_production_mt=60.0,
        target_slag_qty_mt=1.0e6,
        target_slag_basicity_max=b2_max,
    ) == []
