"""Sweep harness for the coke-rate correction across LP and DE runs.

The operator requirement is blunt: the corrected coke rate must not "look
stupidly high" on any run. So rather than spot-checking a few blends, this
sweeps a grid of slag coefficients and slag caps through both solvers and
asserts global properties — monotone response, plausible bands, no NaN, and LP
and DE staying coherent with each other.

The ore universe is priced so that *without* the correction the LP strictly
prefers the cheap lean high-gangue ore. ``_FakeModelService`` returns a constant
cost, which is the honest simulation of production: the real fuel model moves
0.012% when ore per THM doubles, so the correction supplies the entire blend
sensitivity in both.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from utils.bmo.coke_correction import (
    CokeCorrectionReference,
    load_coke_correction_settings,
)
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.nonlinear_optimizer import run_nonlinear_optimizer
from utils.bmo.types import FluxInput, OreChemistry, OreInput

_HM_MT = 2350.0
_TARGET_FE_MT = 2350.0 * 0.945
_CTX = {"PCI_KG/THM": 180.0, "NUT COKE RATE KG/THM": 70.0}

_COKE_BAND = (280.0, 420.0)
_FUEL_BAND = (480.0, 680.0)
_MAX_CORRECTION = 60.0


class _FakePrediction:
    def __init__(self, value: float) -> None:
        self.value = value
        self.details: dict = {}
        self.used_fallback = False


class _FakeModelService:
    """Blend-invariant fuel cost, as the deployed model effectively is.

    12,900 Rs/THM decomposes at baseline prices (coke 28, nut 24, PCI 18) with
    nut coke 70 and PCI 180 pinned to about 300 kg/THM of coke — the plant's
    actual operating point, so the anchor is realistic.
    """

    def __init__(self, value: float = 12_900.0) -> None:
        self.value = value

    def predict(self, feature_payload, history_df):  # noqa: ANN001
        return _FakePrediction(self.value)

    def build_prebuilt_context(self, **kwargs):  # noqa: ANN003
        return None

    def predict_with_prebuilt(self, context, quantities_mt, history_df):  # noqa: ANN001
        return _FakePrediction(self.value)


def _ore(
    ore_id: str,
    *,
    fe_t: float,
    sio2: float,
    al2o3: float,
    cao: float,
    price: float,
    feo: float = 0.0,
) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=ore_id.upper(),
        stock_mt=6000.0,
        price_rs_per_mt=price,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(
            fe_t_pct=fe_t,
            moisture_pct=3.0,
            feo_pct=feo,
            sio2_pct=sio2,
            al2o3_pct=al2o3,
            cao_pct=cao,
        ),
    )


def _universe() -> list[OreInput]:
    """Four burden materials spanning lean-and-cheap to rich-and-dear.

    Prices are calibrated so the uncorrected LP strictly prefers ``lean_ibrm``
    on cost per MT of Fe (8,921 Rs) while a 30 kg/100 kg slag correction makes
    ``pellet`` the cheapest (10,242 vs 11,102 Rs). That is the regime where the
    correction is decisive, which is what these tests are for. A much wider
    price spread would leave lean ore cheapest even after the correction —
    correctly so, and not a useful test of the mechanism.
    """

    return [
        _ore("lean_ibrm", fe_t=52.0, sio2=9.0, al2o3=3.5, cao=1.0, price=4500.0),
        _ore("sinter", fe_t=55.5, sio2=5.2, al2o3=2.0, cao=10.4, price=5000.0, feo=8.0),
        _ore("pellet", fe_t=64.0, sio2=3.0, al2o3=0.8, cao=0.6, price=6000.0, feo=1.0),
        _ore("rich_clo", fe_t=62.0, sio2=4.0, al2o3=1.6, cao=0.5, price=5900.0),
    ]


_RICH_IDS = ("pellet", "rich_clo")


def _limestone() -> FluxInput:
    return FluxInput(
        flux_id="limestone",
        display_name="Limestone",
        enabled=True,
        optimizable=True,
        price_rs_per_mt=1800.0,
        stock_mt=400.0,
        sio2_pct=1.5,
        cao_pct=52.0,
        mgo_pct=1.5,
        loi_pct=40.8,
    )


def _settings(
    *,
    k_slag: float,
    apply_to_objective: bool = True,
    flux_calcination: bool = False,
    max_abs: float = _MAX_CORRECTION,
):
    return load_coke_correction_settings(
        {
            "coke_rate_correction": {
                "enabled": True,
                "apply_to_objective": apply_to_objective,
                "guardrails": {
                    "max_abs_correction_kg_thm": max_abs,
                    "taper_start_fraction": 0.6,
                    "coke_rate_band_kg_thm": list(_COKE_BAND),
                    "total_fuel_rate_band_kg_thm": list(_FUEL_BAND),
                },
                "terms": {
                    "slag_heat": {
                        "enabled": k_slag > 0.0,
                        "kg_coke_per_100kg_slag": k_slag,
                        "max_abs_kg_thm": 45.0,
                        "envelope_halfwidth_kg_per_thm": 60.0,
                        "reference_source": "observed_dpr",
                        "reference_fixed_kg_per_thm": 320.0,
                    },
                    "flux_calcination": {
                        "enabled": flux_calcination,
                        "kg_coke_per_100kg_co2": 30.0,
                        "max_abs_kg_thm": 25.0,
                        "envelope_halfwidth_kg_per_thm": 15.0,
                        "reference_source": "model_current",
                    },
                },
            }
        }
    )


def _reference() -> CokeCorrectionReference:
    return CokeCorrectionReference(
        slag_rate_kg_per_thm=320.0,
        flux_co2_kg_per_thm=0.0,
        sources={"slag_heat": "observed_dpr.test"},
    )


def _run_lp(*, k_slag: float, slag_cap: float, ores=None, **kwargs):
    settings = _settings(k_slag=k_slag, **kwargs) if k_slag or kwargs else None
    return run_lp_baseline(
        ores if ores is not None else _universe(),
        target_production_mt=_TARGET_FE_MT,
        target_slag_qty_mt=slag_cap,
        feo_in_slag_pct=0.4,
        flux_inputs=[_limestone()],
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
        coke_correction_reference=_reference(),
    )


def _lp_with_fuel(*, k_slag: float, slag_cap: float, reference=None, **kwargs):
    """Run the LP, then attach fuel cost the way the Blend Optimizer page does.

    ``run_lp_baseline`` never predicts fuel — it prices the correction linearly
    in its cost vector and stops there. The nonlinear, clamped correction the
    operator actually reads comes from the page's post-hoc
    ``evaluate_blend_with_fuel_prediction`` call, so the band assertions have to
    go through that same step or they check nothing at all.
    """

    ores = _universe()
    settings = _settings(k_slag=k_slag, **kwargs) if k_slag or kwargs else None
    physical, errors = run_lp_baseline(
        ores,
        target_production_mt=_TARGET_FE_MT,
        target_slag_qty_mt=slag_cap,
        feo_in_slag_pct=0.4,
        flux_inputs=[_limestone()],
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
        coke_correction_reference=reference if reference is not None else _reference(),
    )
    if physical is None:
        return None, errors

    solved_flux = physical.diagnostics.get("lp_flux_quantities_mt", {}) or {}
    flux_for_display = [
        replace(flux, wet_qty_mt=float(solved_flux.get(flux.flux_id, flux.wet_qty_mt)))
        for flux in [_limestone()]
    ]
    blend = evaluate_blend_with_fuel_prediction(
        ores=ores,
        quantities_mt=physical.quantities_mt,
        feo_in_slag_pct=0.4,
        model_service=_FakeModelService(),
        process_context=_CTX,
        history_df=None,
        flux_inputs=flux_for_display,
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
        coke_correction_reference=reference if reference is not None else _reference(),
    )
    blend.diagnostics["lp_coke_correction_linear_terms"] = physical.diagnostics.get(
        "lp_coke_correction_linear_terms"
    )
    return blend, errors


def _run_de(*, k_slag: float, slag_cap: float, **kwargs):
    settings = _settings(k_slag=k_slag, **kwargs) if k_slag or kwargs else None
    return run_nonlinear_optimizer(
        _universe(),
        target_production_mt=_TARGET_FE_MT,
        target_slag_qty_mt=slag_cap,
        feo_in_slag_pct=0.4,
        model_service=_FakeModelService(),
        process_context=_CTX,
        history_df=None,
        # Trimmed for suite runtime; the grid is what gives the coverage here.
        de_cfg={"maxiter": 6, "popsize": 5, "seed": 7, "tol": 0.05},
        flux_inputs=[_limestone()],
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
        coke_correction_reference=_reference(),
    )


def _rich_share_pct(blend) -> float:
    return sum(float(blend.shares_pct.get(ore_id, 0.0)) for ore_id in _RICH_IDS)


def _assert_plausible(blend, label: str) -> None:
    """The operator's core requirement, asserted on every grid cell."""

    assert math.isfinite(blend.objective_rs_per_thm), f"{label}: non-finite objective"

    delta = blend.diagnostics.get("coke_correction_delta_kg_thm")
    assert delta is not None, (
        f"{label}: no correction diagnostics — the blend never went through the "
        "fuel-prediction path, so nothing was actually checked"
    )
    delta = float(delta)
    assert math.isfinite(delta), f"{label}: non-finite correction"
    assert abs(delta) <= _MAX_CORRECTION + 1e-6, f"{label}: correction {delta:.1f} kg/THM"

    rates = blend.diagnostics.get("fuel_rate_estimate") or {}
    coke = float(rates.get("coke_rate_kg_thm", 0.0))
    fuel = float(rates.get("total_fuel_rate_kg_thm", 0.0))
    assert _COKE_BAND[0] - 1e-6 <= coke <= _COKE_BAND[1] + 1e-6, (
        f"{label}: coke rate {coke:.1f} kg/THM outside {_COKE_BAND}"
    )
    assert _FUEL_BAND[0] - 1e-6 <= fuel <= _FUEL_BAND[1] + 1e-6, (
        f"{label}: total fuel rate {fuel:.1f} kg/THM outside {_FUEL_BAND}"
    )


_SLAG_CAPS = (600.0, 700.0, 750.0, 800.0)
_K_SLAG = (0.0, 10.0, 20.0, 30.0, 35.0)


# --------------------------------------------------------------------------
# LP sweep
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slag_cap", _SLAG_CAPS)
def test_lp_slag_rate_is_non_increasing_in_the_slag_coefficient(slag_cap):
    slag_rates = []
    rich_shares = []
    for k in _K_SLAG:
        blend, errors = _lp_with_fuel(k_slag=k, slag_cap=slag_cap)
        assert errors == [] and blend is not None, f"LP infeasible at k={k}"
        if k > 0.0:
            _assert_plausible(blend, f"LP k={k} cap={slag_cap}")
        slag_rates.append(blend.slag_rate_kg_per_thm)
        rich_shares.append(_rich_share_pct(blend))

    for previous, current in zip(slag_rates, slag_rates[1:]):
        assert current <= previous + 1e-3, f"slag rate rose with k: {slag_rates}"
    for previous, current in zip(rich_shares, rich_shares[1:]):
        assert current >= previous - 1e-3, f"rich share fell with k: {rich_shares}"


@pytest.mark.parametrize("slag_cap", _SLAG_CAPS)
def test_lp_correction_raises_fe_at_every_slag_cap(slag_cap):
    """The one assertion that catches a sign error end to end."""

    without, _ = _run_lp(k_slag=0.0, slag_cap=slag_cap)
    with_correction, _ = _run_lp(k_slag=30.0, slag_cap=slag_cap)

    assert with_correction.fe_t_pct > without.fe_t_pct, (
        f"cap={slag_cap}: correction did not enrich the burden "
        f"({without.fe_t_pct:.2f} -> {with_correction.fe_t_pct:.2f} %Fe)"
    )
    assert with_correction.slag_rate_kg_per_thm < without.slag_rate_kg_per_thm


def test_lp_tightening_the_slag_cap_never_raises_the_achieved_slag_rate():
    rates = []
    for cap in sorted(_SLAG_CAPS, reverse=True):
        blend, errors = _run_lp(k_slag=30.0, slag_cap=cap)
        assert errors == [] and blend is not None
        rates.append(blend.slag_rate_kg_per_thm)

    for previous, current in zip(rates, rates[1:]):
        assert current <= previous + 1e-3, f"slag rate rose as the cap tightened: {rates}"


def test_lp_stays_plausible_with_flux_calcination_priced_in():
    """Limestone gets ~3,400 Rs/MT dearer; the result must still be sane."""

    blend, errors = _lp_with_fuel(k_slag=30.0, slag_cap=750.0, flux_calcination=True)

    assert errors == [] and blend is not None
    _assert_plausible(blend, "LP with flux calcination")
    coefficients = blend.diagnostics["lp_coke_correction_linear_terms"][
        "coefficients_rs_per_wet_mt"
    ]
    assert coefficients["limestone"] > 3_000.0


# --------------------------------------------------------------------------
# DE sweep
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slag_cap", (700.0, 800.0))
def test_de_stays_plausible_and_never_beats_lp(slag_cap):
    for k in (20.0, 30.0):
        de_blend, de_errors = _run_de(k_slag=k, slag_cap=slag_cap)
        # Compared against the LP *after* its post-hoc fuel evaluation: the raw
        # LP objective carries no fuel term at all, so comparing to it directly
        # would make DE look worse on every run for the wrong reason.
        lp_blend, lp_errors = _lp_with_fuel(k_slag=k, slag_cap=slag_cap)

        assert de_errors == [] and de_blend is not None, f"DE failed at k={k}"
        assert lp_errors == [] and lp_blend is not None
        _assert_plausible(de_blend, f"DE k={k} cap={slag_cap}")

        # The page replaces DE with LP whenever DE scores worse, so DE must not
        # be systematically worse or the correction has broken its seeding.
        assert de_blend.objective_rs_per_thm <= lp_blend.objective_rs_per_thm + 1.0


@pytest.mark.parametrize("slag_cap", (700.0, 800.0))
def test_de_correction_enriches_the_burden(slag_cap):
    without, _ = _run_de(k_slag=0.0, slag_cap=slag_cap)
    with_correction, _ = _run_de(k_slag=30.0, slag_cap=slag_cap)

    assert without is not None and with_correction is not None
    assert with_correction.fe_t_pct > without.fe_t_pct


def test_de_reports_both_uncorrected_and_corrected_coke_rates():
    blend, errors = _run_de(k_slag=30.0, slag_cap=800.0)

    assert errors == [] and blend is not None
    anchor = blend.diagnostics["fuel_rate_estimate_anchor"]["coke_rate_kg_thm"]
    corrected = blend.diagnostics["fuel_rate_estimate"]["coke_rate_kg_thm"]
    delta = blend.diagnostics["coke_correction_delta_kg_thm"]

    assert corrected == pytest.approx(anchor + delta)
    assert _COKE_BAND[0] <= anchor <= _COKE_BAND[1]


# --------------------------------------------------------------------------
# Non-regression and guardrail reachability
# --------------------------------------------------------------------------


def test_reference_taken_from_the_blend_itself_gives_a_zero_correction():
    """Anchor to the blend's own achieved slag rate and the delta must vanish.

    This is the non-regression guarantee at the level the operator sees it: a
    recommendation that reproduces current conditions must report the same coke
    rate the model gave before this feature existed.
    """

    baseline, _ = _lp_with_fuel(k_slag=30.0, slag_cap=750.0)
    achieved_slag_rate = float(baseline.slag_rate_kg_per_thm)

    blend, errors = _lp_with_fuel(
        k_slag=30.0,
        slag_cap=750.0,
        reference=CokeCorrectionReference(
            slag_rate_kg_per_thm=achieved_slag_rate, flux_co2_kg_per_thm=0.0
        ),
    )

    assert errors == [] and blend is not None
    # The LP's linear term is reference-independent (a constant does not change
    # argmin), so the same blend is found and its reported delta collapses.
    assert blend.quantities_mt == pytest.approx(baseline.quantities_mt)
    assert blend.diagnostics["coke_correction_delta_kg_thm"] == pytest.approx(
        0.0, abs=1e-6
    )
    anchor = blend.diagnostics["fuel_rate_estimate_anchor"]
    corrected = blend.diagnostics["fuel_rate_estimate"]
    assert corrected["coke_rate_kg_thm"] == pytest.approx(anchor["coke_rate_kg_thm"])


def test_display_only_mode_leaves_the_lp_blend_unchanged():
    priced_in, _ = _run_lp(k_slag=30.0, slag_cap=750.0)
    display_only, _ = _run_lp(
        k_slag=30.0, slag_cap=750.0, apply_to_objective=False
    )
    uncorrected, _ = _run_lp(k_slag=0.0, slag_cap=750.0)

    assert display_only.quantities_mt == pytest.approx(uncorrected.quantities_mt)
    assert display_only.quantities_mt != pytest.approx(priced_in.quantities_mt)


def test_an_extreme_cell_makes_a_guardrail_fire_visibly():
    """Push far past the envelope and confirm the cap reports itself."""

    blend, errors = _lp_with_fuel(
        k_slag=35.0,
        slag_cap=1200.0,
        max_abs=5.0,
        reference=CokeCorrectionReference(
            slag_rate_kg_per_thm=50.0, flux_co2_kg_per_thm=0.0
        ),
    )

    assert errors == [] and blend is not None
    correction = blend.diagnostics["coke_correction"]
    assert correction["total_clamp_binding"] is True
    assert correction["warnings"], "a binding cap must be reported, not silent"
    assert abs(correction["applied_delta_kg_thm"]) <= 5.0 + 1e-6


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
