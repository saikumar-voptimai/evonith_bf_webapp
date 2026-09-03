"""LP/DE consistency tests for the physics coke-rate correction.

The correction enters both solvers: the LP as a linear addition to its cost
vector, DE through the per-candidate blend evaluation. These tests prove the two
paths price the same physics, that the DE objective is a single consistent
function across every iteration, and that the correction reaches the seed LP DE
starts from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.bmo import lp_solver
from utils.bmo.coke_correction import (
    CokeCorrectionReference,
    load_coke_correction_settings,
)
from utils.bmo.fuel_rates import ASSUMED_FUEL_PRICES_RS_PER_KG
from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.types import FluxInput, OreChemistry, OreInput

_HM_MT = 100.0

# PCI and nut coke are pinned operator run inputs at this plant (PCI ~180,
# nut coke 70 kg/THM), so only coke is back-solved from the predicted cost. The
# decomposition returns nothing at all without them, which would leave no coke
# rate for the correction to act on.
_CTX = {"PCI_KG/THM": 180.0, "NUT COKE RATE KG/THM": 70.0}


class _FakePrediction:
    def __init__(self, value: float) -> None:
        self.value = value
        self.details: dict = {}
        self.used_fallback = False


class _FakeModelService:
    """Constant fuel cost — the honest stand-in for the blend-blind real model."""

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
    fe_t: float = 62.0,
    sio2: float = 5.0,
    cao: float = 1.0,
    feo: float = 0.0,
    price: float = 1000.0,
) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=ore_id.upper(),
        stock_mt=5000.0,
        price_rs_per_mt=price,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(
            fe_t_pct=fe_t,
            moisture_pct=3.0,
            sio2_pct=sio2,
            cao_pct=cao,
            feo_pct=feo,
        ),
    )


def _limestone() -> FluxInput:
    return FluxInput(
        flux_id="limestone",
        display_name="Limestone",
        enabled=True,
        optimizable=True,
        price_rs_per_mt=1800.0,
        stock_mt=500.0,
        sio2_pct=1.5,
        cao_pct=52.0,
        loi_pct=40.8,
    )


def _settings(*, apply_to_objective: bool = True, k_slag: float = 30.0):
    return load_coke_correction_settings(
        {
            "coke_rate_correction": {
                "enabled": True,
                "apply_to_objective": apply_to_objective,
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
                    },
                    "flux_calcination": {
                        "enabled": True,
                        "kg_coke_per_100kg_co2": 30.0,
                        "max_abs_kg_thm": 25.0,
                        "reference_source": "model_current",
                    },
                },
            }
        }
    )


def _reference(slag: float = 300.0, flux_co2: float = 0.0) -> CokeCorrectionReference:
    return CokeCorrectionReference(
        slag_rate_kg_per_thm=slag, flux_co2_kg_per_thm=flux_co2
    )


def _evaluator(*, settings=None, reference=None) -> BmoObjectiveEvaluator:
    return BmoObjectiveEvaluator(
        ores=[_ore("lean", fe_t=52.0, sio2=9.0), _ore("rich", fe_t=64.0, sio2=3.0)],
        target_production_mt=60.0,
        target_slag_qty_mt=5000.0,
        feo_in_slag_pct=0.0,
        model_service=_FakeModelService(),
        process_context=_CTX,
        history_df=None,
        penalty_cfg={},
        flux_inputs=[_limestone()],
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
        coke_correction_reference=reference,
    )


# --------------------------------------------------------------------------
# Objective consistency — the optimizer-stability guarantee
# --------------------------------------------------------------------------


def test_de_objective_is_deterministic_across_repeated_evaluations():
    """The same candidate must always score the same.

    The reference is resolved once and frozen, and every driver is a pure
    function of the candidate's own quantities, so nothing about the correction
    can drift between iterations of the search.
    """

    evaluator = _evaluator(settings=_settings(), reference=_reference())
    qty = np.array([80.0, 40.0])
    flux = np.array([25.0])

    values = [
        evaluator.evaluate_quantities(qty, flux_quantities=flux).objective_value
        for _ in range(8)
    ]

    assert len(set(values)) == 1


def test_de_objective_is_continuous_in_the_decision_variables():
    """No jumps: a smooth sweep must not produce a discontinuous objective.

    A correction that flickered on and off with data availability would show up
    here as a step change, which is exactly what would derail the search.
    """

    evaluator = _evaluator(settings=_settings(), reference=_reference())
    values = [
        evaluator.evaluate_quantities(
            np.array([float(lean), 40.0]), flux_quantities=np.array([25.0])
        ).objective_value
        for lean in np.linspace(20.0, 140.0, 60)
    ]

    steps = np.abs(np.diff(values))
    assert np.all(np.isfinite(values))
    # Neighbouring candidates differ by ~2 MT of ore; nothing should leap.
    assert steps.max() < 10.0 * np.median(steps) + 1.0


def test_missing_reference_disables_the_term_for_the_whole_run():
    """An unresolvable reference must disable its term, not intermittently fire."""

    evaluator = _evaluator(
        settings=_settings(),
        reference=CokeCorrectionReference(slag_rate_kg_per_thm=None),
    )

    for lean in (40.0, 90.0, 150.0):
        result = evaluator.evaluate_quantities(
            np.array([lean, 40.0]), flux_quantities=np.array([10.0])
        )
        blend = result.diagnostics["blend"]
        slag_term = next(
            t
            for t in blend.diagnostics["coke_correction"]["terms"]
            if t["term_id"] == "slag_heat"
        )
        assert slag_term["enabled"] is False
        assert slag_term["disabled_reason"] == "no current-burden reference available"


def test_de_records_both_uncorrected_and_corrected_rates():
    evaluator = _evaluator(settings=_settings(), reference=_reference())
    result = evaluator.evaluate_quantities(
        np.array([120.0, 20.0]), flux_quantities=np.array([0.0])
    )
    blend = result.diagnostics["blend"]

    anchor = blend.diagnostics["fuel_rate_estimate_anchor"]
    corrected = blend.diagnostics["fuel_rate_estimate"]

    assert anchor["coke_rate_kg_thm"] != corrected["coke_rate_kg_thm"]
    assert corrected["coke_rate_kg_thm"] == pytest.approx(
        anchor["coke_rate_kg_thm"] + blend.diagnostics["coke_correction_delta_kg_thm"]
    )
    # Nut coke and PCI are operator run inputs and must survive untouched.
    assert anchor["nut_coke_rate_kg_thm"] == corrected["nut_coke_rate_kg_thm"]
    assert anchor["pci_rate_kg_thm"] == corrected["pci_rate_kg_thm"]
    assert result.components["coke_correction_delta_kg_thm"] != 0.0


def test_display_only_mode_leaves_the_objective_untouched():
    reference = _reference()
    qty = np.array([120.0, 20.0])
    flux = np.array([10.0])

    off = _evaluator().evaluate_quantities(qty, flux_quantities=flux)
    display_only = _evaluator(
        settings=_settings(apply_to_objective=False), reference=reference
    ).evaluate_quantities(qty, flux_quantities=flux)

    assert display_only.objective_value == pytest.approx(off.objective_value)
    blend = display_only.diagnostics["blend"]
    assert blend.diagnostics["coke_correction_applied"] is False
    # Still computed and reported, so the operator can see it before enabling.
    assert blend.diagnostics["coke_correction"]["enabled"] is True


def test_a_leaner_burden_costs_more_once_the_correction_is_on():
    """The whole point: more slag must cost more fuel."""

    settings = _settings()
    reference = _reference()
    flux = np.array([0.0])
    lean_heavy = np.array([140.0, 10.0])
    rich_heavy = np.array([10.0, 140.0])

    evaluator = _evaluator(settings=settings, reference=reference)
    lean = evaluator.evaluate_quantities(lean_heavy, flux_quantities=flux)
    rich = evaluator.evaluate_quantities(rich_heavy, flux_quantities=flux)

    lean_blend = lean.diagnostics["blend"]
    rich_blend = rich.diagnostics["blend"]

    assert lean_blend.slag_rate_kg_per_thm > rich_blend.slag_rate_kg_per_thm
    assert (
        lean_blend.diagnostics["coke_correction_delta_kg_thm"]
        > rich_blend.diagnostics["coke_correction_delta_kg_thm"]
    )


# --------------------------------------------------------------------------
# LP linear term
# --------------------------------------------------------------------------


def _lp(*, settings=None, slag_cap: float = 400.0):
    return run_lp_baseline(
        [
            _ore("lean", fe_t=52.0, sio2=9.0, price=900.0),
            _ore("rich", fe_t=64.0, sio2=3.0, price=1600.0),
        ],
        target_production_mt=60.0,
        target_slag_qty_mt=slag_cap,
        feo_in_slag_pct=0.0,
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
        # The LP now defaults to a pure ore-cost objective; these
        # tests exercise the correction-pricing path DE still uses.
        price_coke_correction=settings is not None,
    )


def test_lp_without_the_correction_prefers_the_cheap_lean_ore():
    blend, errors = _lp()

    assert errors == []
    assert blend is not None
    assert blend.shares_pct["lean"] > blend.shares_pct["rich"]


def test_lp_with_the_correction_shifts_toward_the_richer_ore():
    """The single assertion that would catch a sign error in the LP term."""

    without, _ = _lp()
    with_correction, errors = _lp(settings=_settings())

    assert errors == []
    assert with_correction is not None
    assert with_correction.fe_t_pct > without.fe_t_pct
    assert with_correction.slag_rate_kg_per_thm < without.slag_rate_kg_per_thm


def test_lp_shift_scales_with_the_slag_coefficient():
    fe_by_k = [
        _lp(settings=_settings(k_slag=k))[0].fe_t_pct for k in (0.0, 10.0, 20.0, 30.0)
    ]

    assert fe_by_k == sorted(fe_by_k)


def test_lp_records_the_linear_terms_it_priced():
    blend, _ = _lp(settings=_settings())

    terms = blend.diagnostics["lp_coke_correction_linear_terms"]
    coefficients = terms["coefficients_rs_per_wet_mt"]
    # The lean, high-gangue ore must carry the larger implied fuel cost.
    assert coefficients["lean"] > coefficients["rich"] > 0.0
    assert terms["cost_rs"] > 0.0


def test_lp_reuses_the_shared_slag_coefficients(monkeypatch):
    """Guards against the LP slag constraint and LP slag pricing drifting apart."""

    seen: dict[str, np.ndarray] = {}
    original = lp_solver._build_linear_slag_and_basicity_terms

    def _spy(*args, **kwargs):
        result = original(*args, **kwargs)
        seen["slag_coeff"] = np.array(result.slag_coeff, dtype=float)
        return result

    monkeypatch.setattr(lp_solver, "_build_linear_slag_and_basicity_terms", _spy)

    blend, _ = _lp(settings=_settings())
    coefficients = blend.diagnostics["lp_coke_correction_linear_terms"][
        "coefficients_rs_per_wet_mt"
    ]

    price = ASSUMED_FUEL_PRICES_RS_PER_KG["coke"]
    expected = 1000.0 * 0.30 * price * seen["slag_coeff"]
    assert coefficients["lean"] == pytest.approx(expected[0])
    assert coefficients["rich"] == pytest.approx(expected[1])


def test_lp_correction_only_changes_the_cost_vector_not_feasibility():
    """Only ``c`` moves, so a blend feasible without the correction stays feasible."""

    without, errors_without = _lp(slag_cap=400.0)
    with_correction, errors_with = _lp(settings=_settings(), slag_cap=400.0)

    assert errors_without == [] and errors_with == []
    assert without.feasible and with_correction.feasible
    assert with_correction.slag_mt <= 400.0 + 1e-6


def test_lp_infeasibility_explanation_receives_the_correction(monkeypatch):
    calls: list[dict] = []
    original = lp_solver.run_lp_baseline

    def _spy(ores, **kwargs):
        calls.append(kwargs)
        return original(ores, **kwargs)

    monkeypatch.setattr(lp_solver, "run_lp_baseline", _spy)

    settings = _settings()
    # A slag cap far below what any feasible burden can make sends the explain
    # path past the structural checks and into its re-solves.
    blend, errors = original(
        [_ore("lean", fe_t=52.0, sio2=9.0), _ore("rich", fe_t=64.0, sio2=3.0)],
        target_production_mt=60.0,
        target_slag_qty_mt=0.5,
        feo_in_slag_pct=0.0,
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
    )

    assert blend is None and errors
    # Every re-solve inside the explanation must carry the same objective.
    assert calls, "the explanation should have re-solved at least once"
    assert all(call.get("coke_correction_settings") is settings for call in calls)


# --------------------------------------------------------------------------
# Fuel-rate anchor basis
# --------------------------------------------------------------------------


def _fuel_ash(coke_rate: float = 295.0):
    from utils.bmo.types import FuelAshInput

    return [
        FuelAshInput(
            fuel_id="coke",
            display_name="Coke",
            rate_kg_per_thm=coke_rate,
            price_rs_per_mt=28_000.0,
            ash_pct=11.0,
            sio2_pct=55.0,
        ),
        FuelAshInput(
            fuel_id="nut_coke",
            display_name="Nut Coke",
            rate_kg_per_thm=70.0,
            price_rs_per_mt=24_000.0,
            ash_pct=11.0,
            sio2_pct=55.0,
        ),
        FuelAshInput(
            fuel_id="pci",
            display_name="PCI",
            rate_kg_per_thm=180.0,
            price_rs_per_mt=18_000.0,
            ash_pct=9.0,
            sio2_pct=49.0,
        ),
    ]


def _blend_with_anchor(
    basis: str, *, coke_rate: float = 295.0, settings=None, reference=None
):
    from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction

    return evaluate_blend_with_fuel_prediction(
        ores=[_ore("lean", fe_t=52.0, sio2=9.0), _ore("rich", fe_t=64.0, sio2=3.0)],
        quantities_mt={"lean": 100.0, "rich": 100.0},
        feo_in_slag_pct=0.0,
        model_service=_FakeModelService(),
        process_context=_CTX,
        history_df=None,
        fuel_ash_inputs=_fuel_ash(coke_rate),
        hot_metal_target_mt=_HM_MT,
        fuel_rate_anchor_basis=basis,
        coke_correction_settings=settings,
        coke_correction_reference=reference,
    )


def test_observed_anchor_reports_the_plant_coke_rate_not_the_model_residual():
    """The model's cost is a near-constant, so its residual is not the plant."""

    model = _blend_with_anchor("model_cost")
    observed = _blend_with_anchor("observed", coke_rate=295.0)

    assert observed.diagnostics["fuel_rate_estimate"][
        "coke_rate_kg_thm"
    ] == pytest.approx(295.0)
    assert (
        observed.diagnostics["fuel_rate_estimate_source"] == "observed_fuel_ash_inputs"
    )
    # 12,900 Rs/THM back-solves to something quite different from 295.
    assert model.diagnostics["fuel_rate_estimate"]["coke_rate_kg_thm"] != pytest.approx(
        295.0
    )
    assert model.diagnostics["fuel_rate_estimate_source"] == "model_cost_residual"


def test_observed_anchor_rebases_fuel_cost_to_match_the_rates_it_reports():
    blend = _blend_with_anchor("observed", coke_rate=295.0)

    expected = (
        295.0 * ASSUMED_FUEL_PRICES_RS_PER_KG["coke"]
        + 70.0 * ASSUMED_FUEL_PRICES_RS_PER_KG["nut_coke"]
        + 180.0 * ASSUMED_FUEL_PRICES_RS_PER_KG["pci"]
    )
    assert blend.fuel_cost_per_thm_rs == pytest.approx(expected)
    assert blend.diagnostics["fuel_cost_per_thm_rs_model"] == pytest.approx(12_900.0)


def test_rebasing_leaves_ore_cost_intact():
    """Ore cost is derived as (objective - fuel), so both must move together."""

    model = _blend_with_anchor("model_cost")
    observed = _blend_with_anchor("observed")

    assert observed.ore_cost_per_thm_rs == pytest.approx(model.ore_cost_per_thm_rs)
    derived = observed.objective_rs_per_thm - observed.fuel_cost_per_thm_rs
    assert derived == pytest.approx(observed.ore_cost_per_thm_rs)


def test_observed_anchor_falls_back_when_the_editor_rows_are_incomplete():
    from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction

    blend = evaluate_blend_with_fuel_prediction(
        ores=[_ore("lean", fe_t=52.0, sio2=9.0)],
        quantities_mt={"lean": 200.0},
        feo_in_slag_pct=0.0,
        model_service=_FakeModelService(),
        process_context=_CTX,
        history_df=None,
        fuel_ash_inputs=None,  # no editor rows at all
        hot_metal_target_mt=_HM_MT,
        fuel_rate_anchor_basis="observed",
    )

    assert blend.diagnostics["fuel_rate_estimate_source"] == "model_cost_residual"
    # No rebase happened, so the model cost is untouched.
    assert "fuel_cost_per_thm_rs_model" not in blend.diagnostics
    assert blend.fuel_cost_per_thm_rs == pytest.approx(12_900.0)


def test_default_anchor_basis_is_unchanged_behaviour():
    explicit = _blend_with_anchor("model_cost")
    from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction

    default = evaluate_blend_with_fuel_prediction(
        ores=[_ore("lean", fe_t=52.0, sio2=9.0), _ore("rich", fe_t=64.0, sio2=3.0)],
        quantities_mt={"lean": 100.0, "rich": 100.0},
        feo_in_slag_pct=0.0,
        model_service=_FakeModelService(),
        process_context=_CTX,
        history_df=None,
        fuel_ash_inputs=_fuel_ash(),
        hot_metal_target_mt=_HM_MT,
    )

    assert default.fuel_cost_per_thm_rs == pytest.approx(explicit.fuel_cost_per_thm_rs)
    assert (
        default.diagnostics["fuel_rate_estimate"]
        == explicit.diagnostics["fuel_rate_estimate"]
    )


def test_correction_adds_on_top_of_the_observed_anchor():
    """The reported total fuel must move by the full correction, from the plant rate."""

    settings = _settings()
    reference = _reference()
    blend = _blend_with_anchor(
        "observed", coke_rate=295.0, settings=settings, reference=reference
    )

    anchor = blend.diagnostics["fuel_rate_estimate_anchor"]
    corrected = blend.diagnostics["fuel_rate_estimate"]
    delta = blend.diagnostics["coke_correction_delta_kg_thm"]

    assert anchor["coke_rate_kg_thm"] == pytest.approx(295.0)
    assert anchor["total_fuel_rate_kg_thm"] == pytest.approx(295.0 + 70.0 + 180.0)
    assert corrected["coke_rate_kg_thm"] == pytest.approx(295.0 + delta)
    # The whole delta lands in the total, because nut coke and PCI do not move.
    assert corrected["total_fuel_rate_kg_thm"] == pytest.approx(
        anchor["total_fuel_rate_kg_thm"] + delta
    )


def test_anchor_choice_does_not_change_which_blend_wins():
    """Final-fuel recalculation may move the gap, but not reverse the ranking."""

    settings = _settings()
    reference = _reference()
    qty_a, qty_b = np.array([140.0, 10.0]), np.array([10.0, 140.0])
    flux = np.array([0.0])

    def gap(basis: str) -> float:
        evaluator = BmoObjectiveEvaluator(
            ores=[_ore("lean", fe_t=52.0, sio2=9.0), _ore("rich", fe_t=64.0, sio2=3.0)],
            target_production_mt=60.0,
            target_slag_qty_mt=5000.0,
            feo_in_slag_pct=0.0,
            model_service=_FakeModelService(),
            process_context=_CTX,
            history_df=None,
            penalty_cfg={},
            fuel_ash_inputs=_fuel_ash(),
            flux_inputs=[_limestone()],
            hot_metal_target_mt=_HM_MT,
            coke_correction_settings=settings,
            coke_correction_reference=reference,
            fuel_rate_anchor_basis=basis,
        )
        a = evaluator.evaluate_quantities(qty_a, flux_quantities=flux).objective_value
        b = evaluator.evaluate_quantities(qty_b, flux_quantities=flux).objective_value
        return a - b

    observed_gap = gap("observed")
    model_gap = gap("model_cost")

    assert observed_gap != pytest.approx(0.0, abs=1e-6)
    assert np.sign(observed_gap) == np.sign(model_gap)


def test_shipped_anchor_basis_keeps_the_coke_rate_a_prediction():
    """The shipped coke rate must be predicted, never the live tag read back.

    A reported coke rate seeded from the live coke tag is a restatement of what
    the furnace was running an hour ago, not a forecast. The physics correction
    is a delta on top of the prediction; it is not itself the prediction.

    Two bases satisfy that and ``observed`` does not:

        model_cost      back-solve coke from the ML model's predicted cost
        energy_balance  solve the closed energy balance at current controls,
                        less the rolling bias offset (currently shipped -
                        MAPE 3.37%, R2 +0.74 forward, against 7.24% / +0.07
                        for the same balance uncorrected)

    Note that ``energy_balance`` FALLS BACK to observed at runtime when the live
    tags or the calibration are missing. That is a degradation the page reports
    on screen, not a configured choice, which is why this asserts on the config
    rather than on any particular blend's ``fuel_rate_estimate_source``.
    See setting_bmo.yml for the full note.
    """

    import yaml

    path = Path(__file__).resolve().parents[1] / "src" / "config" / "setting_bmo.yml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["bmo"]

    assert cfg["fuel_rate_anchor_basis"] in ("model_cost", "energy_balance")
    assert cfg["fuel_rate_anchor_basis"] != "observed"


def test_model_cost_anchor_back_solves_coke_and_ignores_the_editor_coke_rate():
    """Coke must come out of the cost residual even when a coke row is present.

    The Fuel Ash table still pins nut coke and PCI - they are operator run inputs
    - but its coke figure must not become the answer.
    """

    editor_coke = 295.0
    blend = _blend_with_anchor("model_cost", coke_rate=editor_coke)

    rates = blend.diagnostics["fuel_rate_estimate"]
    assert blend.diagnostics["fuel_rate_estimate_source"] == "model_cost_residual"
    assert rates["coke_rate_kg_thm"] != pytest.approx(editor_coke)
    # Nut coke and PCI are taken verbatim from the editor rows.
    assert rates["nut_coke_rate_kg_thm"] == pytest.approx(70.0)
    assert rates["pci_rate_kg_thm"] == pytest.approx(180.0)
    # coke = (cost - nut x p_nut - pci x p_pci) / p_coke
    expected = (
        12_900.0
        - 70.0 * ASSUMED_FUEL_PRICES_RS_PER_KG["nut_coke"]
        - 180.0 * ASSUMED_FUEL_PRICES_RS_PER_KG["pci"]
    ) / ASSUMED_FUEL_PRICES_RS_PER_KG["coke"]
    assert rates["coke_rate_kg_thm"] == pytest.approx(expected)
    # No rebase on this path: the cost and the rates are the same number.
    assert "fuel_cost_per_thm_rs_model" not in blend.diagnostics
    assert blend.fuel_cost_per_thm_rs == pytest.approx(12_900.0)


def test_de_seed_lp_receives_the_correction(monkeypatch):
    from utils.bmo import nonlinear_optimizer

    seen: list[dict] = []
    original = nonlinear_optimizer.run_lp_baseline

    def _spy(ores, **kwargs):
        seen.append(kwargs)
        return original(ores, **kwargs)

    monkeypatch.setattr(nonlinear_optimizer, "run_lp_baseline", _spy)

    settings = _settings()
    reference = _reference()
    nonlinear_optimizer.run_nonlinear_optimizer(
        [_ore("lean", fe_t=52.0, sio2=9.0), _ore("rich", fe_t=64.0, sio2=3.0)],
        target_production_mt=60.0,
        target_slag_qty_mt=400.0,
        feo_in_slag_pct=0.0,
        model_service=_FakeModelService(),
        process_context=_CTX,
        history_df=None,
        de_cfg={"maxiter": 2, "popsize": 4, "seed": 1},
        hot_metal_target_mt=_HM_MT,
        coke_correction_settings=settings,
        coke_correction_reference=reference,
    )

    assert seen, "DE must seed from the LP"
    assert seen[0]["coke_correction_settings"] is settings
    assert seen[0]["coke_correction_reference"] is reference


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
