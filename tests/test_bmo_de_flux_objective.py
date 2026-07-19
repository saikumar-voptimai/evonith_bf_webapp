"""Tests for the flux-aware DE objective (BmoObjectiveEvaluator).

Optimisable fluxes (dolomite/quartz) are DE decision variables, so a candidate's
flux quantities change the evaluated slag basicity and the basicity error penalty.
A stub model service keeps the test free of the fuel model and the database.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.bmo.objective import BmoObjectiveEvaluator
from utils.bmo.types import FluxInput, OreChemistry, OreInput


class _FakePrediction:
    def __init__(self, value: float) -> None:
        self.value = value
        self.details: dict = {}
        self.used_fallback = False


class _FakeModelService:
    """Constant fuel-cost stub so the objective needs no model/DB."""

    def predict(self, feature_payload, history_df):  # noqa: ANN001
        return _FakePrediction(5000.0)


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


def _dolomite() -> FluxInput:
    return FluxInput(
        flux_id="dolomite",
        display_name="Dolomite",
        enabled=True,
        optimizable=True,
        price_rs_per_mt=3000.0,
        stock_mt=500.0,
        sio2_pct=1.7,
        cao_pct=30.2,
        mgo_pct=22.3,
    )


def _evaluator() -> BmoObjectiveEvaluator:
    return BmoObjectiveEvaluator(
        ores=[_ore("ore_a", sio2=8.0, cao=1.0), _ore("ore_b", sio2=7.0, cao=1.5)],
        target_production_mt=100.0,
        target_slag_qty_mt=5000.0,
        feo_in_slag_pct=0.0,
        model_service=_FakeModelService(),
        process_context={},
        history_df=None,
        penalty_cfg={},
        target_slag_basicity_min=1.6,
        target_slag_basicity_max=3.0,
        flux_inputs=[_dolomite()],
        hot_metal_target_mt=100.0,
    )


class TestFluxAwareObjective:
    def test_flux_is_a_variable(self):
        assert _evaluator().n_flux == 1

    def test_dolomite_raises_basicity_and_cuts_penalty(self):
        ev = _evaluator()
        qty = np.array([100.0, 100.0])
        # A moderate dolomite dose moves basicity up toward the [1.6, 3.0] band
        # without overshooting past the maximum for this small ore base.
        no_flux = ev.evaluate_quantities(qty, flux_quantities=np.array([0.0]))
        with_flux = ev.evaluate_quantities(qty, flux_quantities=np.array([60.0]))

        b0 = no_flux.diagnostics["blend"].slag_basicity
        b1 = with_flux.diagnostics["blend"].slag_basicity
        assert b1 > b0  # dolomite raised basicity
        # Adding dolomite reduces the basicity shortfall penalty toward zero.
        assert (
            with_flux.components["penalty_slag_basicity"]
            < no_flux.components["penalty_slag_basicity"]
        )

    def test_flux_quantities_recorded_and_costed(self):
        ev = _evaluator()
        result = ev.evaluate_quantities(
            np.array([100.0, 100.0]), flux_quantities=np.array([250.0])
        )
        blend = result.diagnostics["blend"]
        assert blend.diagnostics["lp_flux_quantities_mt"]["dolomite"] == pytest.approx(250.0)
        # 250 MT * 3000 Rs/MT / 100 THM = 7500 Rs/THM flux cost.
        assert result.components["flux_cost_per_thm_rs"] == pytest.approx(7500.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
