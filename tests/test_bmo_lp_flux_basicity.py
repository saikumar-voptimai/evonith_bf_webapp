"""Tests for LP-optimised fluxes satisfying the slag basicity bounds.

The LP baseline treats optimisable fluxes (dolomite, quartz) as decision
variables so it can add just enough flux to hold slag basicity (CaO/SiO2)
within the operator bounds -- a fast linear alternative to the DE solver.
"""

from __future__ import annotations

import pytest

from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.types import FluxInput, OreChemistry, OreInput


def _ore(ore_id: str, *, sio2: float, cao: float, fe_t: float = 62.0) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=ore_id.upper(),
        stock_mt=5000.0,
        price_rs_per_mt=1000.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(
            fe_t_pct=fe_t, moisture_pct=3.0, sio2_pct=sio2, cao_pct=cao
        ),
    )


def _dolomite(*, optimizable: bool = True, price: float = 3000.0) -> FluxInput:
    return FluxInput(
        flux_id="dolomite",
        display_name="Dolomite",
        enabled=True,
        optimizable=optimizable,
        price_rs_per_mt=price,
        stock_mt=500.0,
        sio2_pct=1.7,
        cao_pct=30.2,
        mgo_pct=22.3,
    )


def _limestone(*, optimizable: bool = True, price: float = 1800.0) -> FluxInput:
    return FluxInput(
        flux_id="limestone",
        display_name="Limestone",
        enabled=True,
        optimizable=optimizable,
        price_rs_per_mt=price,
        stock_mt=500.0,
        sio2_pct=4.0,
        cao_pct=47.7,
        mgo_pct=5.1,
    )


def _quartz(*, optimizable: bool = True) -> FluxInput:
    return FluxInput(
        flux_id="quartz",
        display_name="Quartz",
        enabled=True,
        optimizable=optimizable,
        price_rs_per_mt=2000.0,
        stock_mt=500.0,
        sio2_pct=96.5,
    )


class TestLpFluxBasicity:
    def test_low_basicity_adds_dolomite_to_meet_minimum(self):
        # Ores alone give a very low basicity (~0.15); the LP must add dolomite
        # (high CaO) to reach the 0.8 minimum.
        ores = [_ore("ore_a", sio2=8.0, cao=1.0), _ore("ore_b", sio2=7.0, cao=1.5)]
        blend, errors = run_lp_baseline(
            ores,
            target_production_mt=100.0,
            target_slag_qty_mt=2000.0,
            feo_in_slag_pct=0.0,
            target_slag_basicity_min=0.8,
            target_slag_basicity_max=3.0,
            flux_inputs=[_dolomite(), _quartz()],
        )
        assert errors == []
        assert blend is not None and blend.feasible
        assert blend.slag_basicity >= 0.8 - 1e-3
        flux_qty = blend.diagnostics["lp_flux_quantities_mt"]
        assert flux_qty["dolomite"] > 1.0  # dolomite was added
        assert flux_qty["quartz"] == pytest.approx(0.0, abs=1e-6)  # quartz not needed

    def test_high_basicity_adds_quartz_to_meet_maximum(self):
        # Ores alone give a high basicity (~2.0); the LP must add quartz (SiO2)
        # to bring it down to the 1.2 maximum.
        ores = [_ore("ore_a", sio2=4.0, cao=8.0), _ore("ore_b", sio2=4.0, cao=8.0)]
        blend, errors = run_lp_baseline(
            ores,
            target_production_mt=100.0,
            target_slag_qty_mt=2000.0,
            feo_in_slag_pct=0.0,
            target_slag_basicity_min=0.5,
            target_slag_basicity_max=1.2,
            flux_inputs=[_dolomite(), _quartz()],
        )
        assert errors == []
        assert blend is not None and blend.feasible
        assert blend.slag_basicity <= 1.2 + 1e-3
        flux_qty = blend.diagnostics["lp_flux_quantities_mt"]
        assert flux_qty["quartz"] > 1.0  # quartz was added
        assert flux_qty["dolomite"] == pytest.approx(0.0, abs=1e-6)

    def test_without_optimisable_flux_low_basicity_is_infeasible(self):
        # Same low-basicity ores, but fluxes fixed (not optimisable): the ore
        # blend alone cannot reach the basicity minimum, so the LP is infeasible.
        ores = [_ore("ore_a", sio2=8.0, cao=1.0), _ore("ore_b", sio2=7.0, cao=1.5)]
        blend, errors = run_lp_baseline(
            ores,
            target_production_mt=100.0,
            target_slag_qty_mt=2000.0,
            feo_in_slag_pct=0.0,
            target_slag_basicity_min=0.8,
            target_slag_basicity_max=3.0,
            flux_inputs=[_dolomite(optimizable=False), _quartz(optimizable=False)],
            _explain=False,
        )
        assert blend is None
        assert errors  # infeasible

    def test_explains_basicity_flux_conflict_with_slag_cap(self):
        # Dolomite can satisfy the basicity minimum, but the extra flux pushes
        # final slag above this deliberately tight cap.
        ores = [_ore("ore_a", sio2=8.0, cao=1.0), _ore("ore_b", sio2=7.0, cao=1.5)]
        blend, errors = run_lp_baseline(
            ores,
            target_production_mt=100.0,
            target_slag_qty_mt=25.0,
            feo_in_slag_pct=0.0,
            target_slag_basicity_min=0.8,
            target_slag_basicity_max=3.0,
            flux_inputs=[_dolomite(), _quartz()],
        )

        assert blend is None
        assert any("Max Slag cap is lifted" in error for error in errors)
        assert any("LP would add dolomite" in error for error in errors)

    def test_low_basicity_picks_cheapest_cao_source_by_price(self):
        # Both limestone and dolomite can raise basicity; the LP should pick the
        # cheaper CaO source. Limestone (47.7% CaO @ 1800) is cheaper per unit CaO
        # than dolomite (30.2% CaO @ 3000), so limestone wins.
        ores = [_ore("ore_a", sio2=8.0, cao=1.0), _ore("ore_b", sio2=7.0, cao=1.5)]
        blend, errors = run_lp_baseline(
            ores,
            target_production_mt=100.0,
            target_slag_qty_mt=2000.0,
            feo_in_slag_pct=0.0,
            target_slag_basicity_min=0.9,
            target_slag_basicity_max=3.0,
            flux_inputs=[_dolomite(), _limestone(), _quartz()],
        )
        assert errors == []
        fq = blend.diagnostics["lp_flux_quantities_mt"]
        assert fq["limestone"] > 1.0
        assert fq["dolomite"] == pytest.approx(0.0, abs=1e-6)

    def test_dolomite_wins_when_cheaper_than_limestone(self):
        # Flip the economics: make dolomite the cheaper CaO source; LP switches.
        ores = [_ore("ore_a", sio2=8.0, cao=1.0), _ore("ore_b", sio2=7.0, cao=1.5)]
        blend, errors = run_lp_baseline(
            ores,
            target_production_mt=100.0,
            target_slag_qty_mt=2000.0,
            feo_in_slag_pct=0.0,
            target_slag_basicity_min=0.9,
            target_slag_basicity_max=3.0,
            flux_inputs=[_dolomite(price=800.0), _limestone(price=6000.0), _quartz()],
        )
        assert errors == []
        fq = blend.diagnostics["lp_flux_quantities_mt"]
        assert fq["dolomite"] > 1.0
        assert fq["limestone"] == pytest.approx(0.0, abs=1e-6)

    def test_basicity_already_in_bounds_adds_no_flux(self):
        # Ore blend basicity (~1.0) already within [0.8, 1.5]; flux costs money,
        # so the LP should add none.
        ores = [_ore("ore_a", sio2=5.0, cao=5.0), _ore("ore_b", sio2=5.0, cao=5.0)]
        blend, errors = run_lp_baseline(
            ores,
            target_production_mt=100.0,
            target_slag_qty_mt=2000.0,
            feo_in_slag_pct=0.0,
            target_slag_basicity_min=0.8,
            target_slag_basicity_max=1.5,
            flux_inputs=[_dolomite(), _quartz()],
        )
        assert errors == []
        assert blend is not None and blend.feasible
        flux_qty = blend.diagnostics["lp_flux_quantities_mt"]
        assert flux_qty["dolomite"] == pytest.approx(0.0, abs=1e-6)
        assert flux_qty["quartz"] == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
