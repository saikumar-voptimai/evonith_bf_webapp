"""DE start-population strategy.

DE used to seed only from the LP baseline, so an infeasible LP took DE down with
it and the operator got an empty result pane. An infeasible LP does not always
mean the problem is unsatisfiable - the LP optimises a *linearised* slag and
basicity model - so DE can still be worth running from an independent start.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path

from utils.bmo.lp_solver import run_lp_baseline
from utils.bmo.nonlinear_optimizer import (
    _build_random_flux_population,
    _build_random_share_population,
    run_nonlinear_optimizer,
)
from utils.bmo.types import FluxInput, OreChemistry, OreInput

_CTX = {"PCI_KG/THM": 180.0, "NUT COKE RATE KG/THM": 70.0}


class _FakePrediction:
    def __init__(self, value: float) -> None:
        self.value = value
        self.details: dict = {}
        self.used_fallback = False


class _FakeModelService:
    def predict(self, feature_payload, history_df):  # noqa: ANN001
        return _FakePrediction(12_900.0)

    def build_prebuilt_context(self, **kwargs):  # noqa: ANN003
        return None

    def predict_with_prebuilt(self, context, quantities_mt, history_df):  # noqa: ANN001
        return _FakePrediction(12_900.0)


def _ore(ore_id, *, fe, sio2, cao, price, lo=0.0, hi=100.0):
    return OreInput(
        ore_id=ore_id, display_name=ore_id.upper(), stock_mt=8000.0,
        price_rs_per_mt=price, min_share_pct=lo, max_share_pct=hi,
        chemistry=OreChemistry(fe_t_pct=fe, moisture_pct=3.0, sio2_pct=sio2,
                               cao_pct=cao, al2o3_pct=2.0),
    )


def _ores():
    return [
        _ore("sinter", fe=54.6, sio2=5.4, cao=10.8, price=5200.0),
        _ore("lean", fe=52.0, sio2=12.0, cao=0.5, price=3600.0),
        _ore("rich", fe=64.0, sio2=3.0, cao=0.5, price=6800.0),
    ]


_COMMON = dict(
    target_production_mt=2220.0, target_slag_qty_mt=5000.0, feo_in_slag_pct=0.4,
    hot_metal_target_mt=2350.0, flux_inputs=None,
)
# A basicity floor no blend of these ores can reach.
_UNREACHABLE = dict(target_slag_basicity_min=2.60, target_slag_basicity_max=None)


def _run(strategy, **kwargs):
    cfg = {"maxiter": 10, "popsize": 5, "seed": 3, "initial_solution": strategy}
    return run_nonlinear_optimizer(
        _ores(), model_service=_FakeModelService(), process_context=_CTX,
        history_df=None, de_cfg=cfg, **_COMMON, **kwargs,
    )


def test_the_hard_case_really_is_lp_infeasible():
    """Guards the premise of the tests below."""

    blend, errors = run_lp_baseline(_ores(), **_COMMON, **_UNREACHABLE)

    assert blend is None and errors


def test_lp_only_strategy_still_gives_up_when_the_lp_fails():
    """The old behaviour must remain reachable."""

    blend, errors = _run("lp", **_UNREACHABLE)

    assert blend is None
    assert any("infeasible" in e.lower() for e in errors)


@pytest.mark.parametrize("strategy", ["random", "lp_else_random"])
def test_random_start_returns_a_blend_when_the_lp_cannot(strategy):
    blend, errors = _run(strategy, **_UNREACHABLE)

    assert blend is not None, "DE should still search from a random population"
    assert errors == []
    assert blend.diagnostics["de_seed"]["strategy_used"] == "random"
    # The answer is not feasible - it cannot be - but it is reported as such
    # rather than being passed off as a solution.
    assert blend.feasible is False
    assert blend.violations


def test_the_fallback_carries_the_lp_reasons_rather_than_dropping_them():
    blend, _ = _run("lp_else_random", **_UNREACHABLE)

    seed = blend.diagnostics["de_seed"]
    assert seed["strategy_requested"] == "lp_else_random"
    assert seed["lp_seed_available"] is False
    assert seed["lp_seed_errors"], "the operator needs to know why the LP failed"


def test_lp_seed_is_preferred_when_the_lp_solves():
    blend, errors = _run("lp_else_random")

    assert errors == [] and blend is not None
    assert blend.diagnostics["de_seed"]["strategy_used"] == "lp"
    assert blend.diagnostics["de_seed"]["lp_seed_available"] is True


def test_random_strategy_never_consults_the_lp(monkeypatch):
    from utils.bmo import nonlinear_optimizer

    calls: list = []
    original = nonlinear_optimizer.run_lp_baseline

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(nonlinear_optimizer, "run_lp_baseline", _spy)
    blend, _ = _run("random")

    assert blend is not None
    assert calls == [], "random start must not pay for an LP solve"


def test_an_unknown_strategy_falls_back_to_the_safe_default():
    blend, errors = _run("nonsense", **_UNREACHABLE)

    assert blend is not None
    assert blend.diagnostics["de_seed"]["strategy_requested"] == "lp_else_random"


def test_shipped_config_enables_the_fallback():
    path = Path(__file__).resolve().parents[1] / "src" / "config" / "setting_bmo.yml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["bmo"]

    assert cfg["optimization"]["initial_solution"] == "lp_else_random"


# --------------------------------------------------------------------------
# Population builders
# --------------------------------------------------------------------------


def test_random_share_population_respects_bounds_and_sums_to_one():
    lo = np.array([0.10, 0.00, 0.20])
    hi = np.array([0.60, 0.50, 0.70])

    pop = _build_random_share_population(
        min_shares=lo, max_shares=hi, sample_count=40, seed=1
    )

    assert pop.shape == (40, 3)
    assert np.all(pop >= lo - 1e-9)
    assert np.all(pop <= hi + 1e-9)
    assert np.allclose(pop.sum(axis=1), 1.0, atol=1e-6)


def test_random_share_population_actually_spreads():
    """A population clustered on one point would not be a search.

    The LP-seeded population deliberately stays within a few percent of its
    seed; this one has to cover the range, or falling back to it buys nothing.
    """

    lo = np.zeros(3)
    hi = np.ones(3)

    pop = _build_random_share_population(
        min_shares=lo, max_shares=hi, sample_count=60, seed=1
    )

    # Every ore should be explored from near-zero to a dominant share.
    assert np.all(pop.min(axis=0) < 0.10)
    assert np.all(pop.max(axis=0) > 0.60)


def test_random_share_population_is_reproducible():
    kw = dict(min_shares=np.zeros(3), max_shares=np.ones(3), sample_count=20)

    assert np.allclose(
        _build_random_share_population(seed=7, **kw),
        _build_random_share_population(seed=7, **kw),
    )


def test_random_flux_population_includes_the_zero_flux_row():
    """Charging no flux is a common answer and must be in the start population."""

    pop = _build_random_flux_population(
        flux_bounds=[(0.0, 500.0), (0.0, 300.0)], sample_count=25, seed=2
    )

    assert pop.shape == (25, 2)
    assert np.allclose(pop[0], [0.0, 0.0])
    assert np.all(pop[:, 0] <= 500.0 + 1e-9)
    assert np.all(pop[:, 1] <= 300.0 + 1e-9)
    assert np.all(pop >= -1e-9)


def test_random_start_respects_flux_stock():
    limestone = FluxInput(
        flux_id="limestone", display_name="Limestone", enabled=True,
        wet_qty_mt=0.0, moisture_pct=0.2, sio2_pct=4.0, cao_pct=47.7,
        mgo_pct=5.1, loi_pct=40.8, price_rs_per_mt=1800.0,
        stock_mt=40.0, optimizable=True,
    )
    cfg = {"maxiter": 8, "popsize": 5, "seed": 3, "initial_solution": "random"}
    blend, _ = run_nonlinear_optimizer(
        _ores(), model_service=_FakeModelService(), process_context=_CTX,
        history_df=None, de_cfg=cfg,
        **{**_COMMON, "flux_inputs": [limestone]},
    )

    assert blend is not None
    solved = blend.diagnostics["lp_flux_quantities_mt"]["limestone"]
    assert 0.0 - 1e-6 <= solved <= 40.0 + 1e-6


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
