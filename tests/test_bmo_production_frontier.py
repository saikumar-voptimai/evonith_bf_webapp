from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.bmo.production_frontier import (
    FrontierSettings,
    identify_knee,
    production_targets,
    run_frontier_simulation,
    validate_frontier_inputs,
)
from utils.bmo.types import BlendEvaluation, OreChemistry, OreInput


def _ores() -> list[OreInput]:
    return [
        OreInput(
            ore_id="a",
            display_name="Ore A",
            stock_mt=4000.0,
            price_rs_per_mt=1000.0,
            min_share_pct=0.0,
            max_share_pct=80.0,
            chemistry=OreChemistry(
                fe_t_pct=64.0,
                moisture_pct=2.0,
                sio2_pct=3.0,
                al2o3_pct=2.0,
            ),
        ),
        OreInput(
            ore_id="b",
            display_name="Ore B",
            stock_mt=3000.0,
            price_rs_per_mt=1200.0,
            min_share_pct=20.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(
                fe_t_pct=60.0,
                moisture_pct=4.0,
                sio2_pct=5.0,
                al2o3_pct=3.0,
            ),
        ),
    ]


def test_production_grid_is_inclusive_and_solve_limit_is_guarded() -> None:
    settings = FrontierSettings(
        production_min_mt=1700.0,
        production_max_mt=1735.0,
        production_step_mt=10.0,
    )
    assert production_targets(settings).tolist() == [
        1700.0,
        1710.0,
        1720.0,
        1730.0,
        1735.0,
    ]

    oversized = FrontierSettings(
        production_min_mt=1000.0,
        production_max_mt=6000.0,
        production_step_mt=1.0,
        variation_scenarios=1,
    )
    assert any("5,000" in error for error in validate_frontier_inputs(_ores(), oversized))


def test_knee_reports_curvature_plateau_and_capacity_onset() -> None:
    production = np.arange(1700.0, 2310.0, 10.0)
    unit_cost = 14000.0 + np.maximum(production - 2050.0, 0.0) ** 2 / 100.0
    curve = pd.DataFrame(
        {
            "production_mt": production,
            "unit_cost_rs_per_thm": unit_cost,
            "charging_utilization_pct": np.where(production >= 2050.0, 100.0, 95.0),
            "feasible": True,
        }
    )

    result = identify_knee(curve, tolerance_rs_per_thm=25.0)

    assert 2020.0 <= float(result["knee_production_mt"]) <= 2080.0
    assert result["capacity_onset_production_mt"] == 2050.0
    assert result["plateau_end_production_mt"] == 2100.0
    assert result["max_feasible_production_mt"] == 2300.0


def test_frontier_scenarios_are_reproducible_and_report_real_solves(
    monkeypatch,
) -> None:
    def fake_lp(ores, *, hot_metal_target_mt, max_burden_qty_mt, **kwargs):
        production = float(hot_metal_target_mt)
        quantities = {"a": production, "b": production * 0.5}
        ore_cost = sum(
            quantities[ore.ore_id] * float(ore.price_rs_per_mt) for ore in ores
        )
        blend = BlendEvaluation(
            quantities_mt=quantities,
            shares_pct={"a": 66.6667, "b": 33.3333},
            total_qty_mt=sum(quantities.values()),
            ore_cost_total_rs=ore_cost,
            ore_cost_per_thm_rs=ore_cost / production,
            fuel_cost_per_thm_rs=0.0,
            objective_rs_per_thm=ore_cost / production,
            fe_t_pct=62.0,
            effective_fe_pct=62.0,
            fe_production_mt=production,
            slag_pct=15.0,
            slag_mt=production * 0.3,
            feasible=True,
            violations=[],
            slag_rate_kg_per_thm=300.0,
            slag_basicity=1.1,
            diagnostics={"total_burden_qty_mt": min(max_burden_qty_mt, 3000.0)},
        )
        return blend, []

    monkeypatch.setattr(
        "utils.bmo.production_frontier.run_lp_baseline",
        fake_lp,
    )
    settings = FrontierSettings(
        production_min_mt=1700.0,
        production_max_mt=1720.0,
        production_step_mt=10.0,
        variation_scenarios=2,
        random_seed=99,
    )
    events: list[dict] = []
    kwargs = dict(
        settings=settings,
        hm_fe_pct=94.5,
        feo_in_slag_pct=0.4,
        model_to_plant_slag_factor=1.0,
    )

    first = run_frontier_simulation(_ores(), progress_callback=events.append, **kwargs)
    second = run_frontier_simulation(_ores(), **kwargs)

    assert first["solve_count"] == 9
    assert len(first["frontier"]) == 9
    assert events[-1]["stage"] == "complete"
    pd.testing.assert_frame_equal(first["scenarios"], second["scenarios"])


def test_product_is_after_snapshots_and_progress_is_not_artificial() -> None:
    root = Path(__file__).resolve().parents[1]
    page = (root / "src/custom_pages/9_Blend_Optimizer.py").read_text(
        encoding="utf-8"
    )
    ui = (root / "src/ui/bmo/production_frontier.py").read_text(encoding="utf-8")

    assert page.index("render_snapshot_panel(") < page.index(
        "render_production_frontier("
    )
    assert "Stochastic Burden Frontier Engine (SBFE)" in ui
    assert "LP solve {completed:,} of {total:,}" in ui
    assert "time.sleep" not in ui
