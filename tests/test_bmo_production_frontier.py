from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ui.bmo.production_frontier as frontier_ui
from utils.bmo.production_frontier import (
    FrontierSettings,
    identify_knee,
    identify_knee_candidates,
    production_targets,
    run_frontier_simulation,
    validate_frontier_inputs,
)
from utils.bmo.types import BlendEvaluation, FluxInput, OreChemistry, OreInput
from ui.bmo.production_frontier import (
    _default_selected_production,
    _duration_text,
    _estimated_remaining_seconds,
    _frontier_figure,
    _selected_production_from_event,
)


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

    # 5,001 targets x 51 runs = 255,051 solves, past the 200,000 limit.
    oversized = FrontierSettings(
        production_min_mt=1000.0,
        production_max_mt=6000.0,
        production_step_mt=1.0,
        variation_scenarios=50,
    )
    assert any(
        "200,000" in error for error in validate_frontier_inputs(_ores(), oversized)
    )


def test_thousand_scenarios_at_five_mt_fit_the_solve_limit() -> None:
    # The default range at the default 5 MT step: 161 targets x 1,001 runs.
    settings = FrontierSettings(variation_scenarios=1000)

    assert settings.production_step_mt == 5.0
    assert len(production_targets(settings)) == 161
    assert validate_frontier_inputs(_ores(), settings) == []
    too_many = FrontierSettings(variation_scenarios=1001)
    assert any(
        "1,000" in error for error in validate_frontier_inputs(_ores(), too_many)
    )


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


def test_multiple_knees_default_to_lowest_cost_candidate() -> None:
    production = np.arange(1700.0, 1910.0, 10.0)
    unit_cost = np.array(
        [
            100,
            100,
            100,
            101,
            104,
            108,
            109,
            109,
            110,
            114,
            121,
            122,
            122,
            123,
            129,
            140,
            141,
            141,
            142,
            143,
            144,
        ],
        dtype=float,
    )
    curve = pd.DataFrame(
        {
            "production_mt": production,
            "unit_cost_rs_per_thm": unit_cost,
            "charging_utilization_pct": np.where(production >= 1800.0, 100.0, 95.0),
            "feasible": True,
        }
    )

    candidates = identify_knee_candidates(curve)
    summary = identify_knee(curve, tolerance_rs_per_thm=25.0)

    assert len(candidates) >= 2
    lowest_cost = min(
        candidates,
        key=lambda item: (
            float(item["unit_cost_rs_per_thm"]),
            float(item["production_mt"]),
        ),
    )
    assert summary["knee_production_mt"] == lowest_cost["production_mt"]
    assert summary["knee_candidates"] == candidates


def test_selected_frontier_point_replaces_default_knee() -> None:
    result = {
        "base_knees": [
            {"production_mt": 2200.0, "unit_cost_rs_per_thm": 14510.0},
            {"production_mt": 2240.0, "unit_cost_rs_per_thm": 14535.0},
        ],
        "frontier": pd.DataFrame(
            {
                "is_base": [True, True],
                "feasible": [True, True],
                "production_mt": [2200.0, 2240.0],
                "unit_cost_rs_per_thm": [14510.0, 14535.0],
            }
        ),
    }
    event = {
        "selection": {
            "points": [
                {
                    "customdata": [2240.0, "base"],
                    "x": 2240.0,
                }
            ]
        }
    }

    assert _default_selected_production(result) == 2200.0
    assert _selected_production_from_event(event) == 2240.0
    assert _selected_production_from_event(
        {"selection": {"points": [{"x": 2240.0}]}}
    ) == 2240.0
    assert _selected_production_from_event({"selection": {"points": []}}) is None


def test_frontier_eta_uses_observed_iteration_rate() -> None:
    assert _estimated_remaining_seconds(
        elapsed_seconds=40.0,
        completed=20,
        total=100,
    ) == 160.0
    assert _estimated_remaining_seconds(
        elapsed_seconds=0.0,
        completed=0,
        total=100,
    ) is None
    assert _duration_text(3_661.0) == "1h 01m 01s"


def test_frontier_selection_callback_persists_clicked_point(monkeypatch) -> None:
    state = {
        frontier_ui._FRONTIER_CHART_KEY: {
            "selection": {"points": [{"x": 2240.0}]}
        }
    }
    monkeypatch.setattr(frontier_ui.st, "session_state", state)

    frontier_ui._sync_frontier_selection()

    assert state[frontier_ui._SELECTED_PRODUCTION_KEY] == 2240.0


def test_slider_and_dropdown_share_the_feasible_point_selection(monkeypatch) -> None:
    result = {
        "frontier": pd.DataFrame(
            {
                "is_base": [True, True, True, False],
                "feasible": [True, False, True, True],
                "production_mt": [2240.0, 2220.0, 2200.0, 2210.0],
                "unit_cost_rs_per_thm": [14535.0, 14520.0, 14510.0, 14515.0],
            }
        ),
    }
    points = frontier_ui._feasible_points(result)
    productions = [float(value) for value in points["production_mt"]]

    # Only feasible current-state points, in production order.
    assert productions == [2200.0, 2240.0]
    # A chart click's JSON float snaps to the exact option; off-frontier
    # targets do not.
    assert frontier_ui._snap_to_point(2240.0000001, productions) == 2240.0
    assert frontier_ui._snap_to_point(2220.0, productions) is None
    assert frontier_ui._snap_to_point(None, productions) is None

    state = {frontier_ui._POINT_SLIDER_KEY: 2240.0}
    monkeypatch.setattr(frontier_ui.st, "session_state", state)
    frontier_ui._sync_point_control(frontier_ui._POINT_SLIDER_KEY)
    assert state[frontier_ui._SELECTED_PRODUCTION_KEY] == 2240.0

    state[frontier_ui._POINT_PICKER_KEY] = 2200.0
    frontier_ui._sync_point_control(frontier_ui._POINT_PICKER_KEY)
    assert state[frontier_ui._SELECTED_PRODUCTION_KEY] == 2200.0


def test_frontier_figure_exposes_feasible_points_and_knees_for_selection() -> None:
    result = {
        "base_knees": [
            {
                "production_mt": 2200.0,
                "unit_cost_rs_per_thm": 14510.0,
                "reason": "charging capacity onset",
            }
        ],
        "frontier": pd.DataFrame(
            {
                "is_base": [True, True, False, False],
                "feasible": [True, True, True, True],
                "production_mt": [2200.0, 2240.0, 2200.0, 2240.0],
                "unit_cost_rs_per_thm": [14510.0, 14535.0, 14520.0, 14545.0],
            }
        ),
        "aggregate": pd.DataFrame(
            {
                "production_mt": [2200.0, 2240.0],
                "p90_unit_cost_rs_per_thm": [14530.0, 14555.0],
                "p10_unit_cost_rs_per_thm": [14510.0, 14535.0],
                "median_unit_cost_rs_per_thm": [14520.0, 14545.0],
            }
        ),
        "scenarios": pd.DataFrame(
            {
                "is_base": [True, False],
                "knee_production_mt": [2200.0, 2240.0],
            }
        ),
    }

    figure = _frontier_figure(result, selected_production_mt=2240.0)
    traces = {trace.name: trace for trace in figure.data}

    assert "Feasible current-state points" in traces
    assert "Detected knees" in traces
    assert "Selected solution" in traces
    assert list(traces["Feasible current-state points"].customdata[:, 1]) == [
        "base",
        "base",
    ]
    assert traces["Selected solution"].x[0] == 2240.0


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


def _variation_result(knees: list[float], feasibility: list[float]) -> dict:
    targets = [2200.0, 2220.0, 2240.0, 2260.0, 2280.0]
    frontier = pd.DataFrame(
        {
            "is_base": True,
            "feasible": True,
            "production_mt": targets,
            "unit_cost_rs_per_thm": [14000.0, 14000.0, 14100.0, 14300.0, 14600.0],
        }
    )
    return {
        "frontier": frontier,
        "aggregate": pd.DataFrame(
            {
                "production_mt": targets,
                "feasibility_pct": feasibility,
                "p10_unit_cost_rs_per_thm": 14000.0,
                "p90_unit_cost_rs_per_thm": 14200.0,
                "median_unit_cost_rs_per_thm": 14100.0,
            }
        ),
        "scenarios": pd.DataFrame(
            {
                "is_base": [True] + [False] * len(knees),
                "knee_production_mt": [2240.0, *knees],
            }
        ),
        "base_knees": [],
    }


def test_one_variation_scenario_hides_the_meaningless_spread() -> None:
    result = _variation_result([2240.0], [100.0, 100.0, 100.0, 0.0, 0.0])

    names = [trace.name for trace in _frontier_figure(result).data]

    assert frontier_ui._varied_scenario_count(result) == 1
    assert frontier_ui._varied_scenario_count(result) < frontier_ui.MIN_VARIATION_SCENARIOS
    assert "Scenario median" not in names
    assert "Daily variation P10-P90" not in names


def test_variation_summary_and_knee_counts_use_the_production_grid() -> None:
    knees = [2220.0, 2240.0, 2240.0, 2240.0, 2260.0, 2240.0]
    # 2260 recovers after an infeasible 2240, so "every scenario" stops at 2220.
    result = _variation_result(knees, [100.0, 100.0, 40.0, 100.0, 0.0])

    summary = frontier_ui._variation_summary(result)
    figure = frontier_ui._knee_distribution_figure(result)

    assert summary["all_feasible_up_to"] == 2220.0
    assert summary["half_feasible_up_to"] == 2220.0
    assert "Scenario median" in [
        trace.name for trace in _frontier_figure(result).data
    ]
    bars = figure.data[0]
    assert list(bars.x) == [2220.0, 2240.0, 2260.0]
    assert list(bars.y) == [1, 4, 1]
    assert bars.width == 16.0  # 0.8 of the 20 MT grid step
    labels = [annotation.text for annotation in figure.layout.annotations]
    # Quantiles stay on the grid: no 2,230 between two production targets.
    assert labels == ["P10 2,220", "Median 2,240", "P90 2,260"]
    assert (summary["p10"], summary["p90"]) == (2220.0, 2260.0)

    same = frontier_ui._knee_distribution_figure(
        _variation_result([2240.0] * 6, [100.0] * 5)
    )
    assert [a.text for a in same.layout.annotations] == ["P10 / Median / P90 2,240"]


def _single_target(production_mt: float, **overrides) -> FrontierSettings:
    values = dict(
        production_min_mt=production_mt,
        production_max_mt=production_mt,
        production_step_mt=10.0,
        max_slag_rate_kg_per_thm=1000.0,
        basicity_min=0.0,
        basicity_max=5.0,
        enforce_charging_capacity=False,
        variation_scenarios=0,
    )
    values.update(overrides)
    return FrontierSettings(**values)


def test_a_fixed_flux_is_priced_not_free() -> None:
    # The LP costs only the flux it decides. A fixed flux used to enter the
    # slag balance at zero cost, so the frontier read as if it were free.
    limestone = FluxInput(
        flux_id="limestone",
        display_name="Limestone",
        wet_qty_mt=60.0,
        cao_pct=50.0,
        sio2_pct=2.0,
        price_rs_per_mt=1500.0,
        stock_mt=5000.0,
        optimizable=False,
    )

    result = run_frontier_simulation(
        _ores(),
        settings=_single_target(1700.0),
        hm_fe_pct=94.5,
        feo_in_slag_pct=0.4,
        model_to_plant_slag_factor=1.0,
        flux_inputs=[limestone],
    )
    row = result["frontier"].iloc[0]

    assert bool(row["feasible"])
    assert row["flux_mt__limestone"] == pytest.approx(60.0)
    assert row["flux_cost_per_thm_rs"] == pytest.approx(60.0 * 1500.0 / 1700.0)
    assert row["unit_cost_rs_per_thm"] == pytest.approx(
        row["ore_cost_per_thm_rs"] + row["flux_cost_per_thm_rs"]
    )


def test_flux_the_slag_limits_need_is_bought_and_scales_with_production() -> None:
    # Ores without CaO cannot reach basicity 1.0, so limestone is required and
    # its tonnage must follow the production target.
    fluxes = [
        FluxInput(
            flux_id="limestone",
            display_name="Limestone",
            cao_pct=50.0,
            sio2_pct=2.0,
            price_rs_per_mt=1500.0,
            stock_mt=5000.0,
            max_qty_mt=2000.0,
            optimizable=True,
        ),
        FluxInput(
            flux_id="dolomite",
            display_name="Dolomite",
            enabled=False,
            cao_pct=30.0,
            mgo_pct=20.0,
            price_rs_per_mt=1000.0,
            stock_mt=5000.0,
            optimizable=True,
        ),
    ]
    settings = _single_target(
        1700.0, production_max_mt=1720.0, basicity_min=1.0, basicity_max=1.2
    )

    result = run_frontier_simulation(
        _ores(),
        settings=settings,
        hm_fe_pct=94.5,
        feo_in_slag_pct=0.4,
        model_to_plant_slag_factor=1.0,
        flux_inputs=fluxes,
    )
    frontier = result["frontier"]

    assert frontier["feasible"].all()
    assert (frontier["flux_cost_per_thm_rs"] > 0.0).all()
    assert frontier["flux_mt__limestone"].is_monotonic_increasing
    # A disabled flux is not offered, so it has no column to report.
    assert result["flux_names"] == {"limestone": "Limestone"}
    assert result["flux_prices"] == {"limestone": 1500.0}


def test_sbfe_lists_every_material_and_flux() -> None:
    materials = frontier_ui._material_editor_frame(_ores(), {"b"})
    assert materials.set_index("ore_id")["Use"].to_dict() == {"a": False, "b": True}

    fluxes = [
        FluxInput(
            "quartz", "Quartz", optimizable=True, stock_mt=500.0, max_qty_mt=100.0
        ),
        FluxInput(
            "limestone", "Limestone", optimizable=False, wet_qty_mt=40.0, stock_mt=5000.0
        ),
        FluxInput(
            "dolomite", "Dolomite", enabled=False, optimizable=True, stock_mt=2000.0
        ),
    ]
    editor = frontier_ui._flux_editor_frame(fluxes)

    # Non-optimisable fluxes are listed too; they used to be hidden.
    assert list(editor["Flux"]) == ["Quartz", "Limestone", "Dolomite"]
    assert list(editor["Use"]) == [True, True, False]
    limestone = editor.set_index("flux_id").loc["limestone"]
    assert (limestone["Min MT"], limestone["Max MT"]) == (40.0, 40.0)

    solved = frontier_ui._fluxes_from_editor(editor, fluxes)
    assert all(flux.optimizable for flux in solved)
    assert [flux.enabled for flux in solved] == [True, True, False]
    assert (solved[1].min_qty_mt, solved[1].max_qty_mt) == (40.0, 40.0)
    assert solved[1].wet_qty_mt == 0.0


def test_sbfe_slag_limits_follow_the_main_page(monkeypatch) -> None:
    state: dict = {}
    monkeypatch.setattr(frontier_ui.st, "session_state", state)
    limits = {
        "max_slag_rate_kg_per_thm": 340.0,
        "basicity_min": 1.05,
        "basicity_max": 1.15,
        "t_basicity_min": None,
        "t_basicity_max": None,
        "al2o3_max_pct": 20.0,
        "mgo_min_pct": 6.5,
        "mgo_al2o3_min": 0.345,
    }

    frontier_ui._seed_from_page(frontier_ui._slag_limit_defaults(limits))
    assert state["bmo_frontier_max_slag_rate"] == 340.0
    assert state["bmo_frontier_basicity_min"] == 1.05
    assert state["bmo_frontier_mgo_min"] == 6.5
    assert state["bmo_frontier_mgo_al2o3_min"] == 0.345
    assert state["bmo_frontier_t_basicity_min"] == 0.0  # None = off

    # An SBFE-only edit survives a rerun while the page limits are unchanged...
    state["bmo_frontier_mgo_min"] = 7.0
    frontier_ui._seed_from_page(frontier_ui._slag_limit_defaults(limits))
    assert state["bmo_frontier_mgo_min"] == 7.0

    # ...and a change on the main page still reaches SBFE.
    frontier_ui._seed_from_page(
        frontier_ui._slag_limit_defaults({**limits, "mgo_min_pct": 6.0})
    )
    assert state["bmo_frontier_mgo_min"] == 6.0


def test_page_gives_sbfe_the_whole_yard_and_its_slag_limits() -> None:
    root = Path(__file__).resolve().parents[1]
    page = (root / "src/custom_pages/9_Blend_Optimizer.py").read_text(
        encoding="utf-8"
    )
    call = page[page.index("render_production_frontier(") :]

    assert "edited_df.assign(selected=True)" in call
    assert "selected_ore_ids=" in call
    assert '"mgo_min_pct": target_slag_mgo_min_pct' in call
    assert '"basicity_min": target_slag_basicity_min' in call
