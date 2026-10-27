"""services.bmo: page 9's orchestration without Streamlit (SV-3.4, #301).

The end-to-end parity of the page with its pre-refactor output is ``tests/test_bmo_golden.py`` (five plant days). These
tests pin the pieces on small fixtures: the DE guardrail, the run refusals, the coke anchor, the comparison tables, the
DE search table, the Si tile and the snapshot capture, and that the module reads no clock of its own (the page passes
``now``, which is what lets the Sandbox and the golden runs replay a day).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from services import bmo
from utils.bmo.types import BlendEvaluation, OreChemistry, OreInput

SERVICE = Path(bmo.__file__)


def _ore(ore_id: str, name: str, *, fe: float = 60.0, moisture: float = 0.0) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=name,
        stock_mt=1000.0,
        price_rs_per_mt=100.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=fe, moisture_pct=moisture),
    )


def _blend(*, objective: float = 4125.0, feasible: bool = True, **diagnostics: object) -> BlendEvaluation:
    return BlendEvaluation(
        quantities_mt={"ore_a": 100.0, "ore_b": 50.0},
        shares_pct={"ore_a": 66.67, "ore_b": 33.33},
        total_qty_mt=150.0,
        ore_cost_total_rs=12500.0,
        ore_cost_per_thm_rs=125.0,
        fuel_cost_per_thm_rs=4000.0,
        objective_rs_per_thm=objective,
        fe_t_pct=60.0,
        effective_fe_pct=60.0,
        fe_production_mt=75.0,
        slag_pct=20.0,
        slag_mt=30.0,
        feasible=feasible,
        violations=[],
        diagnostics={"hot_metal_target_mt": 100.0, **diagnostics},
    )


# ---------------------------------------------------------------------------- architecture
def test_service_reads_no_clock_and_no_streamlit() -> None:
    source = SERVICE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(name == "streamlit" or name.startswith(("streamlit.", "ui.")) for name in imports)
    # The clock is an argument (page 9 passes it), so a replayed day never reads today's time here.
    assert "Timestamp.now" not in source
    assert "datetime.now" not in source


def test_page_blend_table_is_the_service_builder() -> None:
    from ui.bmo.components import build_blend_table_df

    assert build_blend_table_df is bmo.blend_table


# ---------------------------------------------------------------------------- the LP plan
def test_fall_back_to_lp_keeps_a_cheaper_feasible_de_blend() -> None:
    lp = _blend(objective=4125.0)
    de = _blend(objective=4100.0)
    assert bmo.fall_back_to_lp(lp, de) is de


@pytest.mark.parametrize(
    ("de", "reason"),
    [
        (None, "no DE result"),
        (_blend(objective=4100.0, feasible=False), "DE infeasible"),
        (_blend(objective=4100.0, flux_cost_per_thm_rs=30.0), "DE dearer once its flux is counted"),
    ],
)
def test_fall_back_to_lp_reports_the_lp_blend(de: BlendEvaluation | None, reason: str) -> None:
    lp = _blend(objective=4125.0)
    result = bmo.fall_back_to_lp(lp, de)
    assert result is not lp, reason
    assert result.objective_rs_per_thm == lp.objective_rs_per_thm
    assert result.diagnostics["de_fell_back_to_lp"] is True
    assert "de_fell_back_to_lp" not in lp.diagnostics


def test_fall_back_to_lp_without_a_feasible_lp_keeps_de() -> None:
    de = _blend(objective=9999.0, feasible=False)
    assert bmo.fall_back_to_lp(_blend(feasible=False), de) is de
    assert bmo.fall_back_to_lp(None, None) is None


@dataclass
class _Prediction:
    usable: bool
    value_kg_per_thm: float | None

    def to_dict(self) -> dict[str, object]:
        return {"usable": self.usable, "value_kg_per_thm": self.value_kg_per_thm}


@dataclass
class _Anchor:
    usable: bool
    coke_rate_kg_thm: float


def test_coke_anchor_takes_the_selected_model_only() -> None:
    data_driven = bmo.coke_anchor("data_driven", _Prediction(True, 313.5), _Anchor(True, 330.0))
    assert data_driven == bmo.CokeAnchor("data_driven", 313.5, {"usable": True, "value_kg_per_thm": 313.5})
    unusable = bmo.coke_anchor("data_driven", _Prediction(False, 313.5), None)
    assert unusable.rate_kg_thm is None and unusable.prediction_details == {"usable": False, "value_kg_per_thm": 313.5}
    physics = bmo.coke_anchor("energy_balance", _Prediction(True, 313.5), _Anchor(True, 326.8))
    assert physics == bmo.CokeAnchor("energy_balance", 326.8, None)
    assert bmo.coke_anchor("energy_balance", None, _Anchor(False, 326.8)).rate_kg_thm is None


def _case(**overrides: object) -> bmo.BlendCase:
    values: dict[str, object] = {
        "ores": [_ore("ore_a", "ORE A"), _ore("ore_b", "ORE B")],
        "target_fe_mt": 2200.0,
        "target_production_mt": 2350.0,
        "target_slag_qty_mt": 728.5,
        "feo_in_slag_pct": 0.4,
        "target_slag_basicity_min": 1.0,
        "target_slag_basicity_max": 1.2,
        "target_slag_t_basicity_min": None,
        "target_slag_t_basicity_max": None,
        "target_slag_al2o3_max_pct": 18.0,
        "target_slag_mgo_min_pct": None,
        "target_slag_mgo_al2o3_ratio_min": 0.0,
        "max_burden_qty_mt": None,
        "fuel_ash_inputs": [],
        "flux_inputs": [],
        "dust_inputs": [],
        "slag_balance_settings": None,
        "charge_mass_mt": 26.4,
        "coke_correction_settings": None,
        "observed_slag_rate_kg_per_thm": 320.0,
        "live_si_pct": 0.35,
        "fuel_rate_anchor_basis": "data_driven",
        "lookback_hours": 6.0,
    }
    values.update(overrides)
    return bmo.BlendCase(**values)  # type: ignore[arg-type]


def test_run_refusal_in_the_page_order() -> None:
    usable = _Prediction(True, 313.5)
    assert bmo.run_refusal(_case(), data_driven_prediction=None, basicity_bounds_valid=False, pellet_blocked=True) == (
        bmo.DATA_DRIVEN_REFUSAL
    )
    physics = _case(fuel_rate_anchor_basis="energy_balance")
    assert "basicity bounds" in bmo.run_refusal(
        physics, data_driven_prediction=None, basicity_bounds_valid=False, pellet_blocked=True
    )
    assert "pellet" in bmo.run_refusal(
        _case(), data_driven_prediction=usable, basicity_bounds_valid=True, pellet_blocked=True
    )
    assert "two ores" in bmo.run_refusal(
        _case(ores=[_ore("ore_a", "ORE A")]),
        data_driven_prediction=usable,
        basicity_bounds_valid=True,
        pellet_blocked=False,
    )
    assert (
        bmo.run_refusal(_case(), data_driven_prediction=usable, basicity_bounds_valid=True, pellet_blocked=False)
        is None
    )


def test_run_intensive_passes_the_case_to_the_de_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    from utils.bmo import run_nonlinear_optimizer

    seen: dict[str, object] = {}

    def fake(ores: object, **kwargs: object) -> tuple[None, list[str]]:
        seen.update(kwargs, ores=ores)
        return None, ["stopped"]

    monkeypatch.setattr(bmo, "run_nonlinear_optimizer", fake)
    case = _case()
    anchor = bmo.CokeAnchor("data_driven", 313.5, {"usable": True})
    context = ("model", {"ctx": 1.0}, pd.DataFrame(), {}, [])
    result = bmo.run_intensive(
        case, fuel_context=context, reference="ref", anchor=anchor, de_cfg={"maxiter": 3}, progress_callback=None
    )
    assert result == (None, ["stopped"])
    assert set(seen) - {"ores"} <= set(inspect.signature(run_nonlinear_optimizer).parameters)
    assert seen["ores"] is case.ores and seen["model_service"] == "model" and seen["coke_correction_reference"] == "ref"
    assert seen["target_production_mt"] == 2200.0 and seen["hot_metal_target_mt"] == 2350.0
    assert seen["hot_metal_si_pct"] == 0.35 and seen["anchor_coke_rate_kg_thm"] == 313.5
    assert seen["de_cfg"] == {"maxiter": 3}


def test_case_keywords_match_the_solvers() -> None:
    import inspect

    from utils.bmo import run_lp_baseline
    from utils.bmo.constraints import check_blend_constraints

    case = _case(max_burden_qty_mt=4585.0)
    lp_parameters = set(inspect.signature(run_lp_baseline).parameters)
    assert set(case.lp_kwargs()) <= lp_parameters
    assert set(case.limits()) <= set(inspect.signature(check_blend_constraints).parameters)
    assert case.lp_kwargs()["target_production_mt"] == 2200.0  # the LP's "production" is the Fe target
    assert case.lp_kwargs()["hot_metal_target_mt"] == 2350.0
    assert case.slag_rate_cap_kg_per_thm == pytest.approx(728.5 / 2350.0 * 1000.0)
    assert case.use_charged_engine and not _case(fuel_rate_anchor_basis="energy_balance").use_charged_engine


# ---------------------------------------------------------------------------- result tables
def test_target_quantities_hold_the_fe_target() -> None:
    ores = [_ore("ore_a", "ORE A", fe=60.0, moisture=10.0), _ore("ore_b", "ORE B", fe=50.0)]
    quantities, total, warnings = bmo.target_quantities_from_shares({"ore_a": 30.0, "ore_b": 10.0}, ores, 1000.0)
    assert total == 40.0 and warnings == []
    dry_fe = quantities["ore_a"] * 0.9 * 0.6 + quantities["ore_b"] * 0.5
    assert dry_fe == pytest.approx(1000.0)
    assert quantities["ore_a"] / quantities["ore_b"] == pytest.approx(3.0)
    assert bmo.target_quantities_from_shares({"ore_a": 0.0}, ores, 1000.0)[2]


def test_manual_seed_keeps_a_zero_of_the_plant_record() -> None:
    ores = [_ore("ore_a", "ORE A"), _ore("ore_b", "ORE B"), _ore("ore_c", "ORE C")]
    snapshot = {
        "rows": [
            {"ore_id": "ore_a", "share_pct": 60.0, "quantity_mt": 50.0},
            {"ore_id": "ore_b", "share_pct": 20.0, "quantity_mt": 20.0},
            {"ore_id": "ore_c", "share_pct": 0.0, "quantity_mt": 0.0},
        ],
        "start_time": "2026-09-30T08:30:00",
        "end_time": "2026-09-30T14:30:00",
    }
    optimiser = _blend()
    optimiser.shares_pct = {"ore_a": 40.0, "ore_b": 30.0, "ore_c": 30.0}
    seed = bmo.manual_seed(snapshot, ores, optimiser, lookback_hours=6)
    assert seed.use_last_shift
    assert [row["manual_share_pct"] for row in seed.rows] == [75.0, 25.0, 0.0]
    assert seed.note == (
        "Seeded from the last 6 hours of furnace data (30 Sep, 14:00-20:00 IST). "
        "Not charged in this window, so shown at 0%: ORE C."
    )
    empty = bmo.manual_seed({"rows": []}, ores, optimiser, lookback_hours=6)
    assert not empty.use_last_shift
    assert [row["manual_share_pct"] for row in empty.rows] == [40.0, 30.0, 30.0]


def test_comparison_view_formats_and_ranks_the_options() -> None:
    ores = [_ore("ore_a", "ORE A"), _ore("ore_b", "ORE B")]
    manual = _blend(objective=4300.0, adjusted_objective_rs_per_thm=4310.0)
    lp = _blend(objective=4125.0, flux_cost_per_thm_rs=20.0)
    lp.diagnostics["hot_metal_target_mt"] = 2350.0
    view = bmo.comparison_view(
        options=[("Manual", manual, 0.35), ("Balanced Optimizer", lp, 0.35)],
        optimizer_candidates=[("Balanced Optimizer", lp, 0.35)],
        manual_blend=manual,
        compare_ores=ores,
        charge_mass_mt=26.4,
        shift_baseline={"daily_production_mt": 2200.0},
        rows_by_ore={"ore_a": {"quantity_mt": 50.0}, "ore_b": {"quantity_mt": 20.0}},
        use_last_shift=True,
    )
    assert view.labels == ["Manual", "Balanced Optimizer"]
    assert [h.total_rs_per_thm for h in view.headlines] == [4310.0, 4145.0]
    manual_tile, lp_tile = view.headlines
    assert manual_tile.is_manual and manual_tile.production_mt == 2200.0 and manual_tile.production_change_pct is None
    assert lp_tile.production_delta_mt == pytest.approx(150.0)
    assert lp_tile.production_change_pct == pytest.approx(150.0 / 2200.0 * 100.0)
    assert view.saving == ("Balanced Optimizer", pytest.approx(165.0))
    cost = view.groups["💰 Cost"]
    assert cost.rows[0] == {"Outcome": "Total Cost (Rs/THM)", "Manual": "4,310", "Balanced Optimizer": "4,145"}
    assert cost.best(0) == {"Balanced Optimizer"}
    charging = view.groups["🚚 Charging"]
    delta_row = next(row for row in charging.rows if row["Outcome"] == "Δ Production vs last shift (MT/day)")
    assert delta_row == {"Outcome": "Δ Production vs last shift (MT/day)", "Manual": "—", "Balanced Optimizer": "+150"}
    assert list(view.mix.columns) == [
        "Ore",
        "Manual (%)",
        "Manual actual (t/day)",
        "Balanced Optimizer (%)",
        "Balanced Optimizer (t/day)",
    ]
    assert view.mix["Manual actual (t/day)"].tolist() == [1200, 480]  # the hourly record x 24
    assert set(view.groups) == {"💰 Cost", "🔥 Fuel", "🌋 Slag", "🚚 Charging"}


def test_de_exploration_lists_the_cheapest_distinct_candidates() -> None:
    candidates = [
        {"total_cost_rs_per_thm": 4100.4, "slag_basicity": 1.10, "slag_mt": 700.0, "feasible": True,
         "shares_pct": {"ore_a": 60.0, "ore_b": 40.0}, "flux_mt": {"dolomite": 5.0}},
        {"total_cost_rs_per_thm": 4100.3, "slag_basicity": 1.10, "slag_mt": 700.0, "feasible": True,
         "shares_pct": {"ore_a": 60.0, "ore_b": 40.0}, "flux_mt": {"dolomite": 5.0}},
        {"total_cost_rs_per_thm": 4000.0, "slag_basicity": 1.05, "slag_mt": 710.0, "feasible": False,
         "shares_pct": {"ore_a": 70.0, "ore_b": 30.0}, "flux_mt": {}},
        {"total_cost_rs_per_thm": None, "slag_basicity": 1.0, "slag_mt": 1.0},
        "not a record",
    ]  # fmt: skip
    exploration = bmo.de_exploration(candidates, [_ore("ore_a", "ORE A")])
    assert exploration is not None and exploration.has_shares
    assert len(exploration.cloud) == 3
    table = exploration.table
    assert table["Rank"].tolist() == [1, 2]
    assert table["Total Cost (Rs/THM)"].tolist() == [4000.0, 4100.0]
    assert table["ORE A (%)"].tolist() == [70.0, 60.0]
    assert table["ORE_B (%)"].tolist() == [30.0, 40.0]
    assert table["Dolomite (MT)"].tolist() == [0.0, 5.0]
    assert table["Feasible"].tolist() == [False, True]
    assert bmo.de_exploration([], None) is None
    assert bmo.de_exploration([{"slag_mt": 1.0}], None) is None


def test_live_si_tile_uses_the_first_usable_horizon() -> None:
    record = {
        "status": "warning",
        "reasons": ["stale sample"],
        "issued_at": "2026-09-30T16:40:00+00:00",
        "last_sample_si_pct": 0.54,
        "model_version": "v3",
        "horizons": [
            {"horizon_minutes": 0, "si_pct": None, "status": "missing"},
            {"horizon_minutes": 120, "si_pct": 0.3465, "status": "ok", "target_start": "2026-09-30T18:30:00+00:00"},
        ],
    }
    tile = bmo.live_si_result_tile(record)
    assert tile is not None
    assert tile["si_pct"] == 0.3465 and tile["metric_label"] == "HM Si forecast (+2 h)"
    assert tile["warnings"] == ["stale sample"] and tile["model_version"] == "v3"
    assert bmo.live_si_result_tile(None) is None
    assert bmo.live_si_result_tile({"horizons": []}) is None


def test_effective_pci_prefers_the_override() -> None:
    assert bmo.effective_pci_kg_thm({"coal_rate_actual_value": 180.0}, None) == 180.0
    assert bmo.effective_pci_kg_thm({"coal_rate_actual_value": 180.0}, 150.0) == 150.0
    assert bmo.effective_pci_kg_thm({}, None) == 0.0


def test_control_basis_names_why_it_cannot_run() -> None:
    assert "Live blast" in (bmo.control_basis(_blend(), snapshot={}, live_pci=0.0).unavailable or "")
    no_rates = bmo.control_basis(_blend(), snapshot={"hot_blast_vol_nm3h": 113000.0}, live_pci=180.0)
    assert "Fuel-rate estimate" in (no_rates.unavailable or "")
    blend = _blend(fuel_rate_estimate={"pci_rate_kg_thm": 160.0, "coke_rate_kg_thm": 310.0})
    basis = bmo.control_basis(blend, snapshot={"hot_blast_vol_nm3h": 113000.0}, live_pci=180.0)
    assert basis.unavailable is None and basis.pci_now == 180.0
    assert basis.pci_note is not None and "180 kg/THM" in basis.pci_note
    box = bmo.control_basis(blend, snapshot={"hot_blast_vol_nm3h": 113000.0}, live_pci=0.0)
    assert box.pci_now == 160.0 and box.pci_note is None


# ---------------------------------------------------------------------------- snapshots
def test_take_snapshot_saves_what_capture_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import utils.bmo.snapshot as snapshot_module

    monkeypatch.setattr(snapshot_module, "_git_provenance", lambda: {})
    state = {"bmo_lp_result": _blend(), "bmo_target_production_mt": 2350.0}
    snapshot, path = bmo.take_snapshot(
        copy.deepcopy(state), prefix="bmo_", page_vars={}, label="golden", directory=tmp_path
    )
    assert path.parent == tmp_path and path.is_file()
    assert snapshot["label"] == "golden" and snapshot["id"]
    assert "lp_result" in snapshot["results"]
    assert bmo.finalize_run(state, "bmo_", {}) is False  # no rerun record: nothing to pin, never raises


def test_replay_si_history_reports_a_source_failure() -> None:
    def broken() -> object:
        raise ConnectionError("no feeds")

    history = bmo.replay_si_history(service=object(), make_source=broken, history_end_at="2026-10-01T12:00:00+05:30")
    assert history["error"] == "no feeds"
    assert all(isinstance(history[key], pd.DataFrame) for key in ("forecasts", "actuals", "scored"))
