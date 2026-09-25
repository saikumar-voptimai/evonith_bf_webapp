from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import mock_open

import pandas as pd
import pytest
import yaml

from data.bmo.basicity_defaults import derive_basicity_bounds_from_static_dataset
from data.bmo.ore_editor_preferences import (
    apply_dust_preferences,
    apply_fuel_ash_preferences,
    apply_flux_preferences,
    apply_model_input_preferences,
    apply_ore_editor_preferences,
    build_dust_preferences,
    build_fuel_ash_preferences,
    build_fuel_price_preferences,
    build_flux_preferences,
    build_model_input_preferences,
    build_ore_editor_preferences,
    load_ore_editor_preferences,
    save_dust_preferences,
    save_fuel_ash_preferences,
    save_fuel_price_preferences,
    save_flux_preferences,
    save_model_input_preferences,
    save_ore_editor_preferences,
)
from ui.bmo.components import build_blend_table_df
from ui.bmo import components
from utils.bmo.types import BlendEvaluation, OreChemistry, OreInput


def test_data_and_model_sources_share_the_full_page_rerun_scope() -> None:
    page_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "custom_pages"
        / "9_Blend_Optimizer.py"
    )
    tree = ast.parse(page_path.read_text(encoding="utf-8"))
    renderer = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_render_static_dataset_bar"
    )

    assert not any(
        isinstance(decorator, ast.Name) and decorator.id == "fragment"
        for decorator in renderer.decorator_list
    )

    callbacks: dict[str, str] = {}
    for call in (node for node in ast.walk(renderer) if isinstance(node, ast.Call)):
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "st"
            and call.func.attr in {"segmented_control", "toggle"}
        ):
            continue
        on_change = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "on_change"),
            None,
        )
        if isinstance(on_change, ast.Name):
            callbacks[call.func.attr] = on_change.id

    assert callbacks == {
        "segmented_control": "_clear_bmo_results",
        "toggle": "_clear_bmo_results",
    }


def test_operator_model_names_and_data_driven_default() -> None:
    root = Path(__file__).resolve().parents[1]
    page_source = (root / "src/custom_pages/9_Blend_Optimizer.py").read_text(
        encoding="utf-8"
    )
    component_source = Path(components.__file__).read_text(encoding="utf-8")
    tree = ast.parse(page_source)
    label_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_COKE_ANCHOR_LABELS"
            for target in node.targets
        )
    )
    labels = ast.literal_eval(label_assignment.value)
    config = yaml.safe_load(
        (root / "src/config/setting_bmo.yml").read_text(encoding="utf-8")
    )["bmo"]

    assert labels == {
        "Physics-Driven": "energy_balance",
        "Data-Driven": "data_driven",
    }
    assert config["fuel_rate_anchor_basis"] == "data_driven"
    assert "XGBoost" not in page_source
    assert "XGBoost" not in component_source
    assert "Run Non-linear Model" in page_source
    assert '"Non-linear Result"' in page_source


def test_model_input_alignment_step_and_diagnostics_defaults() -> None:
    page_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "custom_pages"
        / "9_Blend_Optimizer.py"
    )
    tree = ast.parse(page_path.read_text(encoding="utf-8"))

    max_slag_call = next(
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "number_input"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "Max Slag Rate (kg/THM)"
    )
    assert not any(keyword.arg == "help" for keyword in max_slag_call.keywords)
    assert any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "layout_col4"
        and call.func.attr == "caption"
        for call in ast.walk(tree)
    )

    transition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_transition_ladder"
    )
    step_call = next(
        call
        for call in ast.walk(transition)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "number_input"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "Max share change per step (%)"
    )
    step_default = next(
        keyword.value for keyword in step_call.keywords if keyword.arg == "value"
    )
    assert isinstance(step_default, ast.Constant)
    assert step_default.value == 2.0

    diagnostics = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_data_diagnostics"
    )
    diagnostics_expander = next(
        call
        for call in ast.walk(diagnostics)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "expander"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "Data Diagnostics"
    )
    expanded = next(
        keyword.value
        for keyword in diagnostics_expander.keywords
        if keyword.arg == "expanded"
    )
    assert isinstance(expanded, ast.Constant)
    assert expanded.value is False


def test_comparison_uses_currently_applied_ore_prices() -> None:
    page_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "custom_pages"
        / "9_Blend_Optimizer.py"
    )
    source = page_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_selected_ores_from_editor"
    )
    price_keyword = next(
        keyword
        for call in ast.walk(selector)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if keyword.arg == "price_rs_per_mt"
    )
    assert isinstance(price_keyword.value, ast.Call)
    assert isinstance(price_keyword.value.args[0], ast.Subscript)
    assert price_keyword.value.args[0].slice.value == "price_rs_per_mt"

    comparison = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_blend_comparison"
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "compare_ores"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "selected_ores"
        for node in ast.walk(comparison)
    )
    manual_evaluation = next(
        call
        for call in ast.walk(comparison)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "evaluate_blend_with_fuel_prediction"
    )
    ores_keyword = next(
        keyword for keyword in manual_evaluation.keywords if keyword.arg == "ores"
    )
    assert isinstance(ores_keyword.value, ast.Name)
    assert ores_keyword.value.id == "compare_ores"
    assert any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "currently applied ore prices from Ore Selection" in node.value
        for node in ast.walk(comparison)
    )


def _ore(
    ore_id: str,
    name: str,
    *,
    price: float = 100.0,
    stock: float = 1000.0,
) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=name,
        stock_mt=stock,
        price_rs_per_mt=price,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=60.0),
    )


def _blend() -> BlendEvaluation:
    return BlendEvaluation(
        quantities_mt={"ore_a": 100.0, "ore_b": 50.0},
        shares_pct={"ore_a": 66.67, "ore_b": 33.33},
        total_qty_mt=150.0,
        ore_cost_total_rs=12500.0,
        ore_cost_per_thm_rs=125.0,
        fuel_cost_per_thm_rs=4000.0,
        objective_rs_per_thm=4125.0,
        fe_t_pct=60.0,
        effective_fe_pct=60.0,
        fe_production_mt=75.0,
        slag_pct=20.0,
        slag_mt=30.0,
        feasible=True,
        violations=[],
        diagnostics={
            "dry_weight_mt_by_ore": {"ore_a": 90.0, "ore_b": 45.0},
            "fe_contribution_mt_by_ore": {"ore_a": 50.0, "ore_b": 0.0},
            "slag_contribution_mt_by_ore": {"ore_a": 25.0, "ore_b": 5.0},
            "total_dry_qty_mt": 135.0,
            "hot_metal_target_mt": 100.0,
        },
    )


def test_ore_editor_keeps_share_bounds_beside_price(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_data_editor(frame: pd.DataFrame, **kwargs):
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(components.st, "data_editor", fake_data_editor)
    editor_df = components.build_ore_editor_df(
        [_ore("ore_a", "ORE A")], default_selected_ids=["ore_a"]
    )

    returned = components.render_ore_editor(editor_df)

    assert returned is editor_df
    column_order = tuple(captured["column_order"])
    assert column_order[:6] == (
        "selected",
        "ore_name",
        "stock_mt",
        "price_rs_per_mt",
        "min_share_pct",
        "max_share_pct",
    )
    assert column_order.index("max_share_pct") < column_order.index("moisture_pct")
    assert "mn_basis" not in column_order
    assert "ti_basis" not in column_order
    column_config = captured["column_config"]
    for field in set(column_order) - {"selected", "ore_name"}:
        type_config = column_config[field]["type_config"]
        assert type_config["step"] == pytest.approx(0.01)
        assert type_config["format"] == "%.2f"


def test_ore_editor_preferences_persist_operator_defaults_but_not_stock_or_chemistry() -> None:
    edited = pd.DataFrame(
        [
            {
                "selected": True,
                "ore_id": "ore_a",
                "stock_mt": 9999.0,
                "price_rs_per_mt": 7250.0,
                "min_share_pct": 5.0,
                "max_share_pct": 30.0,
                "moisture_pct": 2.5,
                "fe_t_pct": 61.2,
                "sio2_pct": 4.5,
                "al2o3_pct": 2.1,
                "cao_pct": 0.2,
                "mgo_pct": 0.1,
                "mno_pct": 0.0,
                "tio2_pct": 0.05,
            },
            {
                "selected": False,
                "ore_id": "ore_b",
                "stock_mt": 1234.0,
                "price_rs_per_mt": 8100.0,
                "min_share_pct": 0.0,
                "max_share_pct": 20.0,
                "moisture_pct": 1.0,
                "fe_t_pct": 63.0,
                "sio2_pct": 3.9,
                "al2o3_pct": 1.8,
                "cao_pct": 0.1,
                "mgo_pct": 0.1,
                "mno_pct": 0.0,
                "tio2_pct": 0.04,
            },
        ]
    )

    prefs = build_ore_editor_preferences(edited)

    assert prefs["ore_editor"]["selected_ore_ids"] == ["ore_a"]
    assert prefs["ore_editor"]["rows"]["ore_a"]["price_rs_per_mt"] == 7250.0
    assert "stock_mt" not in prefs["ore_editor"]["rows"]["ore_a"]
    assert "moisture_pct" not in prefs["ore_editor"]["rows"]["ore_a"]
    assert "fe_t_pct" not in prefs["ore_editor"]["rows"]["ore_a"]
    assert "sio2_pct" not in prefs["ore_editor"]["rows"]["ore_a"]


def test_ore_editor_preferences_apply_only_planning_fields() -> None:
    fresh = pd.DataFrame(
        [
            {
                "selected": False,
                "ore_id": "ore_a",
                "stock_mt": 500.0,
                "price_rs_per_mt": 100.0,
                "min_share_pct": 0.0,
                "max_share_pct": 100.0,
                "moisture_pct": 0.0,
                "fe_t_pct": 55.0,
                "sio2_pct": 5.0,
            }
        ]
    )
    prefs = {
        "ore_editor": {
            "selected_ore_ids": ["ore_a"],
            "rows": {
                "ore_a": {
                    "stock_mt": 9999.0,
                    "price_rs_per_mt": 7000.0,
                    "min_share_pct": 10.0,
                    "max_share_pct": 40.0,
                    "moisture_pct": 2.2,
                    "fe_t_pct": 61.5,
                    "sio2_pct": 4.4,
                },
                "unknown_ore": {"price_rs_per_mt": 1.0},
            },
        }
    }

    applied = apply_ore_editor_preferences(fresh, prefs)

    row = applied.iloc[0]
    assert bool(row["selected"]) is True
    assert row["stock_mt"] == 500.0
    assert row["price_rs_per_mt"] == 7000.0
    assert row["min_share_pct"] == 10.0
    assert row["max_share_pct"] == 40.0
    assert row["moisture_pct"] == 0.0
    assert row["fe_t_pct"] == 55.0
    assert row["sio2_pct"] == 5.0


def test_ore_editor_preferences_save_writes_yaml(monkeypatch) -> None:
    edited = pd.DataFrame(
        [
            {
                "selected": True,
                "ore_id": "ore_a",
                "price_rs_per_mt": 7250.0,
                "min_share_pct": 5.0,
                "max_share_pct": 30.0,
                "moisture_pct": 2.5,
                "fe_t_pct": 61.2,
            }
        ]
    )
    m = mock_open()
    monkeypatch.setattr("builtins.open", m)

    saved_path = save_ore_editor_preferences(Path("bmo_operator_inputs.yml"), edited)
    written = "".join(call.args[0] for call in m().write.call_args_list)
    loaded = yaml.safe_load(written)

    assert saved_path == Path("bmo_operator_inputs.yml")
    assert loaded["ore_editor"]["selected_ore_ids"] == ["ore_a"]
    assert loaded["ore_editor"]["rows"]["ore_a"]["price_rs_per_mt"] == 7250.0
    assert "fe_t_pct" not in loaded["ore_editor"]["rows"]["ore_a"]


def test_ore_editor_preferences_load_reads_yaml(monkeypatch) -> None:
    m = mock_open(
        read_data=(
            "ore_editor:\n"
            "  selected_ore_ids:\n"
            "    - ore_a\n"
            "  rows:\n"
            "    ore_a:\n"
            "      price_rs_per_mt: 7250.0\n"
        )
    )
    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr("builtins.open", m)

    loaded = load_ore_editor_preferences(Path("bmo_operator_inputs.yml"))

    assert loaded["ore_editor"]["selected_ore_ids"] == ["ore_a"]
    assert loaded["ore_editor"]["rows"]["ore_a"]["price_rs_per_mt"] == 7250.0


def test_model_input_preferences_persist_the_whole_slag_window() -> None:
    """Every operator-set slag limit persists; plant targets do not.

    Target HM and the absolute slag tonnage are per-run planning decisions, so
    they stay out. The slag window (basicity, T basicity, rate, Al2O3, MgO,
    MgO/Al2O3) is a standing spec the operator should not retype each session.
    """

    prefs = build_model_input_preferences(
        {
            "target_production_mt": 2350.0,
            "target_slag_qty_mt": 750.0,
            "target_slag_basicity_min": 1.02,
            "target_slag_basicity_max": 1.14,
            "target_slag_t_basicity_min": 1.24,
            "target_slag_t_basicity_max": 1.40,
            "target_slag_rate_kg_per_thm": 290.0,
            "target_slag_al2o3_max_pct": 20.0,
            "target_slag_mgo_min_pct": 7.0,
            "target_slag_mgo_al2o3_ratio_min": 0.36,
            "max_charges_per_hour": 7.5,
            "charge_mass_mt": 30.1,
            # Not persisted: charging is always 24 h, and nut-coke tonnage is
            # derived from its rate rather than entered.
            "charging_hours_per_day": 24.0,
            "nut_coke_reserved_mt": 160.0,
        }
    )

    assert prefs == {
        "model_inputs": {
            "target_slag_basicity_min": 1.02,
            "target_slag_basicity_max": 1.14,
            "target_slag_t_basicity_min": 1.24,
            "target_slag_t_basicity_max": 1.40,
            "target_slag_rate_kg_per_thm": 290.0,
            "target_slag_al2o3_max_pct": 20.0,
            "target_slag_mgo_min_pct": 7.0,
            "target_slag_mgo_al2o3_ratio_min": 0.36,
            "max_charges_per_hour": 7.5,
            "charge_mass_mt": 30.1,
        }
    }
    assert "target_production_mt" not in prefs["model_inputs"]
    assert "target_slag_qty_mt" not in prefs["model_inputs"]
    assert "charging_hours_per_day" not in prefs["model_inputs"]
    assert "nut_coke_reserved_mt" not in prefs["model_inputs"]


def test_model_input_preferences_override_static_defaults() -> None:
    defaults = {
        "target_slag_basicity_min": 1.03,
        "target_slag_basicity_max": 1.16,
        "target_slag_t_basicity_min": 1.25,
        "target_slag_t_basicity_max": 1.41,
    }
    prefs = {
        "model_inputs": {
            "target_slag_basicity_min": 1.05,
            "target_slag_t_basicity_max": 1.38,
        }
    }

    applied = apply_model_input_preferences(defaults, prefs)

    assert applied["target_slag_basicity_min"] == 1.05
    assert applied["target_slag_basicity_max"] == 1.16
    assert applied["target_slag_t_basicity_min"] == 1.25
    # A saved T-basicity bound now wins over the static-dataset default, the same
    # way a saved CaO/SiO2 bound already did.
    assert applied["target_slag_t_basicity_max"] == 1.38


def test_model_input_save_preserves_ore_preferences(monkeypatch) -> None:
    m = mock_open(
        read_data=(
            "ore_editor:\n"
            "  selected_ore_ids:\n"
            "    - ore_a\n"
            "  rows:\n"
            "    ore_a:\n"
            "      price_rs_per_mt: 7000.0\n"
        )
    )
    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr("builtins.open", m)

    save_model_input_preferences(
        Path("bmo_operator_inputs.yml"),
        {
            "target_slag_basicity_min": 1.02,
            "target_slag_basicity_max": 1.14,
            "target_slag_t_basicity_min": 1.24,
            "target_slag_t_basicity_max": 1.40,
        },
    )
    written = "".join(call.args[0] for call in m().write.call_args_list)
    loaded = yaml.safe_load(written)

    assert loaded["ore_editor"]["rows"]["ore_a"]["price_rs_per_mt"] == 7000.0
    assert loaded["model_inputs"]["target_slag_basicity_min"] == 1.02
    assert loaded["model_inputs"]["target_slag_t_basicity_min"] == 1.24
    assert loaded["model_inputs"]["target_slag_t_basicity_max"] == 1.40


def test_ore_input_save_preserves_model_preferences(monkeypatch) -> None:
    m = mock_open(
        read_data=(
            "model_inputs:\n"
            "  target_slag_basicity_min: 1.02\n"
            "  target_slag_basicity_max: 1.14\n"
            "  target_slag_t_basicity_min: 1.24\n"
            "  target_slag_t_basicity_max: 1.40\n"
        )
    )
    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr("builtins.open", m)

    save_ore_editor_preferences(
        Path("bmo_operator_inputs.yml"),
        pd.DataFrame(
            [
                {
                    "selected": True,
                    "ore_id": "ore_b",
                    "price_rs_per_mt": 7250.0,
                    "min_share_pct": 5.0,
                    "max_share_pct": 30.0,
                }
            ]
        ),
    )
    written = "".join(call.args[0] for call in m().write.call_args_list)
    loaded = yaml.safe_load(written)

    assert loaded["model_inputs"]["target_slag_t_basicity_max"] == 1.40
    assert loaded["ore_editor"]["selected_ore_ids"] == ["ore_b"]


def test_static_dataset_basicity_defaults_are_direct_recent_p10_p90(
    monkeypatch,
) -> None:
    header = pd.DataFrame(
        columns=["time", "SLAG_PCT_CAO", "SLAG_PCT_MGO", "SLAG_PCT_SIO2"]
    )
    data = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-01 00:00", "2026-06-01 00:00", "2026-06-02 00:00"]
            ),
            "SLAG_PCT_CAO": [999.0, 30.0, 40.0],
            "SLAG_PCT_MGO": [999.0, 6.0, 8.0],
            "SLAG_PCT_SIO2": [999.0, 30.0, 32.0],
        }
    )

    def fake_read_csv(_path, *args, **kwargs):
        if kwargs.get("nrows") == 0:
            return header
        return data[list(kwargs["usecols"])]

    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr("data.bmo.basicity_defaults.pd.read_csv", fake_read_csv)

    defaults = derive_basicity_bounds_from_static_dataset(
        Path("furnace_dataset.csv"), window_days=30
    )

    assert defaults["target_slag_basicity_min"] == pytest.approx(1.025)
    assert defaults["target_slag_basicity_max"] == pytest.approx(1.225)
    assert defaults["target_slag_t_basicity_min"] == pytest.approx(1.23)
    assert defaults["target_slag_t_basicity_max"] == pytest.approx(1.47)


def test_blend_table_includes_slag_per_fe_ratio() -> None:
    df = build_blend_table_df(
        _blend(), [_ore("ore_a", "ORE A"), _ore("ore_b", "ORE B")]
    )

    ore_a = df[df["ore_name"] == "ORE A"].iloc[0]
    ore_b = df[df["ore_name"] == "ORE B"].iloc[0]

    assert ore_a["slag_per_fe"] == pytest.approx(0.5)
    assert ore_b["slag_per_fe"] == pytest.approx(0.0)


def test_blend_table_calculates_wet_kg_per_charge() -> None:
    blend = _blend()
    blend.quantities_mt["ore_a"] = 2400.0
    blend.diagnostics.update(
        {
            "total_burden_qty_mt": 6.62 * 24.0 * 26.4,
            "fuel_rate_estimate": {"nut_coke_rate_kg_thm": 0.0},
        }
    )

    df = build_blend_table_df(
        blend,
        [_ore("ore_a", "Sinter")],
    )

    assert df.iloc[0]["kg_per_charge"] == pytest.approx(
        (2400.0 / (6.62 * 24.0)) * 1000.0
    )


class _FakeLayout:
    """Context-manager stand-ins for st.container / st.expander / st.tabs.

    render_blend_metrics groups its tiles into bordered containers and puts the
    reference diagrams behind an expander. A double without these raises
    AttributeError on layout rather than on anything the test is asserting.
    """

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeLayoutMixin:
    def container(self, **_kwargs):
        return _FakeLayout()

    def expander(self, *_args, **_kwargs):
        return _FakeLayout()

    def divider(self):
        pass

    def image(self, *_args, **_kwargs):
        pass


def test_main_metrics_hide_hot_metal_removal_section(monkeypatch) -> None:
    captured = {"markdown": [], "metrics": [], "captions": []}

    class FakeColumn:
        def metric(self, label, value, **_kwargs):
            captured["metrics"].append(str(label))

        def markdown(self, text, **_kwargs):
            captured["markdown"].append(str(text))

    class FakeStreamlit(_FakeLayoutMixin):
        def markdown(self, text, **_kwargs):
            captured["markdown"].append(str(text))

        def columns(self, count):
            return [FakeColumn() for _ in range(int(count))]

        def caption(self, text):
            captured["captions"].append(str(text))

        def warning(self, text):
            captured.setdefault("warnings", []).append(str(text))

    blend = _blend()
    blend.diagnostics.update(
        {
            "slag_balance_enabled": True,
            "hm_reduction_sio2_mt": 11.08,
            "hm_reduction_mno_mt": 4.26,
            "hm_reduction_tio2_mt": 1.12,
            "hm_reduction_alkali_mt": 0.67,
        }
    )

    monkeypatch.setattr(components, "st", FakeStreamlit())
    components.render_blend_metrics("LP Baseline Result", blend)

    rendered = "\n".join(
        [*captured["markdown"], *captured["metrics"], *captured["captions"]]
    )
    assert "Removed by Hot Metal" not in rendered
    assert "SiO2 -> HM Si" not in rendered
    assert "MnO -> HM Mn" not in rendered
    assert "TiO2 -> HM Ti" not in rendered
    assert "Alkali -> Gas" not in rendered


def test_main_metrics_show_production_and_requested_charging_values(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeColumn:
        def metric(self, label, value, **_kwargs):
            captured[str(label)] = str(value)

        def markdown(self, _text, **_kwargs):
            pass

    class FakeStreamlit(_FakeLayoutMixin):
        def markdown(self, _text, **_kwargs):
            pass

        def columns(self, count):
            return [FakeColumn() for _ in range(int(count))]

        def caption(self, _text):
            pass

        def warning(self, _text):
            pass

    blend = _blend()
    blend.diagnostics.update(
        {
            "total_burden_qty_mt": 193.0,
            "fuel_rate_estimate": {
                "coke_rate_kg_thm": 400.0,
                "nut_coke_rate_kg_thm": 70.0,
                "pci_rate_kg_thm": 150.0,
                "total_fuel_rate_kg_thm": 620.0,
            },
            "full_slag_balance": {"actual_pig_iron_mt": 100.0},
        }
    )

    monkeypatch.setattr(components, "st", FakeStreamlit())
    components.render_blend_metrics("LP Baseline Result", blend)

    # Labels carry their unit now and the tiles are grouped, so these match on
    # meaning rather than on the exact wording of a heading.
    def tile(token: str) -> str:
        if token in captured:
            return captured[token]
        matches = [key for key in captured if token.lower() in key.lower()]
        assert matches, f"no tile matching {token!r}; captured: {sorted(captured)}"
        # An ambiguous token would let this assert against the wrong tile —
        # "Coke (MT)" is a substring of "Nut coke (MT)".
        assert len(matches) == 1, f"{token!r} matched several tiles: {matches}"
        return captured[matches[0]]

    assert tile("Production") == "100.0"
    assert tile("Coke (MT)") == "40.0"
    assert tile("PCI (MT)") == "15.0"
    assert tile("Hot metal per charge") == "13.200"
    assert "Planning HM/charge (MT)" not in captured
    hidden = {
        "Fe Produced (MT)",
        "Dry Qty (MT)",
        "IBRM + Flux (MT)",
        "Total Charge Mix (MT)",
        "Charge Mix (MT/hr)",
    }
    assert hidden.isdisjoint(captured)
    # "Slag T Basicity" was hidden alongside the five above, but it does not
    # belong with them: those are informational, whereas T Basicity is a HARD
    # optimizer constraint driven by the Min/Max T Basicity inputs. Hiding it
    # left the optimizer enforcing a limit the operator could not see - and IB4
    # took its display slot, so a tile expected to read ~1.31 read ~0.85.
    assert any("T-Basicity" in key or "T Basicity" in key for key in captured), (
        f"T Basicity tile is missing; captured: {sorted(captured)}"
    )
    assert any("IB4" in key for key in captured)


def _flux_df():
    return pd.DataFrame(
        [
            {
                "flux_id": "dolomite",
                "flux_name": "Dolomite",
                "optimizable": True,
                "price_rs_per_mt": 3100.0,
                "stock_mt": 450.0,
                "cao_pct": 30.2,
                "sio2_pct": 1.7,
            },
            {
                "flux_id": "quartz",
                "flux_name": "Quartz",
                "optimizable": True,
                "price_rs_per_mt": 2100.0,
                "stock_mt": 600.0,
                "cao_pct": 0.0,
                "sio2_pct": 96.5,
            },
        ]
    )


def _fuel_ash_df():
    return pd.DataFrame(
        [
            {
                "fuel_id": "coke",
                "fuel_name": "Coke",
                "enabled": True,
                "rate_kg_per_thm": 340.0,
                "price_rs_per_mt": 28000.0,
                "moisture_pct": 0.4,
                "vm_pct": 0.9,
                "ash_pct": 11.5,
                "sio2_pct": 55.0,
            }
        ]
    )


def test_apply_fuel_prices_changes_only_the_three_price_cells() -> None:
    original = pd.DataFrame(
        [
            {
                "fuel_id": "coke",
                "price_rs_per_mt": 28000.0,
                "rate_kg_per_thm": 340.0,
                "ash_pct": 11.5,
            },
            {
                "fuel_id": "nut_coke",
                "price_rs_per_mt": 24000.0,
                "rate_kg_per_thm": 70.0,
                "ash_pct": 12.0,
            },
            {
                "fuel_id": "pci",
                "price_rs_per_mt": 18000.0,
                "rate_kg_per_thm": 150.0,
                "ash_pct": 9.0,
            },
            {
                "fuel_id": "other",
                "price_rs_per_mt": 1.0,
                "rate_kg_per_thm": 2.0,
                "ash_pct": 3.0,
            },
        ]
    )

    updated = components.apply_fuel_prices(
        original,
        {
            "coke": 31000.0,
            "nut_coke": 22500.0,
            "pci": 19500.0,
            "other": 999.0,
        },
    )

    assert updated["price_rs_per_mt"].tolist() == [
        31000.0,
        22500.0,
        19500.0,
        1.0,
    ]
    pd.testing.assert_frame_equal(
        updated.drop(columns="price_rs_per_mt"),
        original.drop(columns="price_rs_per_mt"),
    )
    assert original["price_rs_per_mt"].tolist() == [
        28000.0,
        24000.0,
        18000.0,
        1.0,
    ]


def test_fuel_price_inputs_are_vertical_and_keyed_for_snapshots(monkeypatch) -> None:
    captured: dict[str, object] = {"inputs": []}
    entered = {
        "bmo_fuel_price_coke_rs_per_mt": 31000.0,
        "bmo_fuel_price_nut_coke_rs_per_mt": 22500.0,
        "bmo_fuel_price_pci_rs_per_mt": 19500.0,
    }

    class FakeStreamlit:
        def caption(self, text):
            captured["caption"] = text

        def number_input(self, label, **kwargs):
            captured["inputs"].append((label, kwargs))
            return entered[kwargs["key"]]

    original = pd.DataFrame(
        [
            {
                "fuel_id": "coke",
                "price_rs_per_mt": 28000.0,
                "rate_kg_per_thm": 340.0,
            },
            {
                "fuel_id": "nut_coke",
                "price_rs_per_mt": 24000.0,
                "rate_kg_per_thm": 70.0,
            },
            {
                "fuel_id": "pci",
                "price_rs_per_mt": 18000.0,
                "rate_kg_per_thm": 150.0,
            },
        ]
    )
    monkeypatch.setattr(components, "st", FakeStreamlit())

    updated = components.render_fuel_price_inputs(original)

    labels_and_kwargs = captured["inputs"]
    assert [label for label, _kwargs in labels_and_kwargs] == [
        "Coke (Rs/MT)",
        "Nut coke (Rs/MT)",
        "PCI (Rs/MT)",
    ]
    assert {kwargs["key"] for _label, kwargs in labels_and_kwargs} == set(entered)
    assert updated["price_rs_per_mt"].tolist() == [31000.0, 22500.0, 19500.0]
    assert (
        updated["rate_kg_per_thm"].tolist()
        == original["rate_kg_per_thm"].tolist()
    )


def test_fuel_price_preferences_contain_prices_only() -> None:
    frame = pd.DataFrame(
        [
            {
                "fuel_id": "coke",
                "price_rs_per_mt": 31000.0,
                "rate_kg_per_thm": 340.0,
                "ash_pct": 11.5,
            },
            {
                "fuel_id": "nut_coke",
                "price_rs_per_mt": 22500.0,
                "rate_kg_per_thm": 85.0,
                "ash_pct": 12.0,
            },
            {
                "fuel_id": "pci",
                "price_rs_per_mt": 19500.0,
                "rate_kg_per_thm": 150.0,
                "ash_pct": 9.0,
            },
        ]
    )

    payload = build_fuel_price_preferences(frame)

    assert payload == {
        "fuel_ash_editor": {
            "rows": {
                "coke": {"price_rs_per_mt": 31000.0},
                "nut_coke": {"price_rs_per_mt": 22500.0},
                "pci": {"price_rs_per_mt": 19500.0},
            }
        }
    }


def test_fuel_price_save_preserves_rates_chemistry_and_other_sections(tmp_path) -> None:
    path = tmp_path / "prefs.yml"
    save_flux_preferences(path, _flux_df())
    save_fuel_ash_preferences(path, _fuel_ash_df())
    changed_price = _fuel_ash_df()
    changed_price.loc[0, "price_rs_per_mt"] = 31500.0
    changed_price.loc[0, "rate_kg_per_thm"] = 999.0
    changed_price.loc[0, "ash_pct"] = 99.0

    save_fuel_price_preferences(path, changed_price)
    loaded = load_ore_editor_preferences(path)
    coke = loaded["fuel_ash_editor"]["rows"]["coke"]

    assert coke["price_rs_per_mt"] == 31500.0
    assert coke["rate_kg_per_thm"] == 340.0
    assert coke["ash_pct"] == 11.5
    assert coke["vm_pct"] == 0.9
    assert loaded["flux_editor"]["rows"]["dolomite"]["stock_mt"] == 450.0


def _dust_df():
    return pd.DataFrame(
        [
            {
                "dust_id": "bf_gas_dust",
                "dust_name": "BF Gas Dust",
                "enabled": True,
                "wet_qty_mt": 12.0,
                "moisture_pct": 4.0,
                "sio2_pct": 8.0,
                "al2o3_pct": 3.0,
                "cao_pct": 5.0,
                "mgo_pct": 2.0,
                "fe_pct": 35.0,
                "mn_pct": 0.5,
                "p_pct": 0.1,
                "s_pct": 0.2,
                "ti_pct": 0.3,
                "zn_pct": 1.0,
                "na2o_pct": 0.4,
                "k2o_pct": 0.6,
                "caf2_pct": 0.0,
            }
        ]
    )


def test_dust_preferences_persist_and_apply_all_editable_values() -> None:
    prefs = build_dust_preferences(_dust_df())
    saved = prefs["dust_editor"]["rows"]["bf_gas_dust"]

    assert saved["enabled"] is True
    assert saved["wet_qty_mt"] == 12.0
    assert saved["moisture_pct"] == 4.0
    assert saved["fe_pct"] == 35.0
    assert saved["k2o_pct"] == 0.6

    fresh = _dust_df()
    fresh.loc[0, ["wet_qty_mt", "moisture_pct", "fe_pct"]] = [1.0, 2.0, 3.0]
    applied = apply_dust_preferences(fresh, prefs)

    assert applied.loc[0, "wet_qty_mt"] == 12.0
    assert applied.loc[0, "moisture_pct"] == 4.0
    assert applied.loc[0, "fe_pct"] == 35.0


def test_dust_save_preserves_fuel_ash_preferences(tmp_path) -> None:
    path = tmp_path / "prefs.yml"
    save_fuel_ash_preferences(path, _fuel_ash_df())

    save_dust_preferences(path, _dust_df())
    loaded = load_ore_editor_preferences(path)

    assert loaded["fuel_ash_editor"]["rows"]["coke"]["vm_pct"] == 0.9
    assert loaded["dust_editor"]["rows"]["bf_gas_dust"]["wet_qty_mt"] == 12.0


def test_fuel_ash_editor_labels_moisture_and_adds_vm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_data_editor(frame: pd.DataFrame, **kwargs):
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(components.st, "data_editor", fake_data_editor)
    editor_df = components.build_fuel_ash_editor_df(
        [
            {
                "fuel_id": "coke",
                "display_name": "Coke",
                "moisture_pct": 0.4,
                "vm_pct": 0.9,
            }
        ]
    )

    returned = components.render_fuel_ash_editor(editor_df)

    assert returned is editor_df
    assert editor_df.loc[0, "vm_pct"] == 0.9
    column_order = tuple(captured["column_order"])
    assert "vm_pct" in column_order
    for hidden in ("mn_basis", "rate_basis", "ti_basis", "chemistry_source"):
        assert hidden not in column_order
    column_config = captured["column_config"]
    assert column_config["moisture_pct"]["label"] == "Moisture (%)"
    assert "TM" in column_config["moisture_pct"]["help"]
    assert "IM" in column_config["moisture_pct"]["help"]
    assert column_config["vm_pct"]["label"] == "Ash Analysis VM (%)"
    assert "fuel_chemistry.vm" in column_config["vm_pct"]["help"]
    for field in set(column_order) - {"enabled", "fuel_name"}:
        type_config = column_config[field]["type_config"]
        assert type_config["step"] == pytest.approx(0.01)
        assert type_config["format"] == "%.2f"


def test_flux_and_dust_editors_hide_internal_columns_and_accept_two_decimals(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_data_editor(frame: pd.DataFrame, **kwargs):
        captured.update(kwargs)
        return frame

    monkeypatch.setattr(components.st, "data_editor", fake_data_editor)
    cases = (
        (
            components.render_flux_editor,
            components.build_flux_editor_df(
                [{"flux_id": "limestone", "display_name": "Limestone"}]
            ),
            {"mn_basis", "ti_basis"},
            {"enabled", "flux_name", "optimizable"},
        ),
        (
            components.render_dust_editor,
            components.build_dust_editor_df(
                [{"dust_id": "bf_gas_dust", "display_name": "BF Gas Dust"}]
            ),
            {"rate_basis", "wet_qty_mt", "source"},
            {"enabled", "dust_name"},
        ),
    )

    for render, editor_df, hidden_fields, nonnumeric_fields in cases:
        captured.clear()
        assert render(editor_df) is editor_df
        column_order = tuple(captured["column_order"])
        assert hidden_fields.isdisjoint(column_order)
        column_config = captured["column_config"]
        for field in set(column_order) - nonnumeric_fields:
            type_config = column_config[field]["type_config"]
            assert type_config["step"] == pytest.approx(0.01)
            assert type_config["format"] == "%.2f"


def test_fuel_ash_preferences_persist_and_apply_all_editable_values() -> None:
    prefs = build_fuel_ash_preferences(_fuel_ash_df())
    saved = prefs["fuel_ash_editor"]["rows"]["coke"]

    assert saved["enabled"] is True
    assert saved["moisture_pct"] == 0.4
    assert saved["vm_pct"] == 0.9
    assert saved["ash_pct"] == 11.5

    fresh = _fuel_ash_df()
    fresh.loc[0, ["rate_kg_per_thm", "moisture_pct", "vm_pct"]] = [1.0, 2.0, 3.0]
    applied = apply_fuel_ash_preferences(fresh, prefs)

    assert applied.loc[0, "rate_kg_per_thm"] == 340.0
    assert applied.loc[0, "moisture_pct"] == 0.4
    assert applied.loc[0, "vm_pct"] == 0.9


def test_fuel_ash_save_preserves_other_preferences(tmp_path) -> None:
    path = tmp_path / "prefs.yml"
    save_flux_preferences(path, _flux_df())

    save_fuel_ash_preferences(path, _fuel_ash_df())
    loaded = load_ore_editor_preferences(path)

    assert loaded["flux_editor"]["rows"]["dolomite"]["stock_mt"] == 450.0
    assert loaded["fuel_ash_editor"]["rows"]["coke"]["vm_pct"] == 0.9


def test_flux_preferences_persist_only_price_and_stock() -> None:
    prefs = build_flux_preferences(_flux_df())
    rows = prefs["flux_editor"]["rows"]
    assert rows["dolomite"] == {"price_rs_per_mt": 3100.0, "stock_mt": 450.0}
    # Chemistry / optimizable flag are config-driven, not persisted.
    assert "cao_pct" not in rows["dolomite"]
    assert "optimizable" not in rows["dolomite"]


def test_flux_preferences_apply_overlays_price_stock_only() -> None:
    # Fresh config frame with default price/stock and chemistry.
    fresh = pd.DataFrame(
        [
            {
                "flux_id": "dolomite",
                "flux_name": "Dolomite",
                "optimizable": True,
                "price_rs_per_mt": 3000.0,
                "stock_mt": 500.0,
                "cao_pct": 30.2,
                "sio2_pct": 1.7,
            }
        ]
    )
    prefs = {"flux_editor": {"rows": {"dolomite": {"price_rs_per_mt": 3100.0, "stock_mt": 450.0}}}}
    applied = apply_flux_preferences(fresh, prefs)
    row = applied.iloc[0]
    assert row["price_rs_per_mt"] == 3100.0
    assert row["stock_mt"] == 450.0
    assert row["cao_pct"] == 30.2  # chemistry untouched


def test_flux_save_preserves_ore_preferences(tmp_path) -> None:
    path = tmp_path / "prefs.yml"
    save_ore_editor_preferences(
        path,
        pd.DataFrame(
            [
                {
                    "ore_id": "ore_a",
                    "selected": True,
                    "price_rs_per_mt": 111.0,
                    "min_share_pct": 5.0,
                    "max_share_pct": 40.0,
                }
            ]
        ),
    )
    save_flux_preferences(path, _flux_df())
    loaded = load_ore_editor_preferences(path)
    # Both sections coexist.
    assert loaded["ore_editor"]["rows"]["ore_a"]["price_rs_per_mt"] == 111.0
    assert loaded["flux_editor"]["rows"]["quartz"]["price_rs_per_mt"] == 2100.0
    assert loaded["flux_editor"]["rows"]["quartz"]["stock_mt"] == 600.0
