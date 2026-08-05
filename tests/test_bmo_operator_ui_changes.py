from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open

import pandas as pd
import pytest
import yaml

from data.bmo.basicity_defaults import derive_basicity_bounds_from_static_dataset
from data.bmo.ore_editor_preferences import (
    apply_fuel_ash_preferences,
    apply_flux_preferences,
    apply_model_input_preferences,
    apply_ore_editor_preferences,
    build_fuel_ash_preferences,
    build_flux_preferences,
    build_model_input_preferences,
    build_ore_editor_preferences,
    load_ore_editor_preferences,
    save_fuel_ash_preferences,
    save_flux_preferences,
    save_model_input_preferences,
    save_ore_editor_preferences,
)
from ui.bmo.components import build_blend_table_df, build_fuel_ash_editor_df
from ui.bmo import components
from utils.bmo.types import BlendEvaluation, OreChemistry, OreInput


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


def test_model_input_preferences_persist_only_basicity_bounds() -> None:
    prefs = build_model_input_preferences(
        {
            "target_production_mt": 2350.0,
            "target_slag_qty_mt": 750.0,
            "target_slag_basicity_min": 1.02,
            "target_slag_basicity_max": 1.14,
            "target_slag_t_basicity_min": 1.24,
            "target_slag_t_basicity_max": 1.40,
        }
    )

    assert prefs == {
        "model_inputs": {
            "target_slag_basicity_min": 1.02,
            "target_slag_basicity_max": 1.14,
        }
    }


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
    assert applied["target_slag_t_basicity_max"] == 1.41


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
    assert "target_slag_t_basicity_min" not in loaded["model_inputs"]
    assert "target_slag_t_basicity_max" not in loaded["model_inputs"]


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


def test_main_metrics_show_target_hot_metal_as_production(monkeypatch) -> None:
    captured = {
        "markdown": [],
        "metrics": [],
        "metric_values": {},
        "metric_kwargs": {},
        "captions": [],
        "events": [],
    }

    class FakeColumn:
        def metric(self, label, value, **kwargs):
            label_text = str(label)
            captured["metrics"].append(label_text)
            captured["metric_values"][label_text] = str(value)
            captured["metric_kwargs"][label_text] = kwargs
            captured["events"].append(("metric", label_text))

    class FakeStreamlit:
        def markdown(self, text, **_kwargs):
            markdown_text = str(text)
            captured["markdown"].append(markdown_text)
            captured["events"].append(("markdown", markdown_text))

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
            "fuel_rate_estimate": {
                "coke_rate_kg_thm": 400.0,
                "nut_coke_rate_kg_thm": 70.0,
                "pci_rate_kg_thm": 150.0,
                "total_fuel_rate_kg_thm": 620.0,
            },
            "coke_correction_delta_kg_thm": 10.9,
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
    assert captured["metric_values"]["Production"] == "100.0 MT"
    fuel_cost_kwargs = captured["metric_kwargs"]["Fuel Cost (Rs/THM)"]
    assert fuel_cost_kwargs.get("border", False) is False
    assert fuel_cost_kwargs["delta"] == "Model Predicted"
    assert fuel_cost_kwargs["delta_color"] == "off"
    assert fuel_cost_kwargs["delta_arrow"] == "off"
    coke_rate_kwargs = captured["metric_kwargs"]["Coke Rate (kg/THM)"]
    assert coke_rate_kwargs.get("border", False) is False
    assert coke_rate_kwargs["delta"] == "Model Predicted"
    assert coke_rate_kwargs["delta_color"] == "off"
    assert coke_rate_kwargs["delta_arrow"] == "off"
    assert "physics" not in (rendered + repr(captured["metric_kwargs"])).lower()
    assert "Fe Produced (MT)" not in captured["metrics"]
    for hidden_metric in (
        "Slag T Basicity",
        "Dry Qty (MT)",
        "IBRM + Flux (MT)",
        "Total Charge Mix (MT)",
        "Charge Mix (MT/hr)",
    ):
        assert hidden_metric not in captured["metrics"]

    charging_heading = captured["events"].index(
        ("markdown", "##### Charging Requirement")
    )
    charging_metrics = [
        value
        for kind, value in captured["events"][charging_heading + 1 :]
        if kind == "metric"
    ]
    assert charging_metrics == [
        "Required Charges (/hr)",
        "Coke in Charges (MT)",
        "Nut Coke in Charges (MT)",
        "PCI in Charges (MT)",
        "Hot Metal per Charge (MT)",
    ]
    assert captured["metric_values"]["Coke in Charges (MT)"] == "40.0"
    assert captured["metric_values"]["PCI in Charges (MT)"] == "15.0"
    assert captured["metric_values"]["Hot Metal per Charge (MT)"] == "16.815"


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


def _fuel_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fuel_id": "coke",
                "fuel_name": "Coke",
                "enabled": True,
                "rate_kg_per_thm": 340.0,
                "price_rs_per_mt": 28_000.0,
                "im_pct": 0.52,
                "vm_pct": 0.94,
                "ash_pct": 11.46,
                "sio2_pct": 55.07,
                "al2o3_pct": 27.09,
                "cao_pct": 3.25,
                "mgo_pct": 1.21,
                "fe2o3_pct": 7.22,
                "tio2_pct": 1.462,
                "na2o_pct": 0.0,
                "k2o_pct": 0.0,
                "s_pct": 0.74,
                "p_pct": 0.04292,
            }
        ]
    )


def test_fuel_editor_uses_separate_im_and_vm_columns(monkeypatch) -> None:
    editor_df = build_fuel_ash_editor_df(
        [{"fuel_id": "coke", "moisture_pct": 0.37, "vm_pct": 0.94}]
    )
    assert editor_df.iloc[0]["im_pct"] == 0.37
    assert editor_df.iloc[0]["vm_pct"] == 0.94
    assert "moisture_pct" not in editor_df.columns

    captured: dict = {}

    class FakeColumnConfig:
        @staticmethod
        def NumberColumn(label, **kwargs):
            return {"label": label, **kwargs}

        @staticmethod
        def CheckboxColumn(label, **_kwargs):
            return {"label": label}

        @staticmethod
        def TextColumn(label, **_kwargs):
            return {"label": label}

    class FakeStreamlit:
        column_config = FakeColumnConfig()

        @staticmethod
        def data_editor(df, **kwargs):
            captured.update(kwargs)
            return df

    monkeypatch.setattr(components, "st", FakeStreamlit())
    components.render_fuel_ash_editor(editor_df)

    assert captured["column_config"]["im_pct"]["label"] == "Ash % IM"
    assert captured["column_config"]["vm_pct"]["label"] == "Ash % VM"
    for column in (
        "rate_kg_per_thm",
        "price_rs_per_mt",
        "im_pct",
        "vm_pct",
        "ash_pct",
        "sio2_pct",
        "al2o3_pct",
        "cao_pct",
        "mgo_pct",
        "fe2o3_pct",
        "tio2_pct",
        "na2o_pct",
        "k2o_pct",
        "s_pct",
    ):
        assert captured["column_config"][column]["step"] <= 0.01
    assert captured["column_config"]["p_pct"]["step"] < 0.01


def test_fuel_preferences_preserve_live_rate_and_restore_other_values() -> None:
    edited = _fuel_df()
    edited.loc[0, "rate_kg_per_thm"] = 999.0
    edited.loc[0, "enabled"] = False
    edited.loc[0, "vm_pct"] = 1.05

    prefs = build_fuel_ash_preferences(edited)
    saved = prefs["fuel_ash_editor"]["rows"]["coke"]
    assert "rate_kg_per_thm" not in saved
    assert saved["enabled"] is False
    assert saved["im_pct"] == 0.52
    assert saved["vm_pct"] == 1.05
    assert saved["p_pct"] == 0.04292

    fresh = _fuel_df()
    fresh.loc[0, "rate_kg_per_thm"] = 355.25
    fresh.loc[0, "vm_pct"] = 9.99
    # Even a stale rate written by an older app version must be ignored.
    prefs["fuel_ash_editor"]["rows"]["coke"]["rate_kg_per_thm"] = 111.0
    restored = apply_fuel_ash_preferences(fresh, prefs).iloc[0]
    assert bool(restored["enabled"]) is False
    assert restored["rate_kg_per_thm"] == 355.25
    assert restored["vm_pct"] == 1.05


def test_fuel_save_preserves_existing_operator_preferences(tmp_path) -> None:
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

    save_fuel_ash_preferences(path, _fuel_df())
    loaded = load_ore_editor_preferences(path)

    assert loaded["ore_editor"]["rows"]["ore_a"]["price_rs_per_mt"] == 111.0
    assert loaded["fuel_ash_editor"]["rows"]["coke"]["vm_pct"] == 0.94
    assert "rate_kg_per_thm" not in loaded["fuel_ash_editor"]["rows"]["coke"]
