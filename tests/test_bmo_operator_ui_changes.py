from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open

import pandas as pd
import pytest
import yaml

from data.bmo.ore_editor_preferences import (
    apply_ore_editor_preferences,
    build_ore_editor_preferences,
    load_ore_editor_preferences,
    save_ore_editor_preferences,
)
from ui.bmo.components import build_blend_table_df
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


def test_blend_table_includes_slag_per_fe_ratio() -> None:
    df = build_blend_table_df(
        _blend(), [_ore("ore_a", "ORE A"), _ore("ore_b", "ORE B")]
    )

    ore_a = df[df["ore_name"] == "ORE A"].iloc[0]
    ore_b = df[df["ore_name"] == "ORE B"].iloc[0]

    assert ore_a["slag_per_fe"] == pytest.approx(0.5)
    assert ore_b["slag_per_fe"] == pytest.approx(0.0)


def test_main_metrics_hide_hot_metal_removal_section(monkeypatch) -> None:
    captured = {"markdown": [], "metrics": [], "captions": []}

    class FakeColumn:
        def metric(self, label, value, **_kwargs):
            captured["metrics"].append(str(label))

    class FakeStreamlit:
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
