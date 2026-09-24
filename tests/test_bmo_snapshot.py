"""Snapshots, exact encoding, frozen plant data and replay, the store, the report, the sandbox."""

from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from utils.bmo import sandbox
from utils.bmo.replay import (
    K_LATEST,
    K_PENDING,
    K_RECORD,
    PageNamespace,
    RecordingProvider,
    ReplayProvider,
    call_key,
    finalize_run,
    frozen_config,
    page_substitutes,
)
from utils.bmo.snapshot import (
    IST,
    _decode_compact_frame,
    capture,
    compact_frame,
    decode,
    encode,
    frozen_state,
    restorable_state,
    results_state,
    summarise,
    validate,
    verify_compact,
)
from utils.bmo.snapshot_report import build_docx
from utils.bmo.snapshot_store import delete, list_snapshots, load, save
from utils.bmo.types import BlendEvaluation, ModelPrediction


def _evaluation(*, fell_back: bool = False, cost: float = 9000.0) -> BlendEvaluation:
    return BlendEvaluation(
        quantities_mt={"ore_a": 1800.0, "sinter": 1200.0},
        shares_pct={"ore_a": 60.0, "sinter": 40.0},
        total_qty_mt=3000.0,
        ore_cost_total_rs=2.0e7,
        ore_cost_per_thm_rs=7000.0,
        fuel_cost_per_thm_rs=2000.0,
        objective_rs_per_thm=cost,
        fe_t_pct=58.2,
        effective_fe_pct=57.9,
        fe_production_mt=1750.0,
        slag_pct=14.0,
        slag_mt=640.0,
        feasible=True,
        violations=[],
        slag_rate_kg_per_thm=320.5,
        slag_basicity=1.12,
        slag_t_basicity=1.41,
        slag_al2o3_pct=17.1,
        slag_mgo_pct=8.2,
        slag_mgo_al2o3_ratio=0.48,
        diagnostics={
            "de_fell_back_to_lp": fell_back,
            "fuel_rate_estimate": {
                "coke_rate_kg_thm": 395.0,
                "nut_coke_rate_kg_thm": 70.0,
                "pci_rate_kg_thm": 170.0,
                "total_fuel_rate_kg_thm": 635.0,
            },
            "fuel_rate_estimate_anchor": {"coke_rate_kg_thm": 398.0},
            "coke_correction": {
                "anchor_coke_rate_kg_thm": 398.0,
                "terms": [{"label": "Slag heat", "enabled": True, "delta_kg_thm": -3.0,
                           "x_blend": 320.5, "x_reference": 334.0}],
            },
            "coke_correction_delta_kg_thm": -3.0,
            "model_prediction": ModelPrediction(value=13507.34, model_loaded=True,
                                                scaler_loaded=True, used_fallback=False),
            "flux_cost_per_thm_rs": 100.0,
            "dry_weight_mt_by_ore": {"ore_a": 1700.0, "sinter": 1190.0},
            "fe_contribution_mt_by_ore": {"ore_a": 1000.0, "sinter": 750.0},
            "slag_contribution_mt_by_ore": {"ore_a": 400.0, "sinter": 240.0},
            "charge_count": 150.0,
            "total_burden_qty_mt": 3000.0,
            "slag_basicity_sio2_mt": 200.0, "slag_basicity_cao_mt": 224.0,
            "ore_slag_mt": 500.0, "flux_slag_mt": 60.0, "fuel_ash_slag_mt": 80.0,
            "shape": (2, 3),
            "missing": float("nan"),
            "when": pd.Timestamp("2026-09-24 10:00", tz="UTC"),
        },
    )


def _ore_table() -> pd.DataFrame:
    return pd.DataFrame({
        "selected": [True, True],
        "ore_id": ["ore_a", "sinter"],
        "ore_name": ["NMDC Lump", "Sinter"],
        "price_rs_per_mt": np.array([5200.5, 4100.0]),
        "stock_mt": np.array([900, 1500], dtype="int64"),
    })


def _state() -> dict:
    return {
        "bmo_target_production_mt": 2400.0,
        "bmo_slag_cap": np.float64(330.0),
        "bmo_lp_result": _evaluation(cost=9100.0),
        "bmo_de_result": _evaluation(cost=9000.0),
        "bmo_source_cache_version": 3,          # internal: skipped
        "bmo_ui_label": "note",                 # panel widget: skipped
        "bmo_refresh_source_data": True,        # a button: not restorable
        "testbmo_target_production_mt": 1.0,    # the other namespace: ignored
        "other_key": 1,
    }


def _history(n: int = 1000) -> pd.DataFrame:
    """Hourly history with a sparse lab column and a derived-feature pair."""

    rng = np.random.default_rng(3)
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "PRODUCTIONTONNESPERHR": rng.uniform(90, 110, n),
        "PCI_CALC_MT": rng.uniform(15, 20, n),
        "COKE_ASH%": np.nan,
        "COUNT": np.arange(n, dtype="int64"),
    }, index=index)
    frame.iloc[::37, frame.columns.get_loc("COKE_ASH%")] = rng.uniform(10, 12, len(frame.iloc[::37]))
    # Recent PCI MT missing: the latest joint rows lie outside the plain tail.
    frame.iloc[-150:, frame.columns.get_loc("PCI_CALC_MT")] = np.nan
    return frame


# --- exact encoding -------------------------------------------------------------------


def test_round_trip_is_exact_for_frames_numbers_and_types():
    frame = _ore_table()
    frame["missing"] = [np.nan, 1.0 / 3.0]
    value = {"t": frame, "n": np.int64(4), "x": np.nan, "tup": (1, "a"),
             "ts": pd.Timestamp("2026-09-24 10:00:00.123", tz="Asia/Kolkata"),
             "f": 0.1 + 0.2, "keys": {1: "one"}}
    back = decode(json.loads(json.dumps(encode(value))))
    pd.testing.assert_frame_equal(back["t"], frame)
    assert back["n"] == 4 and np.isnan(back["x"]) and back["tup"] == (1, "a")
    assert back["ts"] == value["ts"] and back["f"] == 0.1 + 0.2 and back["keys"] == {1: "one"}


def test_datetime_index_frame_round_trip():
    frame = _history(50)
    pd.testing.assert_frame_equal(decode(json.loads(json.dumps(encode(frame)))), frame)


def test_dataclasses_rebuild_typed_and_plain():
    doc = json.loads(json.dumps(encode(_evaluation())))
    typed = decode(doc)
    assert isinstance(typed, BlendEvaluation)
    assert isinstance(typed.diagnostics["model_prediction"], ModelPrediction)
    assert typed.diagnostics["shape"] == (2, 3)
    plain = decode(doc, typed=False)
    assert plain["diagnostics"]["model_prediction"]["value"] == 13507.34


def test_foreign_classes_are_never_constructed():
    doc = {"__type__": "dataclass", "class": "subprocess.Popen", "data": {"args": "x"}}
    assert decode(doc) == {"args": "x"}


def test_unencodable_value_becomes_repr_not_failure():
    assert encode(object())["__type__"] == "repr"


# --- history compaction ----------------------------------------------------------------


def test_compacted_history_answers_every_model_lookup():
    from utils.bmo.feature_builder import _derived_thm_history_lookup, _temporal_feature_value
    from utils.bmo.fuel_rates import _pick_from_history

    full = _history()
    doc = json.loads(json.dumps(compact_frame(full)))
    rebuilt = _decode_compact_frame(doc)
    assert len(doc["positions"]) < len(full) / 3
    assert len(rebuilt) == len(full)
    assert verify_compact(full, rebuilt, 24) == []
    for lag in range(5):
        assert _derived_thm_history_lookup(rebuilt, "PCI_CALC_THM", lag_steps=lag) == \
            _derived_thm_history_lookup(full, "PCI_CALC_THM", lag_steps=lag)
    assert _temporal_feature_value("trend_index", rebuilt) == _temporal_feature_value("trend_index", full)
    assert _temporal_feature_value("day_of_year", rebuilt) == _temporal_feature_value("day_of_year", full)
    assert _pick_from_history(("COKE_ASH%",), history_df=rebuilt, require_positive=True) == \
        _pick_from_history(("COKE_ASH%",), history_df=full, require_positive=True)
    pd.testing.assert_series_equal(rebuilt.iloc[-1], full.iloc[-1].astype(rebuilt.iloc[-1].dtype),
                                   check_dtype=False)


def test_capture_compacts_history_and_reports_it():
    state = _state()
    history = _history()
    state[f"bmo_{K_RECORD}"] = None
    state[f"bmo_{K_LATEST}"] = {
        "get_history_frame(online_lag_hours=4)": (history, []),
        "get_process_context(history_df=<frame>)": ({"A": 1.0}, []),
    }
    snap = capture(state)
    check = snap["frozen"]["history_checks"][0]
    assert check["rows"] == len(history) and check["mismatched_columns"] == []
    replay = frozen_state(json.loads(json.dumps(snap)))
    rebuilt = replay["provider_calls"]["get_history_frame(online_lag_hours=4)"][0]
    assert len(rebuilt) == len(history)


# --- capture -----------------------------------------------------------------------------


def test_capture_classifies_keys():
    snap = capture(_state(), prefix="bmo_", page_vars={"edited_df": _ore_table()},
                   label=" trial ", now=datetime(2026, 9, 24, 10, 0, tzinfo=IST))
    assert snap["schema"] == "bmo-snapshot/2"
    assert snap["created_at"] == "2026-09-24T10:00:00+05:30"
    assert snap["source"] == "Blend Mix Optimiser" and snap["label"] == "trial"
    assert set(snap["results"]) == {"lp_result", "de_result"}
    assert {"target_production_mt", "slag_cap", "applied_ore_editor_df"} <= set(snap["inputs"])
    assert "source_cache_version" not in snap["inputs"]
    assert not any(k.startswith("ui_") for k in snap["inputs"])
    assert "refresh_source_data" in snap["not_restorable"]
    assert snap["run"]["pinned"] is False
    json.dumps(snap)  # fully serialisable


def test_page_tables_override_session_state():
    state = _state()
    state["bmo_applied_ore_editor_df"] = _ore_table().iloc[:1]
    snap = capture(state, page_vars={"edited_df": _ore_table()})
    assert len(decode(snap["inputs"]["applied_ore_editor_df"])) == 2


def test_summary_prefers_de_unless_it_fell_back():
    snap = capture(_state(), page_vars={"edited_df": _ore_table()})
    s = snap["summary"]
    assert s["result"] == "DE total cost"
    assert s["total_cost_rs_thm"] == pytest.approx(9100.0)  # 9000 + 100 flux
    assert s["coke_rate_kg_thm"] == 395.0 and s["fuel_rate_kg_thm"] == 635.0
    assert s["slag_rate_kg_thm"] == 320.5 and s["basicity_b2"] == 1.12
    assert s["production_mt"] == 2400.0
    assert s["blend_pct"] == {"NMDC Lump": 60.0, "Sinter": 40.0}

    state = _state()
    state["bmo_de_result"] = _evaluation(fell_back=True)
    assert capture(state)["summary"]["result"] == "LP baseline"


def test_summary_of_inputs_only_snapshot_is_empty_not_error():
    s = summarise({"inputs": {"target_production_mt": 2000}})
    assert s["result"] == "" and s["production_mt"] == 2000.0 and s["feasible"] is None


# --- recording and pinning a run --------------------------------------------------------------


class _FakeProvider:
    def __init__(self):
        self.history_rows = 10

    def build_ore_inputs(self, mode, window_days):
        return ([f"ores-{mode}"], {"warnings": []})

    def get_history_frame(self, online_lag_hours=0):
        return pd.DataFrame({"a": range(self.history_rows)}), []


def test_run_pins_the_history_read_before_the_results_changed():
    state: dict = {"bmo_manual_quantities_mt": None}
    fake = _FakeProvider()
    provider = RecordingProvider(fake, state, "bmo_")      # rerun with the LP click
    provider.build_ore_inputs(mode="latest", window_days=30)
    provider.get_history_frame(online_lag_hours=4)          # the LP reads 10 rows
    state["bmo_lp_result"] = _evaluation()                   # results stored
    state["bmo_manual_quantities_mt"] = {"x": 1.0}           # comparison writes after the run
    fake.history_rows = 12
    provider.get_history_frame(online_lag_hours=4)          # comparison re-reads 12 rows
    assert finalize_run(state, "bmo_", {"bmo_cfg": {"a": 1}})
    record = state[f"bmo_{K_RECORD}"]
    assert len(record["provider_calls"]["get_history_frame(online_lag_hours=4)"][0]) == 10
    assert record["session_at_run"] == {"manual_quantities_mt": None, "manual_si": None}
    assert record["page"] == {"bmo_cfg": {"a": 1}}

    snap = capture(state)
    assert snap["run"]["pinned"] is True
    # A rerun without a new run pins nothing new.
    RecordingProvider(fake, state, "bmo_").get_history_frame(online_lag_hours=4)
    assert finalize_run(state, "bmo_", {}) is False


def test_replay_answers_from_the_snapshot_and_lists_live_fallbacks():
    fallbacks: list[str] = []
    key = call_key("build_ore_inputs", (), {"mode": "latest", "window_days": 30})
    replay = ReplayProvider(_FakeProvider(), {key: (["frozen"], {})}, fallbacks)
    assert replay.build_ore_inputs(mode="latest", window_days=30) == (["frozen"], {})
    assert replay.build_ore_inputs(mode="avg", window_days=30)[0] == ["ores-avg"]
    assert fallbacks == ["build_ore_inputs(mode='avg', window_days=30)"]


def test_frozen_config_folds_in_keyless_widget_values():
    cfg = frozen_config({
        "bmo_cfg": {"slag_balance": {"enabled": True, "pi_loss_pct": 0.2}, "optimization": {"seed": 42}},
        "slag_settings_values": {"pi_loss_pct": 0.5},
        "de_seed_choice": "random",
    })
    assert cfg["slag_balance"] == {"enabled": True, "pi_loss_pct": 0.5}
    assert cfg["optimization"] == {"seed": 42, "initial_solution": "random"}


def test_page_namespace_substitutes_and_refuses_preference_saves():
    frozen = {"page": {"recent_fuel_rates": {"pci_rate_kg_thm": 168.4}}, "provider_calls": {}}
    code = compile(
        "def _recent_fuel_rates_live():\n    return {'pci_rate_kg_thm': 999.0}\n"
        "def save_new_thing_preferences(*a):\n    return 'written'\n"
        "rates = _recent_fuel_rates_live()\n"
        "def use():\n    return rates\n",
        "<page>", "exec",
    )
    g: dict = {}
    exec(code, g, PageNamespace(g, page_substitutes(frozen, [])))
    assert g["rates"] == {} and g["use"]() == {}
    with pytest.raises(RuntimeError, match="Sandbox"):
        g["save_new_thing_preferences"]()


def test_restore_writes_inputs_results_and_run_start_values():
    snap = json.loads(json.dumps(capture(_state(), page_vars={"edited_df": _ore_table()})))
    writes = restorable_state(snap)
    assert writes["testbmo_target_production_mt"] == 2400.0
    assert isinstance(writes["testbmo_applied_ore_editor_df"], pd.DataFrame)
    assert all(k.startswith("testbmo_") for k in writes)
    assert not any("result" in k for k in writes)
    # Results come back only with frozen plant data beside them.
    assert results_state(snap) == {}
    snap["frozen"]["provider_calls"] = {"x()": 1}
    restored = results_state(snap)
    assert isinstance(restored["testbmo_de_result"], BlendEvaluation)


def test_restore_drops_non_writable_keys_even_if_hand_edited_in():
    writes = restorable_state({"inputs": {
        "refresh_source_data": True, "ui_label": "x", "slag_cap": 300.0, "lp_result": 1,
    }})
    assert writes == {"testbmo_slag_cap": 300.0}


def test_validate_accepts_bare_inputs_and_rejects_junk():
    assert validate({"inputs": {}})["schema"].startswith("bmo-snapshot/")
    for bad in ([], {"inputs": 1}, {"schema": "other/1", "inputs": {}}):
        with pytest.raises(ValueError):
            validate(bad)


# --- store ---------------------------------------------------------------------


def test_store_save_list_load_delete(tmp_path):
    now = datetime(2026, 9, 24, 10, 0, tzinfo=IST)
    first = save(capture(_state(), now=now), tmp_path)
    second = save(capture(_state(), now=now), tmp_path)
    assert first.stem == "20260924_100000_blend_mix_optimiser"
    assert second.stem == first.stem + "_2"
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    rows, broken = list_snapshots(tmp_path)
    assert [r["id"] for r in rows] == [second.stem, first.stem]
    assert broken == ["broken.json"]
    assert load(first.stem, tmp_path)["id"] == first.stem
    assert delete(first.stem, tmp_path) and not first.exists()
    assert not list(tmp_path.glob("*.tmp"))


# --- report --------------------------------------------------------------------


def _docx(data: bytes):
    from docx import Document

    return Document(io.BytesIO(data))


def _report_snapshot() -> dict:
    return json.loads(json.dumps(capture(
        _state(),
        page_vars={"edited_df": _ore_table(), "target_slag_qty_mt": 700.0,
                   "target_slag_basicity_min": 1.05, "target_slag_basicity_max": 1.15,
                   "max_burden_qty_mt": 3100.0, "target_fe_mt": 1700.0},
        label="Shift B",
    )))


def test_report_has_sections_charts_and_constraint_status():
    doc = _docx(build_docx(_report_snapshot()))
    assert doc.core_properties.title == "Blend Mix Optimiser Snapshot Report"
    assert doc.core_properties.author == "Evonith Steel BF2"
    headings = [p.text for p in doc.paragraphs if p.style.name == "Heading 1"]
    for needle in ("Executive summary", "Recommended blend", "Constraint check",
                   "Fuel and coke", "Slag", "Cost and options compared", "Plant data"):
        assert any(needle in h for h in headings), needle
    numbers = [int(h.split(".")[0]) for h in headings]
    assert numbers == list(range(1, len(numbers) + 1))
    assert len(doc.inline_shapes) >= 4  # donut, waterfall, slag window, cost bars
    cells = [c.text for t in doc.tables for r in t.rows for c in r.cells]
    assert "✓ Met" in cells and "NMDC Lump" in " ".join(cells)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Shift B" in text and "Every constraint is met" in text

    from docx.oxml.ns import qn

    # Fixed table geometry must agree at all three OOXML levels; otherwise Word
    # renders the wide appendix tables as equal columns despite the intended widths.
    for table in doc.tables:
        grid = [int(col.get(qn("w:w"))) for col in table._tbl.tblGrid.gridCol_lst]
        tbl_w = table._tbl.tblPr.find(qn("w:tblW"))
        assert tbl_w is not None and tbl_w.get(qn("w:type")) == "dxa"
        assert int(tbl_w.get(qn("w:w"))) == sum(grid)
        for row in table.rows:
            cell_widths = [int(cell._tc.get_or_add_tcPr().tcW.get(qn("w:w"))) for cell in row.cells]
            assert cell_widths == grid
            assert row._tr.get_or_add_trPr().find(qn("w:cantSplit")) is not None
        if len(table.rows) > 1:
            assert table.rows[0]._tr.get_or_add_trPr().find(qn("w:tblHeader")) is not None


@pytest.mark.parametrize("state", [
    {"bmo_lp_result": _evaluation()},
    {"bmo_target_production_mt": 2400.0},
])
def test_report_handles_partial_snapshots(state):
    assert build_docx(json.loads(json.dumps(capture(state))))[:2] == b"PK"


def test_report_reads_schema_1_snapshots():
    v1 = {
        "schema": "bmo-snapshot/1", "id": "old", "created_at": "2026-09-24T17:05:18+05:30",
        "inputs": {"applied_ore_editor_df": {"__type__": "dataframe", "data": json.loads(
            _ore_table().to_json(orient="split")), "dtypes": {}}},
        "results": {"lp_result": {"__type__": "dataclass", "class": "x.Y",
                                  "data": {"shares_pct": {"ore_a": 60.0, "sinter": 40.0}}}},
    }
    v1["summary"] = summarise(v1)
    assert build_docx(v1)[:2] == b"PK"


# --- sandbox -------------------------------------------------------------------


def _constants(code) -> list[str]:
    out, stack = [], [code]
    while stack:
        c = stack.pop()
        for const in c.co_consts:
            if isinstance(const, str):
                out.append(const)
            elif hasattr(const, "co_consts"):
                stack.append(const)
    return out


def test_sandbox_page_has_no_live_keys_left():
    code, renamed, swapped = sandbox.compile_sandbox_page()
    strings = _constants(code)
    assert renamed > 50
    assert not [s for s in strings if sandbox.is_key_shaped(s)]
    assert "testbmo_lp_result" in strings
    assert "_testbmo_source_cache" in strings  # the page's private source cache too
    assert "testbmo_" in strings  # the panel/gate prefix argument
    assert swapped == (
        "ui.bmo.commentary",
        "ui.bmo.components",
        "ui.bmo.model_accuracy",
        "data.bmo.context_provider",
    )


def test_file_names_are_not_renamed():
    assert not sandbox.is_key_shaped("bmo_style.css")
    code, _count = sandbox._renamed_code(sandbox.SRC / "ui" / "bmo" / "components.py")
    strings = _constants(code)
    assert "bmo_style.css" in strings
    assert "testbmo_hm_carbon_pct" in strings and "bmo_hm_carbon_pct" not in strings


def test_helper_modules_found_by_scan():
    assert sandbox.helper_modules_with_keys() == [
        "ui.bmo.commentary",
        "ui.bmo.components",
        "ui.bmo.model_accuracy",
        "data.bmo.context_provider",
    ]


def test_swapped_helper_is_a_copy_and_real_module_untouched():
    import sys

    import ui.bmo.components as real

    _code, sandbox_builtins, _n, _swapped = sandbox._sandbox(sandbox._stamp())
    imp = sandbox_builtins["__import__"]
    copy = imp("ui.bmo.components", {}, {}, ("render_header",), 0)
    via_package = imp("ui.bmo", {}, {}, ("render_header",), 0)
    assert copy is not real and copy.__name__.endswith("__testbmo")
    assert "testbmo_hm_carbon_pct" in _constants(copy.render_hot_metal_chemistry.__code__)
    assert "bmo_hm_carbon_pct" in _constants(real.render_hot_metal_chemistry.__code__)
    assert via_package.render_hot_metal_chemistry is copy.render_hot_metal_chemistry
    assert sys.modules["ui.bmo.components"] is real


def test_non_writable_widgets_are_detected():
    exact, _patterns = sandbox.non_writable_suffixes()
    assert {"refresh_source_data", "load_diagnostics"} <= exact
    assert not sandbox.is_writable_suffix("refresh_source_data")
    assert sandbox.is_writable_suffix("target_production_mt")
