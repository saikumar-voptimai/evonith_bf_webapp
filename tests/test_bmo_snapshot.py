"""Snapshot capture/restore, the file store, the docx report and the TestBMO sandbox."""

from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from utils.bmo import sandbox
from utils.bmo.snapshot import (
    IST,
    capture,
    decode,
    encode,
    restorable_state,
    summarise,
    validate,
)
from utils.bmo.snapshot_report import build_docx
from utils.bmo.snapshot_store import delete, list_snapshots, load, save
from utils.bmo.types import BlendEvaluation


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
        diagnostics={
            "de_fell_back_to_lp": fell_back,
            "fuel_rate_estimate": {
                "coke_rate_kg_thm": 395.0,
                "nut_coke_rate_kg_thm": 70.0,
                "pci_rate_kg_thm": 170.0,
                "total_fuel_rate_kg_thm": 555.0,
            },
            "flux_cost_per_thm_rs": 100.0,
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


# --- encoding ------------------------------------------------------------------


def test_encode_decode_round_trip_keeps_frames_and_numbers():
    frame = _ore_table()
    frame["missing"] = [np.nan, 1.0]
    doc = json.loads(json.dumps(encode({"t": frame, "n": np.int64(4), "x": np.nan})))
    back = decode(doc)
    pd.testing.assert_frame_equal(back["t"], frame)
    assert back["n"] == 4 and back["x"] is None


def test_encode_dataclass_decodes_to_plain_dict():
    back = decode(json.loads(json.dumps(encode(_evaluation()))))
    assert back["shares_pct"] == {"ore_a": 60.0, "sinter": 40.0}
    assert back["diagnostics"]["fuel_rate_estimate"]["coke_rate_kg_thm"] == 395.0


def test_unencodable_value_becomes_repr_not_failure():
    assert encode(object())["__type__"] == "repr"


# --- capture -------------------------------------------------------------------


def test_capture_classifies_keys():
    snap = capture(_state(), prefix="bmo_", page_vars={"edited_df": _ore_table()},
                   label=" trial ", now=datetime(2026, 9, 24, 10, 0, tzinfo=IST))
    assert snap["created_at"] == "2026-09-24T10:00:00+05:30"
    assert snap["source"] == "Blend Mix Optimiser" and snap["label"] == "trial"
    assert set(snap["results"]) == {"lp_result", "de_result"}
    assert {"target_production_mt", "slag_cap", "applied_ore_editor_df"} <= set(snap["inputs"])
    assert "source_cache_version" not in snap["inputs"]
    assert not any(k.startswith("ui_") for k in snap["inputs"])
    assert "refresh_source_data" in snap["not_restorable"]
    assert "refresh_source_data" not in snap["inputs"]
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
    assert s["coke_rate_kg_thm"] == 395.0 and s["fuel_rate_kg_thm"] == 555.0
    assert s["slag_rate_kg_thm"] == 320.5 and s["basicity_b2"] == 1.12
    assert s["production_mt"] == 2400.0
    assert s["blend_pct"] == {"NMDC Lump": 60.0, "Sinter": 40.0}

    state = _state()
    state["bmo_de_result"] = _evaluation(fell_back=True)
    assert capture(state)["summary"]["result"] == "LP baseline"


def test_summary_of_inputs_only_snapshot_is_empty_not_error():
    s = summarise({"inputs": {"target_production_mt": 2000}})
    assert s["result"] == "" and s["production_mt"] == 2000.0 and s["feasible"] is None


# --- restore -------------------------------------------------------------------


def test_restorable_state_writes_inputs_only_into_sandbox_namespace():
    snap = json.loads(json.dumps(capture(_state(), page_vars={"edited_df": _ore_table()})))
    writes = restorable_state(snap)
    assert writes["testbmo_target_production_mt"] == 2400.0
    assert isinstance(writes["testbmo_applied_ore_editor_df"], pd.DataFrame)
    assert not any("lp_result" in k or "de_result" in k for k in writes)
    assert all(k.startswith("testbmo_") for k in writes)


def test_restore_drops_non_writable_keys_even_if_hand_edited_in():
    writes = restorable_state({"inputs": {
        "refresh_source_data": True, "ui_label": "x", "slag_cap": 300.0,
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


def _docx_text(data: bytes) -> tuple[str, int]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    text += "\n".join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    return text, len(doc.inline_shapes)


def test_report_has_sections_and_pie():
    snap = json.loads(json.dumps(capture(
        _state(), page_vars={"edited_df": _ore_table(), "hm_chem_values": {"c": 4.5}},
        label="Shift B",
    )))
    data = build_docx(snap)
    assert data[:2] == b"PK"
    text, pictures = _docx_text(data)
    assert pictures >= 1
    for needle in ("Recommended blend", "NMDC Lump", "Slag", "Shift B"):
        assert needle in text, needle


def test_report_sections_are_numbered_without_gaps():
    from docx import Document

    data = build_docx(json.loads(json.dumps(capture({"bmo_lp_result": _evaluation()}))))
    numbers = [
        int(p.text.split(".")[0]) for p in Document(io.BytesIO(data)).paragraphs
        if p.style.name == "Heading 1"
    ]
    assert numbers == list(range(1, len(numbers) + 1))


@pytest.mark.parametrize("state", [
    {"bmo_lp_result": _evaluation()},
    {"bmo_target_production_mt": 2400.0},
])
def test_report_handles_partial_snapshots(state):
    assert build_docx(json.loads(json.dumps(capture(state))))[:2] == b"PK"


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
    assert "testbmo_" in strings  # the snapshot panel's prefix argument
    assert swapped == ("ui.bmo.components",)


def test_file_names_are_not_renamed():
    assert not sandbox.is_key_shaped("bmo_style.css")
    code, _count = sandbox._renamed_code(sandbox.SRC / "ui" / "bmo" / "components.py")
    strings = _constants(code)
    assert "bmo_style.css" in strings
    assert "testbmo_hm_carbon_pct" in strings and "bmo_hm_carbon_pct" not in strings


def test_helper_modules_found_by_scan():
    assert sandbox.helper_modules_with_keys() == ["ui.bmo.components"]


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
    # A name re-exported by the package resolves to the renamed copy.
    assert via_package.render_hot_metal_chemistry is copy.render_hot_metal_chemistry
    assert sys.modules["ui.bmo.components"] is real


def test_non_writable_widgets_are_detected():
    exact, _patterns = sandbox.non_writable_suffixes()
    assert {"refresh_source_data", "load_diagnostics"} <= exact
    assert not sandbox.is_writable_suffix("refresh_source_data")
    assert sandbox.is_writable_suffix("target_production_mt")
