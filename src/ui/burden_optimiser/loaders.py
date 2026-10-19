"""Cached loaders for the Burden & Fuel Optimiser page.

* Yard: the Blend Mix Optimiser's provider (Neon chemistry + stock) and the operator's saved inputs; if that fails
  (no database connection), the reference yard of 16 Oct 2026 shipped with the engine, clearly labelled.
* Calibration and regime: the Coke Energy pipeline result from the Coke Energy page's own cached loader, so both pages
  use the same shifts; the shipped Jul-Oct 2026 calibration if the pipeline cannot run.
* Runs: cached on a hash of every input, so changing a tab or re-opening an expander does not re-solve.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import streamlit as st

from config.config_loader import load_config
from data.burden_engine.calibration import CalibrationInfo, fallback_calibration, live_calibration
from data.burden_engine.inputs import load_yard
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH, Engine, Limits, Settings
from utils.burden_engine.scenarios import run_cheapest_level, run_fixed, run_max, value_of_limits


@st.cache_data(show_spinner=False)
def page_config(mtime_ns: int) -> dict:
    return load_config("burden_optimiser.yml").get("burden_optimiser", {})


@st.cache_data(show_spinner=False)
def coke_energy_config(mtime_ns: int) -> dict:
    return load_config("coke_energy.yml").get("coke_energy", {})


def reference_yard() -> dict[str, Any]:
    """The 16 Oct 2026 UAT yard (case report), as yard rows."""
    with open(REFERENCE_INPUTS_PATH, encoding="utf-8") as fh:
        ref = json.load(fh)
    mats = [dict(m, selected=m.get("selected_in_uat", True), stock_mt=None, chem_source="reference 16 Oct 2026",
                 chem_time=m.get("chem_period", ""), carried=True, reason="") for m in ref["materials"]]
    fl = [dict(f, stock_mt=None, max_qty_mt=f.get("max_t_day"), carried=True) for f in ref["fluxes"]]
    return dict(materials=mats, fluxes=fl, fuel=ref["fuel"], slag=ref.get("slag", {}), charging_saved=dict(
        charge_t=ref["charging"]["uat_charge_t"], charges_per_h=ref["charging"]["uat_charges_per_h"]), warnings=[])


@st.cache_data(show_spinner="Reading the yard: raw-material chemistry, stock and saved prices…", ttl=1800)
def yard(cfg_json: str, nonce: int) -> tuple[dict[str, Any], str, str]:
    """(yard, source, message). Source is 'live' or 'reference'."""
    cfg = json.loads(cfg_json)
    try:
        y = load_yard(cfg)
        if not any(m["carried"] for m in y["materials"]):
            raise RuntimeError("no material has lab chemistry in the window")
        return y, "live", ""
    except Exception as exc:  # noqa: BLE001 - offline / no DB: fall back to the reference yard
        return reference_yard(), "reference", f"Live yard unavailable ({type(exc).__name__}: {exc})."


def calibration(cfg: Mapping[str, Any], nonce: int) -> tuple[CalibrationInfo, Any]:
    """(CalibrationInfo, Coke Energy PageData or None)."""
    from pathlib import Path

    try:
        from ui.coke_energy.loaders import data_keys, page_data
        from utils.coke_energy.pipeline import PipelineSettings

        path = Path(__file__).resolve().parents[2] / "config" / "coke_energy.yml"
        ce_cfg = coke_energy_config(path.stat().st_mtime_ns if path.exists() else 0)
        data = page_data(data_keys(ce_cfg, nonce), PipelineSettings.from_config(ce_cfg))
        return _live(cfg, data), data
    except Exception as exc:  # noqa: BLE001 - the optimiser must still open without the plant data
        info = CalibrationInfo(fallback_calibration(cfg.get("calibration", {}).get("fallback_path")), None, "fallback",
                               ("", ""), 0, f"Plant data unavailable ({type(exc).__name__}: {exc}): using the shipped "
                               "Jul-Oct 2026 calibration and its operating point.")
        return info, None


def _live(cfg: Mapping[str, Any], data: Any) -> CalibrationInfo:
    key = (json.dumps(cfg.get("calibration", {}), sort_keys=True), id(data.result))
    memo = st.session_state.setdefault("bfo_cal_memo", {})
    if key not in memo:
        memo.clear()
        memo[key] = live_calibration(data.result, cfg)
    return memo[key]


# ---------------------------------------------------------------------------------------------- runs
def _hash(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def engine(inputs: Mapping[str, Any], cal: Mapping[str, Any]) -> Engine:
    key = _hash(inputs, cal)
    store = st.session_state.setdefault("bfo_engine", {})
    if key not in store:
        store.clear()
        store[key] = Engine(dict(inputs), dict(cal))
    return store[key]


@st.cache_data(show_spinner=False, max_entries=16)
def run_mode(mode: str, inputs_json: str, cal_json: str, settings: dict, limits: dict, prod_t_day: float,
             search_o2: bool, n_o2: int, P_lo: float, step: float) -> dict[str, Any]:
    eng = Engine(json.loads(inputs_json), json.loads(cal_json))
    s, L = Settings(**settings), Limits(**limits)
    if mode == "fixed":
        out = run_fixed(eng, prod_t_day, s, L)
        out["value_of_limits"] = value_of_limits(eng, prod_t_day, s, L, out["with_fuel"])
        return out
    if mode == "cheapest":
        return run_cheapest_level(eng, s, L, search_o2=search_o2, n_o2=n_o2, P_lo=P_lo, step=step)
    return run_max(eng, s, L)


def as_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def clear() -> None:
    yard.clear()
    run_mode.clear()
    for k in ("bfo_cal_memo", "bfo_engine"):
        st.session_state.pop(k, None)

