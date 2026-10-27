"""Streamlit caching for the Burden & Fuel Optimiser page.

The yard, calibration and run logic lives in ``services.optimiser`` (#271), shared with the twin-api and the worker.
This module adds ``st.cache_data`` around it (spinners, copy-on-read) and keeps the two per-session memos (live
calibration, solved Engine) in ``st.session_state``. Calibration reads the Coke Energy page's cached ``page_data``, so
both pages use one pipeline run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import streamlit as st

from data.burden_engine.calibration import CalibrationInfo, live_calibration
from services import optimiser as svc
from services.optimiser import as_json, reference_yard  # noqa: F401 (re-exported for the page)
from utils.burden_engine.engine import Engine


@st.cache_data(show_spinner=False)
def page_config(mtime_ns: int) -> dict:
    return svc.read_page_config(mtime_ns)


@st.cache_data(show_spinner=False)
def coke_energy_config(mtime_ns: int) -> dict:
    return svc.read_coke_energy_config(mtime_ns)


@st.cache_data(show_spinner="Reading the yard: raw-material chemistry, stock and saved prices…", ttl=1800)
def yard(cfg_json: str, nonce: int) -> tuple[dict[str, Any], str, str]:
    """(yard, source, message). Source is 'live' or 'reference'."""
    return svc.read_yard(json.loads(cfg_json))


def calibration(cfg: Mapping[str, Any], nonce: int) -> tuple[CalibrationInfo, Any]:
    """(CalibrationInfo, Coke Energy PageData or None)."""
    from ui.coke_energy.loaders import page_data

    return svc.calibration(cfg, nonce, page_data_for=page_data, ce_config_for=coke_energy_config, live=_live)


def _live(cfg: Mapping[str, Any], data: Any) -> CalibrationInfo:
    key = (json.dumps(cfg.get("calibration", {}), sort_keys=True), id(data.result))
    memo = st.session_state.setdefault("bfo_cal_memo", {})
    if key not in memo:
        memo.clear()
        memo[key] = live_calibration(data.result, cfg)
    return memo[key]


def engine(inputs: Mapping[str, Any], cal: Mapping[str, Any]) -> Engine:
    key = svc.input_hash(inputs, cal)
    store = st.session_state.setdefault("bfo_engine", {})
    if key not in store:
        store.clear()
        store[key] = Engine(dict(inputs), dict(cal))
    return store[key]


@st.cache_data(show_spinner="Reading today's charged blend…", ttl=900)
def charged_blend(materials_json: str, nonce: int) -> dict[str, Any]:
    """Ferrous blend charged over the last 24 h (Neon charge_data); empty with the reason when unavailable."""
    return svc.read_charged_blend(json.loads(materials_json))


@st.cache_data(show_spinner=False, max_entries=16)
def run_mode(
    mode: str,
    inputs_json: str,
    cal_json: str,
    settings: dict,
    limits: dict,
    prod_t_day: float,
    search_o2: bool,
    n_o2: int,
    P_lo: float,
    step: float,
    today_json: str = "{}",
    auto: bool = False,
) -> dict[str, Any]:
    # as the worker runs it: with STRATA-ω the independent network solves in solver processes (SV-1.8)
    return svc.solve_case(
        mode, inputs_json, cal_json, settings, limits, prod_t_day, search_o2, n_o2, P_lo, step, today_json, auto
    )


def clear() -> None:
    yard.clear()
    charged_blend.clear()
    run_mode.clear()
    for k in ("bfo_cal_memo", "bfo_engine"):
        st.session_state.pop(k, None)
