"""Streamlit caching for the Burden & Fuel Optimiser page.

The yard, calibration and run logic lives in ``services.optimiser`` (#271), shared with the twin-api and the worker.
This module adds ``st.cache_data`` around it (spinners, copy-on-read) and keeps the two per-session memos (live
calibration, solved Engine) in ``st.session_state``. Calibration reads the Coke Energy page's cached ``page_data``, so
both pages use one pipeline run.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from typing import Any

import streamlit as st

from data.burden_engine.calibration import CalibrationInfo, live_calibration
from services import operator_inputs
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
def cached_yard(cfg_json: str, nonce: int, inputs_token: str = "") -> tuple[dict[str, Any], str, str]:
    return svc.read_yard_for(cfg_json, inputs_token)


def yard(cfg_json: str, nonce: int, inputs_token: str = "") -> tuple[dict[str, Any], str, str]:
    """(yard, source, message). Source is 'live' or 'reference'. ``inputs_token``: the version of the operator inputs
    in force (``services.operator_inputs.token()``), so a saved change is read at once; a yard whose operator inputs
    do not fit the token (store failure, or a save, between the token and the read) is used once and not kept."""
    try:
        return cached_yard(cfg_json, nonce, inputs_token)
    except operator_inputs.TokenMismatch as mismatch:
        return mismatch.value


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


def live_pci() -> float | None:
    """PCI as injected, kg/tHM: the live tag's 1-h mean (``services.optimiser.live_pci_kg_thm``, memoised 10 min)."""
    return svc.live_pci_kg_thm()


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


#: Expected run time, s, of (mode, ω mode) before a run of it has been timed in this process (the SPA's
#: ``expectedSeconds``).
EXPECTED_S = {("fixed", "fixed"): 10.0, ("max", "fixed"): 120.0, ("cheapest", "fixed"): 180.0,
              ("fixed", "blend"): 75.0, ("max", "blend"): 300.0, ("cheapest", "blend"): 1200.0}
_TIMED: dict[tuple[str, str], float] = {}


def _duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def expected_seconds(mode: str, omega_mode: str) -> float:
    """Expected run time, s: the last run of the mode in this process, else ``EXPECTED_S``."""
    key = (mode, "blend" if omega_mode == "blend" else "fixed")
    return _TIMED.get(key, EXPECTED_S.get(key, 60.0))


def expected_text(mode: str, omega_mode: str) -> str:
    return f"Expected run time: about {_duration(expected_seconds(mode, omega_mode))}"


def run_with_progress(*args: Any) -> dict[str, Any]:
    """``run_mode`` in a thread under a progress bar timed against the expected run time: the last run of the same
    mode in this process, else ``EXPECTED_S``. The run's exceptions are raised here."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx

    key = (str(args[0]), "blend" if dict(args[3]).get("omega_mode") == "blend" else "fixed")
    expected = expected_seconds(*key)
    out: dict[str, Any] = {}

    def work() -> None:
        try:
            out["result"] = run_mode(*args)
        except BaseException as exc:  # noqa: BLE001 - raised again in the script thread
            out["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    add_script_run_ctx(thread)
    t0 = time.monotonic()
    thread.start()
    bar = st.progress(0.0, text=f"Optimising · about {_duration(expected)}")
    while True:
        thread.join(0.5)
        if not thread.is_alive():
            break
        elapsed = time.monotonic() - t0
        left = expected - elapsed
        text = (f"Optimising · {_duration(elapsed)} elapsed, about {_duration(left)} left" if left > 0 else
                f"Optimising · {_duration(elapsed)} elapsed, longer than the usual {_duration(expected)}")
        bar.progress(min(elapsed / expected, 0.99), text=text)
    bar.empty()
    if "error" in out:
        raise out["error"]
    elapsed = time.monotonic() - t0
    if elapsed > 2.0:  # a cached result is not a timing
        _TIMED[key] = elapsed
    return out["result"]


def clear() -> None:
    cached_yard.clear()
    charged_blend.clear()
    run_mode.clear()
    svc.live_pci_kg_thm.clear()
    for k in ("bfo_cal_memo", "bfo_engine"):
        st.session_state.pop(k, None)
