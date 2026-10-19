"""Coke Energy Model — BF2.

The BF-02 mass + energy balance (built in the Claude web app, kept almost unchanged in ``utils.coke_energy``) run on
every shift, with two out-of-sample coke-rate models:

    oxygen-balance model   eta CO fixed at the analyser's value, heat-loss allowance from the previous 14 days
    energy-balance model   two-zone (Rist): shaft efficiency omega fixed at its 14-day median; eta CO and the
                           top-gas temperature are outputs

checked against the coke the metered blast burnt and the coke charged.

Layout (step 1 of docs/coke_energy_model_plan.md):
    status line   data freshness + Refresh
    1. Coke rate being used: the two models, burnt and charged coke for the latest (or chosen) shift
    2. Rist diagram: the shift's operating line and the model's line, animated over stable shifts (slider / Play)
    3. Coke-rate timeline: modelled (both), burnt, charged, per shift or per window, with the accuracy table
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.session import is_logged_in

# ── Auth gate ─────────────────────────────────────────────────────────
if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

# ── Lazy imports (only after auth passes) ─────────────────────────────
import pandas as pd

from config.config_loader import load_config
from data.coke_energy.sources import refresh_static_dataset, static_dataset_status
from ui.coke_energy.loaders import clear_caches, data_keys, page_data, rist_frames
from ui.coke_energy.sections import render_latest, render_prediction, render_rist, render_status, shift_window
from utils.coke_energy.freshness import assess, in_use_between, parse_limits
from utils.coke_energy.pipeline import PipelineSettings


@st.cache_data(show_spinner=False)
def _config(mtime_ns: int) -> dict:
    return load_config("coke_energy.yml").get("coke_energy", {})


_cfg_path = Path(__file__).resolve().parents[1] / "config" / "coke_energy.yml"
cfg = _config(_cfg_path.stat().st_mtime_ns if _cfg_path.exists() else 0)
data_cfg = cfg.get("data_sources", {}) or {}
ui_cfg = cfg.get("ui", {}) or {}
settings = PipelineSettings.from_config(cfg)

st.title("Coke Energy Model — BF2")
st.caption(
    "Physics based energy driven modeling using coke rate model. Two models predict the coke rate the furnace is using, and both are "
    "checked against the coke the blast actually burnt and the coke charged. All coke rates are dry, per tonne of hot metal."
)

# ── Keep the shared hourly dataset fresh (same file and URL as the Blend Mix Optimiser) ──
_status = static_dataset_status(data_cfg)
if _status["stale"] and not st.session_state.get("ce_static_refresh_tried"):
    st.session_state["ce_static_refresh_tried"] = True
    with st.spinner("Updating the hourly furnace dataset from the plant server…"):
        refresh_static_dataset(data_cfg)


def _refresh() -> None:
    refresh_static_dataset(data_cfg)
    st.session_state["ce_nonce"] = int(st.session_state.get("ce_nonce", 0)) + 1
    clear_caches()
    st.rerun()


keys = data_keys(cfg, int(st.session_state.get("ce_nonce", 0)))
try:
    data = page_data(keys, settings)
except Exception as exc:  # noqa: BLE001 - show the reason instead of a stack trace
    st.error(f"Could not load the data for the balance: {exc}")
    st.stop()

# ── Freshness: as of now (status line) and per shift (under the tiles) ──
limits = parse_limits(cfg.get("freshness_limits"))
charged = data.charged
last_charge = charged.index.max() + pd.Timedelta(hours=1) if not charged.empty else pd.Timestamp.now()
now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
assessed_now = assess(data.samples, now, limits, in_use_between(charged, last_charge - pd.Timedelta(hours=24), last_charge))


def _assess_shift(skey: str) -> pd.DataFrame:
    start, end = shift_window(T.loc[skey])
    return assess(data.samples, end, limits, in_use_between(charged, start, end))


show_charged = render_status(data.fresh, assessed_now, data.names, _refresh)
T = data.result.shifts
if T.empty:
    st.warning("No shift could be balanced from the data available.")
    st.stop()
if data.result.errors:
    with st.expander(f"{len(data.result.errors)} shifts could not be balanced"):
        st.code("\n".join(data.result.errors[:50]))

st.subheader("Coke rate being used")
selected = render_latest(T, _assess_shift, data.names, show_charged)

st.divider()
st.subheader("Prediction against burnt coke")
render_prediction(T, int(ui_cfg.get("timeline_default_days", 30)), show_charged, selected)

st.divider()
st.subheader("Rist diagram")
st.caption(
    "The furnace's oxygen balance as one straight line. Its slope is the carbon gasified per iron atom (a steeper "
    "line burns more coke); the energy model holds the shaft efficiency ω and lets the line pivot about the heat "
    "balance point P."
)
render_rist(T, lambda skeys: rist_frames(keys, settings, skeys), int(ui_cfg.get("rist_default_days", 14)), show_charged)
