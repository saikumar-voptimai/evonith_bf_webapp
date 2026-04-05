"""render_live_operations and render_furnace_intelligence tabs.

Both tabs are currently commented out in the FurnaceMind page but the code
is preserved here for when they are re-enabled.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config

from core.contextual_analyzer import ContextualAnalyzer
from core.influence_attribution import InfluenceAttribution
from core.recurring_anomaly_tracker import RecurringAnomalyTracker
from core.shift_analyzer import ShiftAnalyzer
from core.stability_index import FurnaceStabilityIndex
from llm.llm_client import OpenRouterClient
from memory.aggregation import run_aggregation_if_ready
from memory.retriever import ContextRetriever
from memory.schemas import ShiftSummary
from utils.logger import get_logger
from utils.payload_helpers import build_shift_payload
from utils.settings import settings

from ui.furnacemind_sections import (  # shared helpers (thin re-export file)
    FREQUENCY_TO_TIMEDTA,
    MEASUREMENT_LABELS,
    _ensure_ist,
    load_schemas,
)

logger = get_logger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

_SHIFT_HOURS   = 8
_WINDOW_MINUTES = 15
_ROWS_PER_SHIFT = (_SHIFT_HOURS * 60) // _WINDOW_MINUTES


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_shift_start(ts: datetime) -> datetime:
    ts = _ensure_ist(ts)
    shift_hour = (ts.hour // _SHIFT_HOURS) * _SHIFT_HOURS
    return ts.replace(hour=shift_hour, minute=0, second=0, microsecond=0)


def _get_shift_label(shift_start: datetime) -> str:
    hour = _ensure_ist(shift_start).hour
    return {0: "A", 8: "B"}.get(hour, "C")


def _get_shift_id(shift_start: datetime) -> str:
    label = _get_shift_label(shift_start)
    return f"{shift_start:%Y-%m-%d}_SHIFT_{label}"


@st.cache_data(show_spinner=False, ttl=timedelta(minutes=14))
def _fetch_recent_online(time_range: str, window_by: str = "15 minutes") -> pd.DataFrame:
    config = load_config("setting_ds_dv.yml")
    field_labels = {
        internal_key: human_label
        for mapping in config["data_mapping"].values()
        for human_label, internal_key in mapping.items()
    }
    return dr.fetch_online_df(
        selected_measurements=list(MEASUREMENT_LABELS.keys()),
        time_range=time_range,
        window_by=window_by,
        FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
        MEASUREMENT_LABELS=MEASUREMENT_LABELS,
        FIELD_LABELS=field_labels,
    )


# ── Public renderers ─────────────────────────────────────────────────────────

def render_live_operations(*, structured_store, vector_store) -> None:
    """Render the Live Operations tab with real-time shift detection."""
    schemas = load_schemas()

    st_autorefresh(interval=_WINDOW_MINUTES * 60 * 1000, key="online_refresh")
    st.header("📡 Live Operations — Shift Intelligence")

    for key, default in [
        ("online_shift_buffer",     pd.DataFrame()),
        ("current_shift_start",     None),
        ("completed_shift",         None),
        ("shift_ready_for_analysis", False),
        ("shift_waiting_for_operator", False),
        ("generated_shift_data",    None),
        ("generated_structured",    None),
        ("generated_llm_text",      None),
        ("generated_contextual_text", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Fetch 8h UI snapshot ─────────────────────────────────────────────────
    ui_df = pd.DataFrame()
    try:
        ui_df = _fetch_recent_online("last 8 hours", window_by="15 minutes")
    except Exception as exc:
        st.error("Failed to fetch UI online data.")
        st.exception(exc)
        st.stop()

    if ui_df is None or ui_df.empty:
        st.info("Waiting for online data…")
        st.stop()

    ui_df = ui_df.sort_index().tail(_ROWS_PER_SHIFT)
    st.subheader("📊 Live Online Data (Last 8 Hours)")
    st.dataframe(ui_df, use_container_width=True)

    if st.session_state.online_shift_buffer.empty:
        logger.info("Cold start — backfilling shift buffer from UI data.")
        st.session_state.online_shift_buffer = ui_df.copy()

    # ── Append latest 15-min slice ───────────────────────────────────────────
    try:
        delta_df = _fetch_recent_online("last 15 minutes", window_by="15 minutes")
        if delta_df is not None and not delta_df.empty:
            st.session_state.online_shift_buffer = (
                pd.concat([st.session_state.online_shift_buffer, delta_df])
                .sort_index()
                .drop_duplicates()
            )
    except Exception as exc:
        st.warning("Failed to fetch delta online data (continuing without delta).")
        st.exception(exc)

    # ── Shift boundary detection ─────────────────────────────────────────────
    now            = datetime.now(_IST)
    new_shift_start = _get_shift_start(now)

    if st.session_state.current_shift_start is None:
        st.session_state.current_shift_start = new_shift_start
        logger.info("Initialized current_shift_start = %s", new_shift_start)

        buf = st.session_state.online_shift_buffer
        if not buf.empty:
            buf_shift_start = _get_shift_start(_ensure_ist(buf.index.min().to_pydatetime()))
            if buf_shift_start < new_shift_start:
                prev_data = buf[buf.index < pd.Timestamp(new_shift_start)]
                curr_data = buf[buf.index >= pd.Timestamp(new_shift_start)]
                if not prev_data.empty:
                    _mark_shift_complete(buf_shift_start, new_shift_start, prev_data, curr_data)

    elif new_shift_start > st.session_state.current_shift_start:
        completed_start = st.session_state.current_shift_start
        completed_end   = completed_start + timedelta(hours=_SHIFT_HOURS)
        _mark_shift_complete(
            completed_start, completed_end,
            st.session_state.online_shift_buffer,
            pd.DataFrame(),
        )
        st.session_state.current_shift_start = new_shift_start

    # ── Analyse completed shift if ready ─────────────────────────────────────
    if st.session_state.shift_ready_for_analysis:
        _run_shift_analysis(structured_store, vector_store)

    # ── Show results if waiting for operator sign-off ─────────────────────────
    if st.session_state.shift_waiting_for_operator:
        _show_shift_results(structured_store, vector_store, schemas)


def _mark_shift_complete(
    shift_start: datetime,
    shift_end: datetime,
    shift_data: pd.DataFrame,
    remaining_data: pd.DataFrame,
) -> None:
    st.session_state.completed_shift = {
        "shift_id":    _get_shift_id(shift_start),
        "shift_start": shift_start,
        "shift_end":   shift_end,
        "df":          shift_data.copy(),
    }
    st.session_state.online_shift_buffer      = remaining_data.copy() if not remaining_data.empty else pd.DataFrame()
    st.session_state.generated_structured     = None
    st.session_state.generated_llm_text       = None
    st.session_state.generated_contextual_text = None
    st.session_state.shift_waiting_for_operator = False
    st.session_state.shift_ready_for_analysis  = True


def _run_shift_analysis(structured_store, vector_store) -> None:
    completed = st.session_state.completed_shift
    shift_df  = completed["df"]

    st.subheader("🕒 Completed Shift Detected")
    st.caption(f"{completed['shift_start']} → {completed['shift_end']}")

    try:
        llm               = OpenRouterClient()
        shift_analyzer    = ShiftAnalyzer(settings.anomaly)
        contextual_analyzer = ContextualAnalyzer(llm)
        retriever         = ContextRetriever(structured_store, vector_store)

        shift_data = type(
            "ShiftData", (),
            {
                "shift_id":    completed["shift_id"],
                "shift_start": completed["shift_start"],
                "shift_end":   completed["shift_end"],
                "data":        shift_df,
            },
        )()

        llm_text, structured = shift_analyzer.analyze(
            shift_data=shift_data, prev_shift_data=None, llm=llm,
        )

        if not _has_content(llm_text):
            llm_text = (structured or {}).get("summary_text") or (structured or {}).get("raw_response")

        if isinstance(structured, str):
            structured = {"raw_text": structured}
        if not isinstance(structured, dict):
            structured = {}

        fsi = FurnaceStabilityIndex(
            critical_parameters=[
                "Process Params - BF2_BODY_ETACO",
                "Process Params - BF2_PROC Top Temp Average",
                "Process Params - BF2_PROC Top Pressure Average",
                "Process Params - coke_rate",
                "Process Params - BF2 CO in BF Gas(%)",
                "Process Params - BF2 CO2 in BF Gas (%)",
                "Process Params - BF2_BODY_PERMEABILITY",
            ],
            primary_kpi="Process Params - BF2_BODY_ETACO",
        )
        fsi_result = fsi.compute(df=shift_df, anomaly_count=structured.get("anomaly_count", 0))
        structured["stability_index"]    = fsi_result["stability_index"]
        structured["stability_status"]   = fsi_result["stability_status"]
        structured["stability_penalties"] = fsi_result["penalties"]

        try:
            context = retriever.retrieve_context(
                current_shift_id=shift_data.shift_id,
                current_shift_text=llm_text,
                top_k_similar=3,
            )
        except (AttributeError, TypeError):
            context = {"previous_shift": None, "historical_similar": []}

        contextual_text, _ = contextual_analyzer.build_day_summary(
            day_id=shift_data.shift_id,
            shift_payloads=[{
                "shift_name":      shift_data.shift_id,
                "start_time":      shift_data.shift_start.isoformat(),
                "end_time":        shift_data.shift_end.isoformat(),
                "summary_text":    llm_text,
                "stability_index": structured.get("stability_index"),
                "stability_status": structured.get("stability_status"),
            }],
            previous_shift=context.get("previous_shift"),
            historical_similar=context.get("historical_similar"),
        )

        if not _has_content(contextual_text):
            contextual_text = llm_text or "Contextual summary unavailable."

        st.session_state.generated_shift_data      = shift_data
        st.session_state.generated_structured      = structured
        st.session_state.generated_llm_text        = llm_text
        st.session_state.generated_contextual_text = contextual_text
        st.session_state.shift_ready_for_analysis  = False
        st.session_state.shift_waiting_for_operator = True

    except Exception as exc:
        st.error(f"❌ Shift analysis failed: {exc}")
        st.exception(exc)
        st.session_state.shift_ready_for_analysis = False


def _show_shift_results(structured_store, vector_store, schemas: dict) -> None:
    llm_text = st.session_state.generated_llm_text
    ctx_text = st.session_state.generated_contextual_text

    if not _has_content(llm_text) and not _has_content(ctx_text):
        st.warning(
            "⚠️ Shift analysis completed but returned no usable content. "
            "Check ShiftAnalyzer/ContextualAnalyzer response parsing."
        )
        st.session_state.shift_waiting_for_operator = False
        return

    if _has_content(llm_text):
        st.subheader("🕒 Shift Operational Summary")
        st.markdown(llm_text)

    if _has_content(ctx_text):
        st.subheader("🧠 Context-Aware Insight")
        st.markdown(ctx_text)

    with st.form("operator_submit_form"):
        operator_notes   = st.text_area("📝 Operator Notes")
        operator_rating  = st.slider("⭐ Shift Rating", 1, 5, 3)
        operator_comment = st.text_input("💬 Feedback Comment")
        submit           = st.form_submit_button("✅ Submit & Save Shift")

    if submit:
        shift_data  = st.session_state.generated_shift_data
        structured  = st.session_state.generated_structured
        contextual_text = st.session_state.generated_contextual_text

        operator_context = {
            "notes":    operator_notes,
            "feedback": {"rating": operator_rating, "comment": operator_comment},
        }
        payload = build_shift_payload(
            shift_data=shift_data,
            structured_summary=structured,
            llm_text=contextual_text,
            prev_shift=None,
            schema=schemas["shift"],
            operator_context=operator_context,
        )
        vector_store.add_window(
            window_id=payload["window_id"],
            embedding_text=payload["summary_text"],
            payload=payload,
        )
        structured_store.save_shift_summary(ShiftSummary(
            shift_id          = shift_data.shift_id,
            shift_start       = shift_data.shift_start,
            shift_end         = shift_data.shift_end,
            generated_at      = datetime.now(_IST),
            stability_index   = structured["stability_index"],
            stability_status  = structured["stability_status"],
            stability_penalties = structured["stability_penalties"],
            operator_context  = operator_context,
        ))
        run_aggregation_if_ready(
            new_shift=structured_store.load_shift_summary(shift_data.shift_id),
            store=structured_store,
            vector_store=vector_store,
            schemas={"day": schemas["day"], "week": schemas["week"], "biweek": schemas["biweek"]},
            shifts_per_day=3,
            days_per_week=7,
        )
        st.success("✅ Shift saved. Aggregation triggered.")
        st.session_state.shift_waiting_for_operator = False


def _has_content(text) -> bool:
    return isinstance(text, str) and text.strip() not in ("", "none")


# ── Furnace Intelligence ─────────────────────────────────────────────────────

def render_furnace_intelligence(*, structured_store) -> None:
    """Render the Furnace Intelligence tab (FSI, recurring anomalies, attribution)."""
    st.header("🧠 Furnace Intelligence")

    latest_shift = structured_store.load_latest_shift_summary()
    all_shifts   = structured_store.load_all_shift_summaries() or []

    if latest_shift is None:
        st.warning("No shift summaries found yet. Run shift detection / generate summaries first.")
        st.stop()

    valid_shifts   = sorted(
        [s for s in all_shifts if getattr(s, "shift_end", None) is not None],
        key=lambda s: s.shift_end,
    )
    prev_shift     = valid_shifts[-2] if len(valid_shifts) >= 2 else None
    display_shift  = next(
        (s for s in (latest_shift, prev_shift) if getattr(s, "stability_index", None) is not None),
        None,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if display_shift:
            delta = None
            if display_shift is latest_shift and prev_shift and prev_shift.stability_index is not None:
                delta = round(latest_shift.stability_index - prev_shift.stability_index, 1)
            st.metric("Furnace Stability Index", round(display_shift.stability_index, 1),
                      delta=f"{delta:+}" if delta is not None else None)
        else:
            st.metric("Furnace Stability Index", "—")

    with c2:
        st.markdown("### Current Status")
        if display_shift and getattr(display_shift, "stability_status", None):
            status = display_shift.stability_status.upper()
            if status == "STABLE":
                st.success("🟢 STABLE")
            elif status == "WATCH":
                st.warning("🟡 WATCH")
            else:
                st.error("🔴 UNSTABLE")
        else:
            st.info("Stability data not available yet")

    with c3:
        st.markdown("### Latest Shift")
        if display_shift:
            st.write(display_shift.shift_id)
            st.caption(f"Ended at: {display_shift.shift_end}")
        elif latest_shift:
            st.write(latest_shift.shift_id)
            st.caption("Stability not computed yet")
        else:
            st.write("No shifts yet")

    st.divider()

    st.subheader("🔁 Recurring Anomaly Patterns")
    tracker           = RecurringAnomalyTracker(min_occurrences=3)
    recurring_anomalies = tracker.detect(structured_store.load_last_n_shift_summaries(n=20))

    if recurring_anomalies:
        st.dataframe(
            pd.DataFrame([
                {"Parameter": p, "Frequency": d["count"], "Pattern": d["pattern"], "Last Seen": d["last_seen"]}
                for p, d in recurring_anomalies.items()
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No recurring anomaly patterns detected.")

    st.divider()

    st.subheader("🧭 Influence Attribution")
    attrib           = InfluenceAttribution()
    influence_result = attrib.compute(shift_summary=latest_shift, recurring_anomalies=recurring_anomalies)

    def _classify(index: float) -> str:
        if index >= 0.30: return "🔴 Dominant contributor"
        if index >= 0.15: return "🟠 Significant contributor"
        if index >= 0.05: return "🟡 Moderate contributor"
        return "🟢 Minor contributor"

    if influence_result:
        st.dataframe(
            pd.DataFrame([
                {"Parameter": r["parameter"], "Influence Index": r["influence_index"],
                 "Contribution Level": _classify(r["influence_index"]), "Rank": r["rank"]}
                for r in influence_result
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "ℹ️ Influence Index shows relative contribution to instability within this shift. "
            "Higher values indicate stronger contribution compared to other parameters."
        )
    else:
        st.info("No significant contributors identified.")
