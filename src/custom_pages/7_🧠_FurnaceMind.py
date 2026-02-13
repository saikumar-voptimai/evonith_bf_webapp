# app.py
# FurnaceMind — Single-page industrial-grade Streamlit app

import json
from pathlib import Path
from datetime import datetime, date, timedelta, timezone

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd

from FurnaceMind.utils.settings import settings
from FurnaceMind.utils.logger import get_logger
from FurnaceMind.utils.payload_helpers import build_shift_payload

from FurnaceMind.ui.layout import render_page_header
from FurnaceMind.ui.components import show_report
from FurnaceMind.ui.styles import apply_styles

from FurnaceMind.core.shift_analyzer import ShiftAnalyzer
from FurnaceMind.core.contextual_analyzer import ContextualAnalyzer
from FurnaceMind.core.stability_index import FurnaceStabilityIndex
from FurnaceMind.core.recurring_anomaly_tracker import RecurringAnomalyTracker
from FurnaceMind.core.influence_attribution import InfluenceAttribution

from FurnaceMind.llm.llm_client import OpenAIClient, OpenRouterClient

from FurnaceMind.memory.structured_store import StructuredStore
from FurnaceMind.memory.vector_store import QdrantVectorStore
from FurnaceMind.memory.retriever import ContextRetriever
from FurnaceMind.memory.schemas import ShiftSummary
from FurnaceMind.memory.aggregation import run_aggregation_if_ready

from FurnaceMind.utils.window_helpers import (
    build_shift_window_id,
    build_day_window_id,
    fetch_from_qdrant,
)

from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Load schemas — use a robust path that works both locally and on Streamlit Cloud
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
# Try the original path first; fall back to sibling "FurnaceMind/config" dir
_CANDIDATE_PATHS = [
    _THIS_DIR.parent / "FurnaceMind" / "config",   # original layout
    _THIS_DIR / "FurnaceMind" / "config",           # flat / Cloud layout
    _THIS_DIR / "config",                           # if already inside FurnaceMind
]
BASE = next((p for p in _CANDIDATE_PATHS if p.is_dir()), _CANDIDATE_PATHS[0])

SHIFT_SCHEMA = json.load(open(BASE / "shift_payload_schema.json"))
DAY_SCHEMA = json.load(open(BASE / "day_payload_schema.json"))
WEEK_SCHEMA = json.load(open(BASE / "weekly_payload_schema.json"))
BIWEEK_SCHEMA = json.load(open(BASE / "biweekly_payload_schema.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def validate_shift_dataframe(df: pd.DataFrame, shift_hours: int = 8):
    if df.empty:
        raise ValueError("Uploaded file is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("CSV must have a DatetimeIndex.")

    df = df.sort_index()
    sampling_minutes = (
        df.index.to_series().diff().dropna().median().total_seconds() / 60
    )
    expected_rows = int((shift_hours * 60) / sampling_minutes)

    if len(df) != expected_rows:
        raise ValueError(f"Expected ~{expected_rows} rows, got {len(df)}.")


def load_shift_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    validate_shift_dataframe(df)
    return df.sort_index()


config = load_config("setting_ds_dv.yml")

MEASUREMENT_LABELS = {
    "heatload_delta_t": "Heatload Delta T",
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile",
}

FREQUENCY_TO_TIMEDTA = {
    "None": None,
    "1 minute": "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour": "1h",
    "6 hours": "6h",
    "8 hours": "8h",
    "12 hours": "12h",
    "1 day": "1d",
}

FIELD_LABELS = {
    internal_key: human_label
    for mapping in config["data_mapping"].values()
    for human_label, internal_key in mapping.items()
}


# ---------------------------------------------------------------------------
# FIX #5 — Helper to normalize any datetime to UTC-aware.
#          Prevents "can't compare offset-naive and offset-aware" errors.
# ---------------------------------------------------------------------------
def _ensure_utc(dt: datetime) -> datetime:
    """Make sure a datetime is timezone-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# FIX #1 — Added  ttl=timedelta(minutes=14)  so data actually refreshes
#          on every 15-min auto-refresh cycle instead of being cached forever.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=timedelta(minutes=14))
def fetch_recent_online(
    tr: str = "last 8 hours",
    request_type: str = "windowed-average",
    window_by: str = "15 minutes",
) -> pd.DataFrame:
    selected_measurements = list(MEASUREMENT_LABELS.keys())

    if request_type != "windowed-average":
        raise ValueError(
            f"Unsupported request_type={request_type!r}. "
            "Only 'windowed-average' is supported by fetch_online_df right now."
        )

    return dr.fetch_online_df(
        selected_measurements=selected_measurements,
        time_range=tr,
        average_range=window_by,
        FREQUENCY_TO_TIMEDTA=FREQUENCY_TO_TIMEDTA,
        MEASUREMENT_LABELS=MEASUREMENT_LABELS,
        FIELD_LABELS=FIELD_LABELS,
    )


# ===========================================================================
# MAIN APP
# ===========================================================================
def main():
    st.set_page_config(layout="wide")
    render_page_header()
    apply_styles()

    structured_store = StructuredStore()
    vector_store = QdrantVectorStore()

    # SIDEBAR — NAVIGATION
    st.sidebar.title("FurnaceMind")

    app_mode = st.sidebar.radio(
        "Navigation",
        [
            "📤 Data Management",
            "📊 Reports",
            "🧠 Furnace Intelligence",
        ],
    )

    st.sidebar.divider()

    # ===================================================================
    # 📤 DATA MANAGEMENT — Online Shift (Shift-Aware, No Upload)
    # ===================================================================
    if app_mode == "📤 Data Management":

        # CONFIG
        SHIFT_HOURS = 8
        WINDOW_MINUTES = 15
        ROWS_PER_SHIFT = (SHIFT_HOURS * 60) // WINDOW_MINUTES  # 32

        def get_shift_start(ts: datetime) -> datetime:
            ts = _ensure_utc(ts)
            shift_hour = (ts.hour // SHIFT_HOURS) * SHIFT_HOURS
            return ts.replace(hour=shift_hour, minute=0, second=0, microsecond=0)

        def get_shift_id(shift_start: datetime) -> str:
            return f"{shift_start:%Y-%m-%d_%H}_SHIFT"

        # AUTO REFRESH — every 15 minutes
        st_autorefresh(
            interval=WINDOW_MINUTES * 60 * 1000,
            key="online_refresh",
        )

        st.header("📤 Online Blast Furnace Shift Intelligence")

        # SESSION STATE
        if "online_shift_buffer" not in st.session_state:
            st.session_state.online_shift_buffer = pd.DataFrame()

        if "current_shift_start" not in st.session_state:
            st.session_state.current_shift_start = None

        if "completed_shift" not in st.session_state:
            st.session_state.completed_shift = None

        if "shift_ready_for_analysis" not in st.session_state:
            st.session_state.shift_ready_for_analysis = False

        if "shift_waiting_for_operator" not in st.session_state:
            st.session_state.shift_waiting_for_operator = False

        # -------------------------------------------------------------------
        # FIX #2 — Removed the duplicate / mis-indented dead code block
        #          that was at old lines 207-212.  Only the correct try/except
        #          version below is kept.
        # -------------------------------------------------------------------

        # STEP 1 — UI VIEW (LAST 8 HOURS)
        ui_df = pd.DataFrame()

        try:
            ui_df = fetch_recent_online(
                tr="last 8 hours",
                request_type="windowed-average",
                window_by="15 minutes",
            )
        except Exception as e:
            st.error("Failed to fetch UI online data.")
            st.exception(e)
            st.stop()

        # If fetch returned None for some reason, normalize to empty DF
        if ui_df is None:
            ui_df = pd.DataFrame()

        if ui_df.empty:
            st.info("Waiting for online data…")
            st.stop()

        ui_df = ui_df.sort_index().tail(ROWS_PER_SHIFT)
        st.subheader("📊 Live Online Data (Last 8 Hours)")
        st.dataframe(ui_df, use_container_width=True)

        # -------------------------------------------------------------------
        # FIX #3 — Startup backfill.
        #
        # On Streamlit Cloud the app cold-starts with an empty session_state.
        # If the buffer is empty we seed it with the full 8-hour UI fetch so
        # that shift-boundary detection has data to work with immediately
        # instead of requiring 8 hours of continuous uptime.
        # -------------------------------------------------------------------
        if st.session_state.online_shift_buffer.empty and not ui_df.empty:
            logger.info("Cold start detected — backfilling shift buffer from UI data.")
            st.session_state.online_shift_buffer = ui_df.copy()

        # STEP 2 — INGEST DELTA (LAST 15 MIN)
        delta_df = pd.DataFrame()

        try:
            delta_df = fetch_recent_online(
                tr="last 15 minutes",
                request_type="windowed-average",
                window_by="15 minutes",
            )
        except Exception as e:
            st.warning("Failed to fetch delta online data (continuing without delta).")
            st.exception(e)
            delta_df = pd.DataFrame()

        if delta_df is not None and not delta_df.empty:
            st.session_state.online_shift_buffer = (
                pd.concat([st.session_state.online_shift_buffer, delta_df])
                .sort_index()
                .drop_duplicates()
            )

        # STEP 3 — SHIFT BOUNDARY DETECTION
        now = datetime.now(timezone.utc)
        new_shift_start = get_shift_start(now)

        if st.session_state.current_shift_start is None:
            st.session_state.current_shift_start = new_shift_start
            logger.info(f"Initialized current_shift_start = {new_shift_start}")

            # -----------------------------------------------------------------
            # FIX #4 — On first init, check if the backfilled buffer already
            # contains a full previous shift worth of data.  If the buffer
            # spans a shift boundary, treat the earlier portion as completed.
            # -----------------------------------------------------------------
            if not st.session_state.online_shift_buffer.empty:
                buf = st.session_state.online_shift_buffer
                buf_start = _ensure_utc(buf.index.min().to_pydatetime())
                buf_shift_start = get_shift_start(buf_start)

                # If the buffer starts in a PREVIOUS shift window, that shift
                # is already complete — split it off and trigger analysis.
                if buf_shift_start < new_shift_start:
                    completed_end = new_shift_start
                    prev_data = buf[buf.index < pd.Timestamp(new_shift_start)]
                    curr_data = buf[buf.index >= pd.Timestamp(new_shift_start)]

                    if not prev_data.empty:
                        logger.info(
                            f"Backfill detected completed shift: "
                            f"{buf_shift_start} → {completed_end}"
                        )
                        st.session_state.completed_shift = {
                            "shift_id": get_shift_id(buf_shift_start),
                            "shift_start": buf_shift_start,
                            "shift_end": completed_end,
                            "df": prev_data.copy(),
                        }
                        st.session_state.online_shift_buffer = (
                            curr_data.copy() if not curr_data.empty else pd.DataFrame()
                        )
                        st.session_state.generated_structured = None
                        st.session_state.generated_llm_text = None
                        st.session_state.shift_waiting_for_operator = False
                        st.session_state.shift_ready_for_analysis = True

        elif new_shift_start > st.session_state.current_shift_start:
            # Normal shift-boundary crossing (app was already running)
            completed_start = st.session_state.current_shift_start
            completed_end = completed_start + timedelta(hours=SHIFT_HOURS)

            logger.info(
                f"Shift boundary crossed: {completed_start} → {completed_end}"
            )

            st.session_state.completed_shift = {
                "shift_id": get_shift_id(completed_start),
                "shift_start": completed_start,
                "shift_end": completed_end,
                "df": st.session_state.online_shift_buffer.copy(),
            }

            # RESET STATE
            st.session_state.generated_structured = None
            st.session_state.generated_llm_text = None
            st.session_state.shift_waiting_for_operator = False

            # Prepare for next shift
            st.session_state.online_shift_buffer = pd.DataFrame()
            st.session_state.current_shift_start = new_shift_start
            st.session_state.shift_ready_for_analysis = True

        # STEP 4 — AUTO SHIFT SUMMARY + FSI
        if st.session_state.shift_ready_for_analysis:
            completed = st.session_state.completed_shift
            shift_df = completed["df"]

            st.subheader("🕒 Completed Shift Detected")
            st.caption(f"{completed['shift_start']} → {completed['shift_end']}")

            llm = OpenRouterClient()
            shift_analyzer = ShiftAnalyzer(settings.anomaly)
            contextual_analyzer = ContextualAnalyzer(llm)
            retriever = ContextRetriever(structured_store, vector_store)

            shift_data = type(
                "ShiftData",
                (),
                {
                    "shift_id": completed["shift_id"],
                    "shift_start": completed["shift_start"],
                    "shift_end": completed["shift_end"],
                    "data": shift_df,
                },
            )()

            # Shift analysis
            llm_text, structured = shift_analyzer.analyze(
                shift_data=shift_data,
                prev_shift_data=None,
                llm=llm,
            )

            # Furnace Stability Index (FSI)
            fsi_calculator = FurnaceStabilityIndex(
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

            fsi_result = fsi_calculator.compute(
                df=shift_df,
                anomaly_count=structured.get("anomaly_count", 0),
            )

            structured["stability_index"] = fsi_result["stability_index"]
            structured["stability_status"] = fsi_result["stability_status"]
            structured["stability_penalties"] = fsi_result["penalties"]

            # Contextual summary
            context = retriever.retrieve_context(
                current_shift_id=shift_data.shift_id,
                current_shift_text=llm_text,
                top_k_similar=3,
            )

            contextual_text, _ = contextual_analyzer.build_day_summary(
                day_id=shift_data.shift_id,
                shift_payloads=[
                    {
                        "shift_name": shift_data.shift_id,
                        "start_time": shift_data.shift_start.isoformat(),
                        "end_time": shift_data.shift_end.isoformat(),
                        "summary_text": llm_text,
                        "stability_index": structured["stability_index"],
                        "stability_status": structured["stability_status"],
                    }
                ],
                previous_shift=context.get("previous_shift"),
                historical_similar=context.get("historical_similar"),
            )

            st.session_state.generated_shift_data = shift_data
            st.session_state.generated_structured = structured
            st.session_state.generated_llm_text = contextual_text

            st.session_state.shift_ready_for_analysis = False
            st.session_state.shift_waiting_for_operator = True

        # STEP 5 — OPERATOR REVIEW & SAVE
        if st.session_state.shift_waiting_for_operator:
            st.subheader("🧠 Shift Summary (Review)")
            show_report("Shift Summary", st.session_state.generated_llm_text)

            with st.form("operator_submit_form"):
                operator_notes = st.text_area("📝 Operator Notes")
                operator_rating = st.slider("⭐ Shift Rating", 1, 5, 3)
                operator_comment = st.text_input("💬 Feedback Comment")
                submit = st.form_submit_button("✅ Submit & Save Shift")

            if submit:
                shift_data = st.session_state.generated_shift_data
                structured = st.session_state.generated_structured
                summary_text = st.session_state.generated_llm_text

                operator_context = {
                    "notes": operator_notes,
                    "feedback": {
                        "rating": operator_rating,
                        "comment": operator_comment,
                    },
                }

                payload = build_shift_payload(
                    shift_data=shift_data,
                    structured_summary=structured,
                    llm_text=summary_text,
                    prev_shift=None,
                    schema=SHIFT_SCHEMA,
                    operator_context=operator_context,
                )

                vector_store.add_window(
                    window_id=payload["window_id"],
                    embedding_text=payload["summary_text"],
                    payload=payload,
                )

                structured_store.save_shift_summary(
                    ShiftSummary(
                        shift_id=shift_data.shift_id,
                        shift_start=shift_data.shift_start,
                        shift_end=shift_data.shift_end,
                        generated_at=datetime.now(timezone.utc),

                        stability_index=structured["stability_index"],
                        stability_status=structured["stability_status"],
                        stability_penalties=structured["stability_penalties"],

                        operator_context=operator_context,
                    )
                )

                run_aggregation_if_ready(
                    new_shift=structured_store.load_shift_summary(shift_data.shift_id),
                    store=structured_store,
                    vector_store=vector_store,
                    schemas={
                        "day": DAY_SCHEMA,
                        "week": WEEK_SCHEMA,
                        "biweek": BIWEEK_SCHEMA,
                    },
                    shifts_per_day=3,
                    days_per_week=7,
                )

                st.success("✅ Shift saved. Aggregation triggered.")
                st.session_state.shift_waiting_for_operator = False

    # ===================================================================
    # 📊 REPORTS — Historical Retrieval
    # ===================================================================
    elif app_mode == "📊 Reports":
        st.header("📊 Historical Reports")

        report_level = st.sidebar.radio(
            "Report Type", ["Shift", "Day", "Week", "Bi-week"]
        )

        fetch_report = False

        if report_level == "Shift":
            selected_date = st.sidebar.date_input("Select date", date.today())
            shift_label = st.sidebar.selectbox("Select shift", ["A", "B", "C"])
            fetch_report = st.sidebar.button("Fetch Report")

        elif report_level == "Day":
            selected_date = st.sidebar.date_input("Select date", date.today())
            fetch_report = st.sidebar.button("Fetch Report")

        elif report_level == "Week":
            selected_week = st.sidebar.text_input("Week window_id (YYYY-MM-DD/YYYY-MM-DD)")
            fetch_report = st.sidebar.button("Fetch Report")

        elif report_level == "Bi-week":
            selected_biweek = st.sidebar.text_input(
                "Bi-week window_id (YYYY-MM-DD/YYYY-MM-DD)"
            )
            fetch_report = st.sidebar.button("Fetch Report")

        if fetch_report:
            if report_level == "Shift":
                window_id = build_shift_window_id(selected_date, shift_label)
            elif report_level == "Day":
                window_id = build_day_window_id(selected_date)
            elif report_level == "Week":
                window_id = f"week_{selected_week}"
            else:
                window_id = f"bi_week_{selected_biweek}"

            payload = fetch_from_qdrant(vector_store, window_id)

            if payload:
                show_report(f"📄 Report ({window_id})", payload["summary_text"])
            else:
                st.warning("No report found.")

    # ===================================================================
    # 🧠 FURNACE INTELLIGENCE — HEALTH OVERVIEW
    # ===================================================================
    else:
        st.header("🧠 Furnace Health Overview")

        # Load latest & all shifts
        latest_shift = structured_store.load_latest_shift_summary()
        all_shifts = structured_store.load_all_shift_summaries() or []

        if latest_shift is None:
            st.warning("No shift summaries found yet. Run shift detection / generate summaries first.")
            st.stop()

        # Sort shifts safely by shift_end
        valid_shifts = [
            s for s in all_shifts
            if getattr(s, "shift_end", None) is not None
        ]

        valid_shifts = sorted(valid_shifts, key=lambda s: s.shift_end)

        prev_shift = valid_shifts[-2] if len(valid_shifts) >= 2 else None

        # Decide which shift to display
        display_shift = None

        if latest_shift and getattr(latest_shift, "stability_index", None) is not None:
            display_shift = latest_shift
        elif prev_shift and getattr(prev_shift, "stability_index", None) is not None:
            display_shift = prev_shift

        # Row 1: Stability Snapshot
        c1, c2, c3 = st.columns(3)

        # Column 1: Stability Index
        with c1:
            if display_shift:
                delta = None
                if (
                    display_shift is latest_shift
                    and prev_shift
                    and prev_shift.stability_index is not None
                ):
                    delta = round(
                        latest_shift.stability_index - prev_shift.stability_index, 1
                    )

                st.metric(
                    label="Furnace Stability Index",
                    value=round(display_shift.stability_index, 1),
                    delta=f"{delta:+}" if delta is not None else None,
                )
            else:
                st.metric("Furnace Stability Index", "—")

        # Column 2: Status
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

        # Column 3: Shift Info
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

        # Row 3: Recurring Anomalies
        st.subheader("🔁 Recurring Anomaly Patterns")

        # Step 1: Load recent historical shifts
        recent_shifts = structured_store.load_last_n_shift_summaries(n=20)

        # Step 2: Run tracker
        tracker = RecurringAnomalyTracker(min_occurrences=3)
        recurring_anomalies = tracker.detect(recent_shifts)

        # Step 3: Render UI
        if recurring_anomalies:
            rows = [
                {
                    "Parameter": param,
                    "Frequency": data["count"],
                    "Pattern": data["pattern"],
                    "Last Seen": data["last_seen"],
                }
                for param, data in recurring_anomalies.items()
            ]

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No recurring anomaly patterns detected.")

        st.divider()

        # Influence Attribution
        st.subheader("🧭 Influence Attribution")

        def classify_influence(index: float) -> str:
            if index >= 0.30:
                return "🔴 Dominant contributor"
            if index >= 0.15:
                return "🟠 Significant contributor"
            if index >= 0.05:
                return "🟡 Moderate contributor"
            return "🟢 Minor contributor"

        attrib = InfluenceAttribution()

        influence_result = attrib.compute(
            shift_summary=latest_shift,
            recurring_anomalies=recurring_anomalies,
        )

        if influence_result:
            df = pd.DataFrame([
                {
                    "Parameter": r["parameter"],
                    "Influence Index": r["influence_index"],
                    "Contribution Level": classify_influence(r["influence_index"]),
                    "Rank": r["rank"],
                }
                for r in influence_result
            ])

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "ℹ️ Influence Index shows relative contribution to instability within this shift. "
                "Higher values indicate stronger contribution compared to other parameters."
            )

        else:
            st.info("No significant contributors identified.")


if __name__ == "__main__":
    main()