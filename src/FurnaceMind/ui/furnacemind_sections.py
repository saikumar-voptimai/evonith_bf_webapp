from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, date, timedelta, timezone

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from FurnaceMind.utils.settings import settings
from FurnaceMind.utils.logger import get_logger
from FurnaceMind.utils.payload_helpers import build_shift_payload

from FurnaceMind.ui.components import show_report

from FurnaceMind.core.shift_analyzer import ShiftAnalyzer
from FurnaceMind.core.contextual_analyzer import ContextualAnalyzer
from FurnaceMind.core.stability_index import FurnaceStabilityIndex
from FurnaceMind.core.recurring_anomaly_tracker import RecurringAnomalyTracker
from FurnaceMind.core.influence_attribution import InfluenceAttribution

from FurnaceMind.llm.llm_client import OpenRouterClient

from FurnaceMind.memory.retriever import ContextRetriever
from FurnaceMind.memory.schemas import ShiftSummary
from FurnaceMind.memory.aggregation import run_aggregation_if_ready

from FurnaceMind.memory.vector_store import QdrantVectorStore

from FurnaceMind.embeddings.cloud_embedding import CloudEmbeddingClient
from FurnaceMind.memory.knowledge_vector_store import KnowledgeVectorStore
from FurnaceMind.multimodal.ingestion import process_file
from FurnaceMind.agents.furnace_tools import (
    get_openai_tool_schemas,
    execute_openai_tool_call,
)
from FurnaceMind.memory.copilot_memory import (
    load_copilot_memory,
    save_copilot_memory,
    add_recent_turn,
    build_persistent_context,
)

from FurnaceMind.utils.window_helpers import (
    build_day_window_id,
    fetch_from_qdrant,
)

from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config


logger = get_logger(__name__)


NAV_TABS = [
    "🤖 AI Co-Operate",
    "📊 Reports",
    "📡 Live Operations",
    "🧠 Furnace Intelligence",
]


# ---------------------------------------------------------------------------
# Config + schema loading
# ---------------------------------------------------------------------------
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


def load_schemas() -> dict:
    """Load schema JSON files with a robust path for Streamlit Cloud."""
    this_dir = Path(__file__).resolve().parent

    candidate_paths = [
        this_dir.parent / "config",  # src/FurnaceMind/config
        this_dir / "config",  # src/FurnaceMind/ui/config (unlikely)
        this_dir.parents[1] / "FurnaceMind" / "config",  # fallback
    ]

    base = next((p for p in candidate_paths if p.is_dir()), candidate_paths[0])

    return {
        "shift": json.load(open(base / "shift_payload_schema.json")),
        "day": json.load(open(base / "day_payload_schema.json")),
        "week": json.load(open(base / "weekly_payload_schema.json")),
        "biweek": json.load(open(base / "biweekly_payload_schema.json")),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))


def _ensure_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def _normalize_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _chat_history_to_messages(max_messages: int = 16) -> list[dict]:
    """Convert Streamlit chat_history into OpenAI-format messages.

    Plotly messages are omitted from the LLM context.
    """
    msgs: list[dict] = []
    history = st.session_state.get("chat_history") or []
    for m in history[-max_messages:]:
        if m.get("type") == "plotly":
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            msgs.append({"role": role, "content": content})
    return msgs


def select_nav_tab() -> str:
    """Tab-style navigation without executing all tabs."""
    try:
        if hasattr(st, "segmented_control"):
            return st.segmented_control(
                "Navigation",
                NAV_TABS,
                default=NAV_TABS[0],
                key="furnacemind_nav",
            )
    except TypeError:
        pass

    return st.radio(
        "Navigation",
        NAV_TABS,
        horizontal=True,
        index=0,
        key="furnacemind_nav",
    )


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


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_live_operations(*, structured_store, vector_store) -> None:
    schemas = load_schemas()

    SHIFT_HOURS = 8
    WINDOW_MINUTES = 15
    ROWS_PER_SHIFT = (SHIFT_HOURS * 60) // WINDOW_MINUTES

    def get_shift_start(ts: datetime) -> datetime:
        ts = _ensure_ist(ts)
        shift_hour = (ts.hour // SHIFT_HOURS) * SHIFT_HOURS
        return ts.replace(hour=shift_hour, minute=0, second=0, microsecond=0)

    def get_shift_label(shift_start: datetime) -> str:
        hour = _ensure_ist(shift_start).hour
        if hour == 0:
            return "A"
        if hour == 8:
            return "B"
        return "C"

    def get_shift_id(shift_start: datetime) -> str:
        label = get_shift_label(shift_start)
        return f"{shift_start:%Y-%m-%d}_SHIFT_{label}"

    st_autorefresh(interval=WINDOW_MINUTES * 60 * 1000, key="online_refresh")
    st.header("📡 Live Operations — Shift Intelligence")

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
    if "generated_shift_data" not in st.session_state:
        st.session_state.generated_shift_data = None
    if "generated_structured" not in st.session_state:
        st.session_state.generated_structured = None
    if "generated_llm_text" not in st.session_state:
        st.session_state.generated_llm_text = None
    if "generated_contextual_text" not in st.session_state:
        st.session_state.generated_contextual_text = None

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

    if ui_df is None:
        ui_df = pd.DataFrame()

    if ui_df.empty:
        st.info("Waiting for online data…")
        st.stop()

    ui_df = ui_df.sort_index().tail(ROWS_PER_SHIFT)
    st.subheader("📊 Live Online Data (Last 8 Hours)")
    st.dataframe(ui_df, use_container_width=True)

    if st.session_state.online_shift_buffer.empty and not ui_df.empty:
        logger.info("Cold start detected — backfilling shift buffer from UI data.")
        st.session_state.online_shift_buffer = ui_df.copy()

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

    now = datetime.now(IST)
    new_shift_start = get_shift_start(now)

    if st.session_state.current_shift_start is None:
        st.session_state.current_shift_start = new_shift_start
        logger.info(f"Initialized current_shift_start = {new_shift_start}")

        if not st.session_state.online_shift_buffer.empty:
            buf = st.session_state.online_shift_buffer
            buf_start = _ensure_ist(buf.index.min().to_pydatetime())
            buf_shift_start = get_shift_start(buf_start)

            if buf_shift_start < new_shift_start:
                completed_end = new_shift_start
                prev_data = buf[buf.index < pd.Timestamp(new_shift_start)]
                curr_data = buf[buf.index >= pd.Timestamp(new_shift_start)]

                if not prev_data.empty:
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
                    st.session_state.generated_contextual_text = None
                    st.session_state.shift_waiting_for_operator = False
                    st.session_state.shift_ready_for_analysis = True

    elif new_shift_start > st.session_state.current_shift_start:
        completed_start = st.session_state.current_shift_start
        completed_end = completed_start + timedelta(hours=SHIFT_HOURS)

        st.session_state.completed_shift = {
            "shift_id": get_shift_id(completed_start),
            "shift_start": completed_start,
            "shift_end": completed_end,
            "df": st.session_state.online_shift_buffer.copy(),
        }

        st.session_state.generated_structured = None
        st.session_state.generated_llm_text = None
        st.session_state.generated_contextual_text = None
        st.session_state.shift_waiting_for_operator = False

        st.session_state.online_shift_buffer = pd.DataFrame()
        st.session_state.current_shift_start = new_shift_start
        st.session_state.shift_ready_for_analysis = True

    if st.session_state.shift_ready_for_analysis:
        completed = st.session_state.completed_shift
        shift_df = completed["df"]

        st.subheader("🕒 Completed Shift Detected")
        st.caption(f"{completed['shift_start']} → {completed['shift_end']}")

        try:
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

            llm_text, structured = shift_analyzer.analyze(
                shift_data=shift_data,
                prev_shift_data=None,
                llm=llm,
            )

            if llm_text is None or (
                isinstance(llm_text, str) and llm_text.strip().lower() == "none"
            ):
                if isinstance(structured, dict) and structured.get("summary_text"):
                    llm_text = structured["summary_text"]
                elif isinstance(structured, dict) and structured.get("raw_response"):
                    llm_text = structured["raw_response"]

            if isinstance(structured, str):
                structured = {"raw_text": structured}
            if not isinstance(structured, dict):
                structured = {}

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
                shift_payloads=[
                    {
                        "shift_name": shift_data.shift_id,
                        "start_time": shift_data.shift_start.isoformat(),
                        "end_time": shift_data.shift_end.isoformat(),
                        "summary_text": llm_text,
                        "stability_index": structured.get("stability_index"),
                        "stability_status": structured.get("stability_status"),
                    }
                ],
                previous_shift=context.get("previous_shift"),
                historical_similar=context.get("historical_similar"),
            )

            if contextual_text is None or (
                isinstance(contextual_text, str)
                and contextual_text.strip().lower() == "none"
            ):
                contextual_text = llm_text if llm_text else "Contextual summary unavailable."

            st.session_state.generated_shift_data = shift_data
            st.session_state.generated_structured = structured
            st.session_state.generated_llm_text = llm_text
            st.session_state.generated_contextual_text = contextual_text

            st.session_state.shift_ready_for_analysis = False
            st.session_state.shift_waiting_for_operator = True

        except Exception as e:
            st.error(f"❌ Shift analysis failed: {e}")
            st.exception(e)
            st.session_state.shift_ready_for_analysis = False

    if st.session_state.shift_waiting_for_operator:
        llm_text = st.session_state.generated_llm_text
        ctx_text = st.session_state.generated_contextual_text

        def _has_content(text):
            if text is None or not isinstance(text, str):
                return False
            return text.strip() != "" and text.strip().lower() != "none"

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
            operator_notes = st.text_area("📝 Operator Notes")
            operator_rating = st.slider("⭐ Shift Rating", 1, 5, 3)
            operator_comment = st.text_input("💬 Feedback Comment")
            submit = st.form_submit_button("✅ Submit & Save Shift")

        if submit:
            shift_data = st.session_state.generated_shift_data
            structured = st.session_state.generated_structured
            contextual_text = st.session_state.generated_contextual_text

            operator_context = {
                "notes": operator_notes,
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

            structured_store.save_shift_summary(
                ShiftSummary(
                    shift_id=shift_data.shift_id,
                    shift_start=shift_data.shift_start,
                    shift_end=shift_data.shift_end,
                    generated_at=datetime.now(IST),
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
                    "day": schemas["day"],
                    "week": schemas["week"],
                    "biweek": schemas["biweek"],
                },
                shifts_per_day=3,
                days_per_week=7,
            )

            st.success("✅ Shift saved. Aggregation triggered.")
            st.session_state.shift_waiting_for_operator = False


def render_reports(*, vector_store) -> None:
    st.header("📊 Historical Reports")

    report_level = st.sidebar.radio("Report Type", ["Shift", "Day", "Week", "Bi-week"])
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
        selected_biweek = st.sidebar.text_input("Bi-week window_id (YYYY-MM-DD/YYYY-MM-DD)")
        fetch_report = st.sidebar.button("Fetch Report")

    if not fetch_report:
        return

    if report_level == "Shift":
        window_id = f"{selected_date:%Y-%m-%d}_SHIFT_{shift_label}"
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


def render_ai_cooperate(*, field_labels: dict) -> None:
    st.header("🤖 FurnaceMind — AI Co-Operate")

    AI_COOPERATE_SYSTEM = """
    You are FurnaceMind — AI Co-Operate, an industrial co-pilot that helps humans run manufacturing safely, efficiently, and consistently.

    Mission:
    - Co-operate with the operator/engineer: propose actions, ask for confirmation when actions are risky, and explain trade-offs.
    - Stay grounded in the provided sources (live trends, shift summaries, uploaded documents). Never invent tags, readings, events, or document content.
    - Prefer practical guidance: setpoints, checks, thresholds, step-by-step troubleshooting, and “what to do next”.

    How to respond (keep it short and easy to scan):
    - Total length: <= 8 lines unless the user explicitly asks for detail.
    - Use plain language and numbers.
    1) **Conclusion (1 line)**: what’s happening / what to do.
    2) **Actions (max 3 bullets)**: concrete next steps.
    3) **Evidence (max 2 bullets)**: which signals/shift/docs you used.

    Tool & routing discipline:
    - If the question is about live behavior / trends / “last N hours” → use LIVE DATA context.
    - If it’s about what happened in a shift / why performance changed → use SHIFT CONTEXT.
    - If it’s about SOPs / procedures / specs / policies → use DOCUMENT CONTEXT.
    - If context is empty, say so and request the missing artifact.
    Keep the tone professional, concise, and operator-friendly.
    """.strip()

    embedding_client = CloudEmbeddingClient()
    knowledge_store = KnowledgeVectorStore(embedding_client)

    shift_store = QdrantVectorStore()

    # Make stores available to LangChain tools via session_state
    st.session_state["knowledge_store"] = knowledge_store
    st.session_state["shift_store"] = shift_store

    def _read_static_file(path: Path, *, max_chars: int = 20000) -> str:
        try:
            if not path.exists():
                return ""
            txt = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                return ""
            if len(txt) <= max_chars:
                return txt
            return txt[:max_chars].rstrip() + "\n\n[...truncated...]"
        except Exception:
            return ""

    def _read_static_context() -> str:
        parts: list[str] = []

        # Repo root = .../evonith_webapp
        repo_root = Path(__file__).resolve().parents[3]

        claude_md = _read_static_file(repo_root / "CLAUDE.md", max_chars=24000)
        if claude_md:
            parts.append("CLAUDE.md (blast furnace domain context):\n" + claude_md)

        tools_md = _read_static_file(Path(__file__).resolve().parents[1] / "data" / "copilot" / "TOOLS.md", max_chars=12000)
        if tools_md:
            parts.append("TOOLS.md (available tools + calling rules):\n" + tools_md)

        return "\n\n---\n\n".join(parts).strip()

    def _read_recent_tool_errors(max_chars: int = 2500) -> str:
        try:
            tool_errors_path = Path(__file__).resolve().parents[1] / "agents" / "tool_errors.md"
            if not tool_errors_path.exists():
                return ""
            txt = tool_errors_path.read_text(encoding="utf-8")
            return txt[-max_chars:].strip()
        except Exception:
            return ""

    copilot_memory = load_copilot_memory()
    persistent_context = build_persistent_context(copilot_memory)
    tool_errors_tail = _read_recent_tool_errors()
    static_context = _read_static_context()

    def _build_system_prompt(extra_context: str = "") -> str:
        parts = [AI_COOPERATE_SYSTEM]
        if static_context:
            parts.append("STATIC CONTEXT (read this before answering):\n" + static_context)
        if persistent_context:
            parts.append(persistent_context)
        if tool_errors_tail:
            parts.append("RECENT TOOL ERRORS (avoid repeating these failure modes):\n" + tool_errors_tail)
        if extra_context:
            parts.append(extra_context.strip())
        return "\n\n".join(parts).strip()

    file_types = ["pdf", "docx", "pptx", "xls", "xlsx", "txt"]
    with st.sidebar.expander("Knowledge (optional)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload Knowledge Files",
            type=[str(x) for x in file_types],
            accept_multiple_files=True,
            key="knowledge_uploader",
        )
        status = st.empty()
        if uploaded_files:
            for f in uploaded_files:
                process_file(f, knowledge_store, embedding_client)
            status.success("Documents indexed successfully.")

    chat_col, artifacts_col = st.columns([0.55, 0.45], gap="large")

    with artifacts_col:
        st.subheader("Artifacts")
        st.markdown("### Plot")
        plot_placeholder = st.empty()
        st.markdown("### Data")
        df_placeholder = st.empty()

    def _render_artifacts() -> None:
        fig = st.session_state.get("copilot_fig")
        if fig is not None:
            plot_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                key="furnacemind_artifact_plot",
            )
        else:
            plot_placeholder.info("No plot yet.")

        df = st.session_state.get("copilot_df")
        if df is not None:
            df_placeholder.dataframe(
                df,
                use_container_width=True,
                key="furnacemind_artifact_df",
            )
        else:
            df_placeholder.info("No dataframe yet.")

    _render_artifacts()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with chat_col:
        for msg in st.session_state.chat_history:
            # Plots are shown in the Artifacts panel, not in the chat.
            if msg.get("type") == "plotly":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask about shifts, live trends, documents…")
        if not user_query:
            return

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

    # Always route through OpenRouterClient for easy model swapping
    llm = OpenRouterClient()

    tool_policy = (
        "You may call tools. Use tools whenever you need any of: live telemetry, offline reports, shift history, knowledge docs, or plots. "
        "Never guess numeric values.\n\n"
        "DATA POLICIES:\n"
        "- Online telemetry max lookback: 90 days.\n"
        "- Online averaging default: if lookback > 1 day => 1 hour; else 15 minutes (unless user explicitly asks otherwise).\n"
        "- Offline averaging defaults: HM/Slag & Charge hourly (1h), Raw material composition shiftwise (8h), DPR daily (1d).\n"
        "- When merging: repeat/forward-fill offline onto online timestamps.\n\n"
        "OFFLINE REPORT TYPES:\n"
        "- HM_SLAG, CHARGE, RAW_MATERIAL_COMPOSITION (Bunker Report), DPR\n"
    )

    tools = get_openai_tool_schemas()

    # Tool-calling loop (no regex routing)
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(tool_policy)},
        *_chat_history_to_messages(max_messages=14),
    ]

    final_response = ""
    last_tool_name = None
    last_tool_result = None

    with chat_col:
        with st.chat_message("assistant"):
            with st.spinner("Thinking / running tools…"):
                for _ in range(4):
                    completion = llm.chat_completions(messages=messages, tools=tools, tool_choice="auto")
                    msg = completion.choices[0].message

                    tool_calls = getattr(msg, "tool_calls", None)
                    content = getattr(msg, "content", None) or ""

                    if tool_calls:
                        # Add assistant tool-call message to the transcript
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in tool_calls
                                ],
                            }
                        )

                        for tc in tool_calls:
                            last_tool_name = tc.function.name
                            try:
                                args = json.loads(tc.function.arguments or "{}")
                            except Exception:
                                args = {}
                            result = execute_openai_tool_call(name=tc.function.name, arguments=args)
                            last_tool_result = result
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "name": tc.function.name,
                                    "content": result,
                                }
                            )
                        continue

                    final_response = content.strip()
                    break

        # Tools may have updated df/fig; refresh the artifacts pane.
        with artifacts_col:
            _render_artifacts()

        if not final_response:
            # Fallback: show last tool result if the model didn't produce a final message
            final_response = last_tool_result or "No response generated."

        st.markdown(final_response)
        response = final_response

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        copilot_memory = add_recent_turn(copilot_memory, user=user_query, assistant=response)
        save_copilot_memory(copilot_memory)


def render_furnace_intelligence(*, structured_store) -> None:
    st.header("🧠 Furnace Intelligence")

    latest_shift = structured_store.load_latest_shift_summary()
    all_shifts = structured_store.load_all_shift_summaries() or []

    if latest_shift is None:
        st.warning("No shift summaries found yet. Run shift detection / generate summaries first.")
        st.stop()

    valid_shifts = [s for s in all_shifts if getattr(s, "shift_end", None) is not None]
    valid_shifts = sorted(valid_shifts, key=lambda s: s.shift_end)
    prev_shift = valid_shifts[-2] if len(valid_shifts) >= 2 else None

    display_shift = None
    if latest_shift and getattr(latest_shift, "stability_index", None) is not None:
        display_shift = latest_shift
    elif prev_shift and getattr(prev_shift, "stability_index", None) is not None:
        display_shift = prev_shift

    c1, c2, c3 = st.columns(3)

    with c1:
        if display_shift:
            delta = None
            if display_shift is latest_shift and prev_shift and prev_shift.stability_index is not None:
                delta = round(latest_shift.stability_index - prev_shift.stability_index, 1)
            st.metric(
                label="Furnace Stability Index",
                value=round(display_shift.stability_index, 1),
                delta=f"{delta:+}" if delta is not None else None,
            )
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
    recent_shifts = structured_store.load_last_n_shift_summaries(n=20)
    tracker = RecurringAnomalyTracker(min_occurrences=3)
    recurring_anomalies = tracker.detect(recent_shifts)

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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("No recurring anomaly patterns detected.")

    st.divider()

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
    influence_result = attrib.compute(shift_summary=latest_shift, recurring_anomalies=recurring_anomalies)

    if influence_result:
        df = pd.DataFrame(
            [
                {
                    "Parameter": r["parameter"],
                    "Influence Index": r["influence_index"],
                    "Contribution Level": classify_influence(r["influence_index"]),
                    "Rank": r["rank"],
                }
                for r in influence_result
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "ℹ️ Influence Index shows relative contribution to instability within this shift. "
            "Higher values indicate stronger contribution compared to other parameters."
        )
    else:
        st.info("No significant contributors identified.")
