from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, date, timedelta, timezone

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from utils.helper_functions_explorer import data_retrieval as dr

from FurnaceMind.utils.settings import settings
from FurnaceMind.utils.logger import get_logger
from FurnaceMind.utils.payload_helpers import build_shift_payload
from FurnaceMind.ui.components import show_report



from FurnaceMind.llm.llm_client import OpenRouterClient
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
)

from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config

config = load_config("setting_ds_dv.yml")  # Load the configuration file

logger = get_logger(__name__)


NAV_TABS = [
    "🤖 AI Co-Operate",
    "📊 Reports",
    "📡 Live Operations",
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

def render_live_operations() -> None:
    """
    Render the live operations tab. This tab monitors and displays live telemetry data.
    """
    st_autorefresh(interval=15 * 60 * 1000, key="online_refresh")
    st.header("📡 Live Operations")

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

    if ui_df is not None and not ui_df.empty:
        # Define ROWS_PER_SHIFT for sizing if needed, or just show the tail.
        # 8 hours * 4 samples/hour = 32 rows if 15min windowing is used.
        ui_df = ui_df.sort_index().tail(32)
        st.subheader("📊 Live Online Data (Last 8 Hours)")
        st.dataframe(ui_df, use_container_width=True)
    else:
        st.info("Waiting for online data…")





def render_reports(*, vector_store) -> None:
    """
    Render the Reports tab, allowing users to fetch and view historical shift handover reports from Qdrant.
    Users select the date and shift label (A, B, C) to retrieve the report summary.
    """
    st.header("📊 Shift Handover Reports")

    with st.sidebar:
        st.divider()
        st.subheader("Report Selection")
        selected_date = st.date_input("Select date", date.today())
        shift_label = st.selectbox("Select shift", ["A", "B", "C"])
        fetch_clicked = st.button("Fetch Report", type="primary", use_container_width=True)

    if fetch_clicked:
        date_str = selected_date.strftime("%Y-%m-%d")
        payload = vector_store.get_report_by_metadata(date_str, shift_label)

        if payload:
            report_text = payload.get("summary", payload.get("summary_text", "No summary content found."))
            show_report(f"📄 Report for {date_str} (Shift {shift_label})", report_text)
        else:
            st.warning(f"No report found for {date_str} Shift {shift_label}.")


def render_ai_cooperate(*, field_labels: dict) -> None:
    """
    Render the AI Co-Operate tab, which provides an LLM-powered assistant for operators.
    The assistant uses live data, historical context, and tool usage to provide actionable insights and recommendations
    to the operator. This function sets up the UI and manages the interaction flow with the LLM and tools.
    Parameters:
    - field_labels: A dictionary mapping internal field keys to human-readable labels, used for displaying data
    """
    st.header("🤖 FurnaceMind — AI Co-Operate")

    AI_COOPERATE_SYSTEM = """
    You are FurnaceMind — AI Co-Operate, an industrial co-pilot that helps humans run manufacturing safely, efficiently, and consistently.

    Mission:
    - Co-operate with the operator/engineer: propose actions, ask for confirmation when actions are risky, and explain trade-offs.
    - Stay grounded in the provided sources (live trends, historic trends from dbs, shift summaries, uploaded documents). Never invent tags, readings, events, or document content.
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
        """
        Read a static context file with a character limit. This is used for loading CLAUDE.md and TOOLS*.md to provide domain and tool usage context to the LLM.
        Parameters:
        - path: Path to the static context file.
        - max_chars: Maximum number of characters to read from the file.
        Returns:
        - A string containing the file content, truncated if necessary, or an empty string if the file doesn't exist or is empty.
        """
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
        """
        Read static context files (CLAUDE.md, TOOLS*.md) to provide domain and tool usage context to the LLM.
        
        Returns:
        - A string containing the combined static context, or an empty string if no context files are found.
        """
        parts: list[str] = []

        # Repo root = .../evonith_webapp
        repo_root = Path(__file__).resolve().parents[3]

        claude_md = _read_static_file(repo_root / "CLAUDE.md", max_chars=24000)
        if claude_md:
            logger.info("Loaded CLAUDE.md with %d characters", len(claude_md))
            parts.append("CLAUDE.md (blast furnace domain context):\n" + claude_md)

        tools_folder = Path(__file__).resolve().parents[1] / "data" / "copilot"
        # Check and load all TOOLS{i}.md files, sorted by i
        tool_files = sorted(tools_folder.glob("TOOLS*.md"), key=lambda p: p.name)
        for tool_file in tool_files:
            tool_md = _read_static_file(tool_file, max_chars=12000)
            if tool_md:
                logger.info("Loaded %s with %d characters", tool_file.name, len(tool_md))
                parts.append(f"{tool_file.name} (available tools + calling rules):\n" + tool_md)

        # Load SKILLS*.md files (benchmark data for skill-based actions)
        skill_files = sorted(tools_folder.glob("SKILLS*.md"), key=lambda p: p.name)
        for skill_file in skill_files:
            skill_md = _read_static_file(skill_file, max_chars=14000)
            if skill_md:
                logger.info("Loaded %s with %d characters", skill_file.name, len(skill_md))
                parts.append(f"{skill_file.name} (skill benchmark data):\n" + skill_md)

        return "\n\n---\n\n".join(parts).strip()

    def _read_recent_tool_errors(max_chars: int = 2500) -> str:
        """
        Read the tail of tool_errors.md to provide recent failure context to the LLM.
        This helps the LLM avoid repeating recent mistakes. We read from the file system each time to capture updates across interactions.
        
        Parameters:
        - max_chars: Maximum number of characters to read from the end of the file.
        Returns:
        - A string containing the recent tool errors, or an empty string if the file doesn't exist or can't be read.
        """
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
        """
        Build the system prompt by combining the AI_COOPERATE_SYSTEM instructions with various context sources.
        Context sources include:
        - Static context from files (CLAUDE.md, TOOLS*.md)
        - Persistent context from memory (conversation summary, do-not-repeat rules)
        - Recent tool errors (tail of tool_errors.md)
        - Any extra context passed in (e.g. from the current conversation)
        The resulting prompt is structured with clear section headers for readability.
        """
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
        """
        Render the latest plot and dataframe artifacts from session_state.
        This is called after each tool execution to refresh the right-hand pane.
        """
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

    # ── Shift selector helpers ──────────────────────────────────────────
    IST = timezone(timedelta(hours=5, minutes=30))

    def _last_completed_shift() -> tuple[date, str]:
        """Return (date, label) of the most recently completed 8-hour shift."""
        now_ist = datetime.now(IST)
        hour = now_ist.hour
        if hour < 8:
            return (now_ist.date() - timedelta(days=1)), "C"
        if hour < 16:
            return now_ist.date(), "A"
        return now_ist.date(), "B"

    def _build_shift_to_best_prompt(shift_date: str, shift_label: str) -> str:
        return (
            f"[SKILL: Shift to Best]\n"
            f"Analyze shift {shift_date} Shift {shift_label} and compare it against the "
            f"best-shift benchmarks from SKILLS_BESTSHIFT.md.\n\n"
            f"Steps:\n"
            f"1. Call load_static_shift_data(shift_date=\"{shift_date}\", shift_label=\"{shift_label}\") "
            f"to load the shift data.\n"
            f"   - If it returns an error about date range, use fetch_online_data and "
            f"fetch_offline_data instead.\n"
            f"2. Compute 8-hour averages for all Tier 1, Tier 2, and Tier 3 parameters "
            f"listed in SKILLS_BESTSHIFT.md.\n"
            f"3. For each parameter, compare the shift average against the best-shift operating band.\n"
            f"4. For parameters in the adverse region, provide: **ACTION** / **REASON** / **MAGNITUDE**.\n"
            f"5. Respect the lag noted for each parameter when interpreting.\n"
            f"6. Produce a concise Shift-to-Best report:\n"
            f"   - Overall gap assessment (how far from best-shift envelope)\n"
            f"   - Top 3 actionable improvements for fuel efficiency\n"
            f"   - Gas flow symmetry check (bosh_quad_spread, uptake_quad_spread from Tier 3)\n"
            f"   - Guardrail status (Tier 3 checks)\n"
            f"7. Call execute_python_plot to generate a comparison bar chart showing "
            f"key Tier 1 + Tier 2 parameters vs best-shift bands.\n"
        )

    default_date, default_label = _last_completed_shift()

    with chat_col:
        for msg in st.session_state.chat_history:
            # Plots are shown in the Artifacts panel, not in the chat.
            if msg.get("type") == "plotly":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("display", msg["content"]))

        # ── Quick-action button bar ─────────────────────────────────────
        st.markdown("---")

        mode_col, date_col, shift_col = st.columns([0.2, 0.4, 0.4])

        with mode_col:
            hist_mode = st.toggle("Historical", key="skill_hist_mode", value=False)

        if hist_mode:
            with date_col:
                selected_date = st.date_input(
                    "Date", value=default_date, key="skill_date",
                    max_value=default_date,
                )
            with shift_col:
                label_options = ["A", "B", "C"]
                selected_label = st.radio(
                    "Shift", label_options, horizontal=True, key="skill_shift",
                    index=label_options.index(default_label),
                )
        else:
            selected_date = default_date
            selected_label = default_label
            with date_col:
                st.caption(f"Last completed: **{default_date}** Shift **{default_label}**")

        btn_cols = st.columns(6)
        with btn_cols[0]:
            if st.button("🎯 Shift to Best", use_container_width=True):
                prompt = _build_shift_to_best_prompt(str(selected_date), selected_label)
                st.session_state["pending_skill_prompt"] = {
                    "prompt": prompt,
                    "display": f"🎯 **Shift to Best**: {selected_date}, Shift {selected_label}",
                }
                st.rerun()
        with btn_cols[1]:
            st.button("📋 Shift Summary", use_container_width=True, disabled=True)
        with btn_cols[2]:
            st.button("⚠️ Channeling", use_container_width=True, disabled=True)
        with btn_cols[3]:
            st.button("🌡️ Heat Balance", use_container_width=True, disabled=True)
        with btn_cols[4]:
            st.button("📦 RM Impact", use_container_width=True, disabled=True)
        with btn_cols[5]:
            st.button("📉 Cost Trend", use_container_width=True, disabled=True)

        st.markdown("---")

        # ── Chat input ──────────────────────────────────────────────────
        typed_query = st.chat_input("Ask about shifts, live trends, documents…")

        # Pending skill prompt takes priority over typed input
        user_query = None
        user_display = None
        if "pending_skill_prompt" in st.session_state:
            pending = st.session_state.pop("pending_skill_prompt")
            user_query = pending["prompt"]
            user_display = pending["display"]
        elif typed_query:
            user_query = typed_query
            user_display = typed_query

        if not user_query:
            return

        st.session_state.chat_history.append(
            {"role": "user", "content": user_query, "display": user_display}
        )
        with st.chat_message("user"):
            st.markdown(user_display)

    # Always route through OpenRouterClient for easy model swapping
    llm = OpenRouterClient()

    tool_policy = (
        "You may call tools. Use tools whenever you need any of: live telemetry, offline reports, knowledge docs, or plots. "
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


