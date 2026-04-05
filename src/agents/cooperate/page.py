"""render_ai_cooperate — thin orchestrator for the AI Co-Operate tab.

Wires together:
  - SystemPromptContext  (context.py)  — loads static files + memory
  - SkillEngine          (skills.py)   — pre-computes skill analysis
  - render_artifacts_panel (artifacts.py) — Plot + Data tabs
  - run_agent_loop       (agent.py)    — tool-calling LLM loop
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import streamlit as st

from agents.furnace_tools import get_openai_tool_schemas
from embeddings.cloud_embedding import CloudEmbeddingClient
from llm.llm_client import OpenRouterClient
from memory.copilot_memory import add_recent_turn, save_copilot_memory
from memory.knowledge_vector_store import KnowledgeVectorStore
from memory.vector_store import QdrantVectorStore
from multimodal.ingestion import process_file

from agents.cooperate.agent     import run_agent_loop
from agents.cooperate.artifacts import render_artifacts_panel
from agents.cooperate.context   import SystemPromptContext
from agents.cooperate.prompts   import TOOL_POLICY
from agents.cooperate.skills    import SkillEngine

_IST = timezone(timedelta(hours=5, minutes=30))


def _last_completed_shift() -> tuple[date, str]:
    """Return (date, label) of the most recently completed 8-hour shift (IST)."""
    now  = datetime.now(_IST)
    hour = now.hour
    if hour < 8:
        return (now.date() - timedelta(days=1)), "C"
    if hour < 16:
        return now.date(), "A"
    return now.date(), "B"


def _chat_history_to_messages(max_messages: int = 14) -> list[dict]:
    """Convert Streamlit chat_history to OpenAI-format messages (plotly entries excluded)."""
    msgs: list[dict] = []
    for m in (st.session_state.get("chat_history") or [])[-max_messages:]:
        if m.get("type") == "plotly":
            continue
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            msgs.append({"role": role, "content": content})
    return msgs


def render_ai_cooperate(*, field_labels: dict) -> None:  # noqa: ARG001
    """Render the AI Co-Operate tab."""
    st.header("🤖 FurnaceMind — AI Co-Operate")

    # ── Stores ───────────────────────────────────────────────────────────────
    embedding_client = CloudEmbeddingClient()
    knowledge_store  = KnowledgeVectorStore(embedding_client)
    shift_store      = QdrantVectorStore()
    st.session_state["knowledge_store"] = knowledge_store
    st.session_state["shift_store"]     = shift_store

    # ── Knowledge upload (sidebar) ───────────────────────────────────────────
    with st.sidebar.expander("Knowledge (optional)", expanded=False):
        uploaded = st.file_uploader(
            "Upload Knowledge Files",
            type=["pdf", "docx", "pptx", "xls", "xlsx", "txt"],
            accept_multiple_files=True,
            key="knowledge_uploader",
        )
        upload_status = st.empty()
        if uploaded:
            for f in uploaded:
                process_file(f, knowledge_store, embedding_client)
            upload_status.success("Documents indexed successfully.")

    # ── Layout ───────────────────────────────────────────────────────────────
    chat_col, artifacts_col = st.columns([0.55, 0.45], gap="large")

    with artifacts_col:
        render_artifacts_panel()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Context (loaded once per render) ────────────────────────────────────
    ctx    = SystemPromptContext()
    engine = SkillEngine()

    default_date, default_label = _last_completed_shift()

    with chat_col:
        # Render existing messages
        for msg in st.session_state.chat_history:
            if msg.get("type") == "plotly":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("display", msg["content"]))

        st.markdown("---")

        # ── Shift selector ───────────────────────────────────────────────────
        mode_col, date_col, shift_col = st.columns([0.2, 0.4, 0.4])
        with mode_col:
            hist_mode = st.toggle("Historical", key="skill_hist_mode", value=False)

        if hist_mode:
            with date_col:
                selected_date = st.date_input(
                    "Date", value=default_date, key="skill_date", max_value=default_date,
                )
            with shift_col:
                label_opts    = ["A", "B", "C"]
                selected_label = st.radio(
                    "Shift", label_opts, horizontal=True, key="skill_shift",
                    index=label_opts.index(default_label),
                )
        else:
            selected_date, selected_label = default_date, default_label
            with date_col:
                st.caption(f"Last completed: **{default_date}** Shift **{default_label}**")

        # ── Skill buttons ────────────────────────────────────────────────────
        def _fire_skill(prompt: str, display: str) -> None:
            """Clear stale artifacts and queue the skill prompt for the next render."""
            st.session_state.pop("copilot_fig",     None)
            st.session_state.pop("copilot_df",      None)
            st.session_state.pop("copilot_df_meta", None)
            st.session_state["pending_skill_prompt"] = {"prompt": prompt, "display": display}
            st.rerun()

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("💰 Optimise Unit Cost", use_container_width=True, type="primary"):
                _fire_skill(
                    engine.optimise_prompt(),
                    "💰 **Optimise Unit Cost** — analysing last 30 days vs best-shift targets",
                )
        with b2:
            if st.button("🎯 Shift to Best", use_container_width=True):
                _fire_skill(
                    engine.shift_to_best_prompt(str(selected_date), selected_label),
                    f"🎯 **Shift to Best**: {selected_date}, Shift {selected_label}",
                )
        with b3:
            if st.button("🌡️ Check Heatloads", use_container_width=True):
                _fire_skill(
                    engine.heatload_prompt(),
                    "🌡️ **Check Heatloads** — last 8h vs 2-month baseline",
                )

        st.markdown("---")

        # ── Chat input ───────────────────────────────────────────────────────
        typed_query = st.chat_input("Ask about shifts, live trends, documents…")

        user_query = user_display = None
        if "pending_skill_prompt" in st.session_state:
            pending      = st.session_state.pop("pending_skill_prompt")
            user_query   = pending["prompt"]
            user_display = pending["display"]
        elif typed_query:
            user_query = user_display = typed_query

        if not user_query:
            return

        st.session_state.chat_history.append(
            {"role": "user", "content": user_query, "display": user_display}
        )
        with st.chat_message("user"):
            st.markdown(user_display)

    # ── Agent loop ───────────────────────────────────────────────────────────
    llm   = OpenRouterClient()
    tools = get_openai_tool_schemas()

    messages: list[dict] = [
        {"role": "system", "content": ctx.build(extra=TOOL_POLICY)},
        *_chat_history_to_messages(),
    ]

    with chat_col:
        with st.chat_message("assistant"):
            status_box   = st.empty()
            response_box = st.empty()
            status_box.status("Thinking…", expanded=False)

            final_response = run_agent_loop(
                llm=llm,
                messages=messages,
                tools=tools,
                status_box=status_box,
                response_box=response_box,
            )

    # ── Persist and rerun ────────────────────────────────────────────────────
    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_response, "display": final_response}
    )
    updated_memory = add_recent_turn(ctx.memory, user=user_query, assistant=final_response)
    save_copilot_memory(updated_memory)
    st.rerun()
