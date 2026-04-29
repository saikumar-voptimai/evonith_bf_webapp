"""render_ai_cooperate — thin orchestrator for the AI Co-Operate tab.

Wires together:
  - SystemPromptContext  (context.py)  — loads static files + memory
  - SkillEngine          (skills.py)   — pre-computes skill analysis
  - render_artifacts_panel (artifacts.py) — Plot + Data tabs
  - run_agent_loop       (agent.py)    — tool-calling LLM loop
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timedelta, timezone

import streamlit as st

from agents.cooperate.agent import run_agent_loop
from agents.cooperate.artifacts import render_artifacts_panel
from agents.cooperate.context import SystemPromptContext
from agents.cooperate.prompts import TOOL_POLICY
from agents.cooperate.skills import SkillEngine
from agents.furnace_tools import get_openai_tool_schemas
from data.db import Database
from embeddings.cloud_embedding import CloudEmbeddingClient
from llm.llm_client import OpenRouterClient
from memory.copilot_memory import add_recent_turn, save_copilot_memory
from memory.knowledge_vector_store import KnowledgeVectorStore
from memory.vector_store import QdrantVectorStore
from multimodal.ingestion import compute_content_hash, process_file, read_uploaded_file_bytes
from utils.settings import settings

_IST = timezone(timedelta(hours=5, minutes=30))


def _last_completed_shift() -> tuple[date, str]:
    """Return (date, label) of the most recently completed 8-hour shift (IST)."""
    now = datetime.now(_IST)
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
        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            msgs.append({"role": role, "content": content})
    return msgs


def _summarize_uploaded_document(filename: str, text: str) -> str:
    """Generate a concise operator-facing summary for an uploaded document."""
    if not text.strip():
        return "Uploaded file indexed for knowledge search. No readable text was extracted."

    sample = text[:12000]
    llm = OpenRouterClient(model=settings.llm.openrouter.ai_cooperate_fast_model_name)
    system_prompt = (
        "You summarize uploaded blast-furnace operating knowledge for future retrieval. "
        "Be concise, factual, and useful to operators and developers."
    )
    user_prompt = f"""
Document: {filename}

Summarize this document in Markdown with:
- Purpose
- Key points
- Operational cautions or dependencies
- When AI Co-Operate should use this knowledge

Document text:
{sample}
"""
    try:
        summary = llm.generate(system_prompt, user_prompt).strip()
    except Exception as exc:
        preview = " ".join(text.split())[:800]
        summary = (
            f"Summary generation failed: {exc}\n\n"
            f"Extracted text preview:\n\n{preview}"
        )

    return summary or "Summary unavailable."


def _summary_download_text(record: dict) -> str:
    """Build a Markdown download body for a knowledge-memory record."""
    return "\n\n".join(
        [
            f"# {record.get('filename', 'Knowledge Document')}",
            f"Uploaded by: {record.get('uploaded_by', 'unknown')}",
            f"Uploaded at: {record.get('uploaded_at', '')}",
            "## Summary",
            record.get("summary") or "",
        ]
    )


def _summary_download_filename(filename: str) -> str:
    """Return a safe summary filename."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("_") or "knowledge"
    return f"{stem}_summary.md"


def _render_knowledge_memory_manager(
    *,
    db: Database,
    knowledge_store: KnowledgeVectorStore,
    current_user: str,
) -> None:
    """Render the Postgres-backed knowledge memory list/actions."""
    st.markdown("### Knowledge Memory")

    records = db.list_knowledge_memory(status="active")
    if not records:
        st.caption("No uploaded knowledge documents yet.")
        return

    def _label(doc_id: str) -> str:
        rec = next((item for item in records if item["doc_id"] == doc_id), None)
        if rec is None:
            return doc_id
        return f"{rec['filename']} · {rec['uploaded_by']}"

    selected_doc_id = st.selectbox(
        "Saved documents",
        [r["doc_id"] for r in records],
        format_func=_label,
        key="knowledge_memory_doc_selector",
    )
    selected = next(r for r in records if r["doc_id"] == selected_doc_id)

    st.caption(
        f"Uploaded by `{selected['uploaded_by']}` · "
        f"{selected['uploaded_at']} · {selected['file_type']}"
    )
    summary_preview = (selected.get("summary") or "").strip()
    st.markdown(summary_preview[:700] + ("..." if len(summary_preview) > 700 else ""))

    st.download_button(
        "Download summary",
        data=_summary_download_text(selected),
        file_name=_summary_download_filename(selected["filename"]),
        mime="text/markdown",
        key=f"download_knowledge_summary_{selected_doc_id}",
        use_container_width=True,
    )

    if st.button(
        "Remove document",
        key=f"remove_knowledge_doc_{selected_doc_id}",
        use_container_width=True,
    ):
        try:
            knowledge_store.delete_points(selected.get("qdrant_point_ids") or [])
            db.remove_knowledge_memory(doc_id=selected_doc_id, removed_by=current_user)
        except Exception as exc:
            st.error(f"Could not remove document: {exc}")
        else:
            st.success("Knowledge document removed.")
            st.rerun()


def render_ai_cooperate(*, field_labels: dict) -> None:  # noqa: ARG001
    """Render the AI Co-Operate tab."""
    st.header("🤖 FurnaceMind — AI Co-Operate")

    with st.sidebar:
        fast_mode = st.toggle("Fast", value=False, key="ai_cooperate_fast_mode")

    model_name = (
        settings.llm.openrouter.ai_cooperate_fast_model_name
        if fast_mode
        else settings.llm.openrouter.ai_cooperate_reasoning_model_name
    )

    # ── Stores ───────────────────────────────────────────────────────────────
    db = Database()
    embedding_client = CloudEmbeddingClient()
    knowledge_store = KnowledgeVectorStore(embedding_client)
    shift_store = QdrantVectorStore()
    st.session_state["knowledge_store"] = knowledge_store
    st.session_state["shift_store"] = shift_store
    current_user = st.session_state.get("auth_user", "unknown")

    # ── Knowledge upload (sidebar) ───────────────────────────────────────────
    with st.sidebar.expander("Knowledge (optional)", expanded=False):
        uploaded = st.file_uploader(
            "Upload Knowledge Files",
            type=[
                "pdf",
                "docx",
                "pptx",
                "xls",
                "xlsx",
                "txt",
                "md",
                "csv",
                "json",
                "log",
                "png",
                "jpg",
                "jpeg",
            ],
            accept_multiple_files=True,
            key="knowledge_uploader",
        )

        st.session_state.setdefault("knowledge_upload_hashes", set())
        if uploaded:
            for f in uploaded:
                try:
                    file_bytes = read_uploaded_file_bytes(f)
                    content_hash = compute_content_hash(file_bytes)
                    upload_key = f"{f.name}:{content_hash}"
                    if upload_key in st.session_state["knowledge_upload_hashes"]:
                        continue

                    existing = db.get_active_knowledge_memory_by_hash(content_hash)
                    if existing:
                        st.info(
                            f"`{f.name}` is already saved by "
                            f"`{existing['uploaded_by']}`."
                        )
                        st.session_state["knowledge_upload_hashes"].add(upload_key)
                        continue

                    with st.spinner(f"Indexing {f.name}..."):
                        ingested = process_file(
                            f,
                            knowledge_store,
                            embedding_client,
                            doc_id=str(uuid.uuid4()),
                            file_bytes=file_bytes,
                        )

                    if ingested is None:
                        st.warning(f"`{f.name}` could not be parsed.")
                        st.session_state["knowledge_upload_hashes"].add(upload_key)
                        continue

                    summary = _summarize_uploaded_document(
                        ingested["filename"], ingested.get("text", "")
                    )
                    try:
                        db.create_knowledge_memory(
                            doc_id=ingested["doc_id"],
                            filename=ingested["filename"],
                            file_type=ingested["file_type"],
                            content_hash=ingested["content_hash"],
                            file_size_bytes=ingested["file_size_bytes"],
                            uploaded_by=current_user,
                            summary=summary,
                            extracted_text_preview=ingested["text_preview"],
                            qdrant_collection=ingested["qdrant_collection"],
                            qdrant_point_ids=ingested["qdrant_point_ids"],
                        )
                    except Exception:
                        knowledge_store.delete_points(ingested["qdrant_point_ids"])
                        raise
                    st.session_state["knowledge_upload_hashes"].add(upload_key)
                    st.success(f"`{f.name}` saved to Knowledge Memory.")
                except Exception as exc:
                    st.error(f"Could not process `{f.name}`: {exc}")

        _render_knowledge_memory_manager(
            db=db,
            knowledge_store=knowledge_store,
            current_user=current_user,
        )

    # ── Layout ───────────────────────────────────────────────────────────────
    chat_col, artifacts_col = st.columns([0.55, 0.45], gap="large")

    with artifacts_col:
        render_artifacts_panel()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Context (loaded once per render) ────────────────────────────────────
    ctx = SystemPromptContext()
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
                    "Date",
                    value=default_date,
                    key="skill_date",
                    max_value=default_date,
                )
            with shift_col:
                label_opts = ["A", "B", "C"]
                selected_label = st.radio(
                    "Shift",
                    label_opts,
                    horizontal=True,
                    key="skill_shift",
                    index=label_opts.index(default_label),
                )
        else:
            selected_date, selected_label = default_date, default_label
            with date_col:
                st.caption(
                    f"Last completed: **{default_date}** Shift **{default_label}**"
                )

        # ── Skill buttons ────────────────────────────────────────────────────
        def _fire_skill(prompt: str, display: str, skill_id: str) -> None:
            """Clear stale artifacts and queue the skill prompt for the next render."""
            st.session_state.pop("copilot_fig", None)
            st.session_state.pop("copilot_df", None)
            st.session_state.pop("copilot_df_meta", None)
            st.session_state["pending_skill_prompt"] = {
                "prompt": prompt,
                "display": display,
                "skill_id": skill_id,
            }
            st.rerun()

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button(
                "💰 Optimise Unit Cost", use_container_width=True):
                _fire_skill(
                    engine.optimise_prompt(),
                    "💰 **Optimise Unit Cost** — analysing last 30 days vs best-shift targets",
                    "optimise",
                )
        with b2:
            if st.button("🎯 Shift to Best", use_container_width=True):
                _fire_skill(
                    engine.shift_to_best_prompt(str(selected_date), selected_label),
                    f"🎯 **Shift to Best**: {selected_date}, Shift {selected_label}",
                    "shift_to_best",
                )
        with b3:
            if st.button("🌡️ Check Heatloads", use_container_width=True):
                _fire_skill(
                    engine.heatload_prompt(),
                    "🌡️ **Check Heatloads** — last 8h vs 2-month baseline",
                    "heatload",
                )

        st.markdown("---")

        # ── Chat input ───────────────────────────────────────────────────────
        typed_query = st.chat_input("Ask about shifts, live trends, documents…")

        user_query = user_display = None
        active_skill_id: str | None = None
        if "pending_skill_prompt" in st.session_state:
            pending = st.session_state.pop("pending_skill_prompt")
            user_query = pending["prompt"]
            user_display = pending["display"]
            active_skill_id = pending.get("skill_id")
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
    llm = OpenRouterClient(model=model_name)
    tools = get_openai_tool_schemas()

    messages: list[dict] = [
        {"role": "system", "content": ctx.build(extra=TOOL_POLICY, skill_id=active_skill_id)},
        *_chat_history_to_messages(),
    ]

    with chat_col:
        with st.chat_message("assistant"):
            status_box = st.empty()
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
    updated_memory = add_recent_turn(
        ctx.memory, user=user_query, assistant=final_response
    )
    save_copilot_memory(updated_memory)
    st.rerun()
