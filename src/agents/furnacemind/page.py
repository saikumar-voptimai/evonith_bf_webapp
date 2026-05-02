"""Streamlit page for FurnaceMind AI Co-Operate."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from agents.embeddings.cloud_embedding import CloudEmbeddingClient
from agents.furnace_tools import get_openai_tool_schemas
from agents.furnacemind.agent import run_agent_loop
from agents.furnacemind.artifacts import render_artifacts_panel
from agents.furnacemind.context import SystemPromptContext
from agents.furnacemind.prompts import TOOL_POLICY
from agents.furnacemind.skills import SkillEngine
from agents.llm.llm_client import OpenRouterClient
from agents.llm.model_modes import (
    REASONING_EFFORT_OPTIONS,
    configured_default_reasoning_effort,
    configured_model_name,
    normalize_reasoning_effort,
    reasoning_effort_label,
)
from agents.memory.context_budget import build_context_budget, estimate_text_tokens
from agents.memory.conversation_history import ConversationHistoryStore
from agents.memory.fm_memory import add_recent_turn, save_fm_memory
from agents.memory.knowledge_vector_store import KnowledgeVectorStore
from agents.memory.semantic_memory import (
    SemanticMemoryStore,
    build_feedback_lessons_context,
    build_long_term_memory_context,
)
from agents.memory.vector_store import QdrantVectorStore
from agents.multimodal.ingestion import extract_text_from_file, process_file

_IST = timezone(timedelta(hours=5, minutes=30))


@st.cache_resource(show_spinner=False)
def _get_conversation_store() -> ConversationHistoryStore:
    """Return a cached PostgreSQL conversation store."""
    return ConversationHistoryStore()


def _current_user_id() -> str:
    """Return the current Streamlit user identity."""
    return str(st.session_state.get("auth_user") or "anonymous").strip() or "anonymous"


def _load_store() -> ConversationHistoryStore | None:
    """Load the conversation store and show a sidebar warning on failure."""
    try:
        return _get_conversation_store()
    except Exception as exc:
        st.sidebar.warning(f"Chat database unavailable: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def _get_semantic_memory_store() -> SemanticMemoryStore:
    """Return a cached Qdrant-backed semantic memory store."""
    return SemanticMemoryStore()


def _load_semantic_memory_store() -> SemanticMemoryStore | None:
    """Load semantic memory and show a sidebar warning on failure."""
    try:
        return _get_semantic_memory_store()
    except Exception as exc:
        st.sidebar.warning(f"Semantic memory unavailable: {exc}")
        return None


def _last_completed_shift() -> tuple[date, str]:
    """Return the most recently completed 8-hour shift in IST."""
    now = datetime.now(_IST)
    if now.hour < 8:
        return (now.date() - timedelta(days=1)), "C"
    if now.hour < 16:
        return now.date(), "A"
    return now.date(), "B"


def _clear_loaded_conversation() -> None:
    """Clear session state tied to the currently loaded conversation."""
    st.session_state.chat_history = []
    st.session_state.pop("_fm_hydrated_conversation_id", None)
    st.session_state.pop("_fm_reasoning_conversation_id", None)


def _ensure_conversation(
    *,
    store: ConversationHistoryStore | None,
    user_id: str,
) -> str | None:
    """Ensure a persisted conversation exists for the current user."""
    if store is None:
        return None
    conversation_id = st.session_state.get("fm_conversation_id")
    try:
        conversation_id = store.get_or_create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            title="FurnaceMind chat",
        )
    except Exception as exc:
        st.sidebar.warning(f"Could not load conversation: {exc}")
        return None
    st.session_state["fm_conversation_id"] = conversation_id
    return conversation_id


def _conversation_label(conversation: dict) -> str:
    """Build a compact label for a conversation selector option."""
    title = str(conversation.get("title") or "FurnaceMind chat").strip()
    updated_at = conversation.get("updated_at")
    if updated_at is None:
        return title
    try:
        stamp = updated_at.astimezone(_IST).strftime("%d %b %H:%M")
    except Exception:
        stamp = str(updated_at)
    return f"{title} - {stamp}"


def _render_conversation_selector(
    *,
    store: ConversationHistoryStore | None,
    user_id: str,
) -> str | None:
    """Render new-chat and chat-history controls."""
    if store is None:
        return None
    if st.sidebar.button("New Chat", key="fm_new_chat", width="stretch"):
        st.session_state["fm_conversation_id"] = store.create_conversation(
            user_id=user_id
        )
        _clear_loaded_conversation()
        st.rerun()

    conversation_id = _ensure_conversation(store=store, user_id=user_id)
    if conversation_id is None:
        return None

    conversations = store.list_conversations(user_id=user_id)
    option_ids = [item["conversation_id"] for item in conversations]
    labels = {item["conversation_id"]: _conversation_label(item) for item in conversations}
    if conversation_id not in option_ids:
        option_ids.insert(0, conversation_id)
        labels[conversation_id] = "Current chat"

    selected_id = st.sidebar.selectbox(
        "Chat History",
        option_ids,
        index=option_ids.index(conversation_id),
        format_func=lambda item_id: labels.get(item_id, item_id),
        key="fm_conversation_selector",
    )
    if selected_id != conversation_id:
        st.session_state["fm_conversation_id"] = selected_id
        _clear_loaded_conversation()
        st.rerun()
    return selected_id


def _hydrate_chat_history(
    *,
    store: ConversationHistoryStore | None,
    conversation_id: str | None,
) -> None:
    """Load persisted messages into session state once per conversation."""
    if store is None or conversation_id is None:
        return
    if st.session_state.get("_fm_hydrated_conversation_id") == conversation_id:
        return
    try:
        st.session_state.chat_history = store.load_chat_history(
            conversation_id=conversation_id
        )
        st.session_state["_fm_hydrated_conversation_id"] = conversation_id
    except Exception as exc:
        st.sidebar.warning(f"Could not restore chat history: {exc}")


def _render_reasoning_selector(
    *,
    store: ConversationHistoryStore | None,
    conversation_id: str | None,
    user_id: str,
) -> str:
    """Render and persist the reasoning effort selector."""
    default_effort = configured_default_reasoning_effort()
    if store is not None and conversation_id is not None:
        try:
            default_effort = normalize_reasoning_effort(
                store.get_conversation_model_mode(
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
            )
        except Exception as exc:
            st.sidebar.warning(f"Could not load reasoning effort: {exc}")

    widget_key = f"fm_reasoning_effort_{conversation_id or 'new'}"
    selected = normalize_reasoning_effort(
        st.sidebar.radio(
            "Reasoning effort",
            REASONING_EFFORT_OPTIONS,
            index=REASONING_EFFORT_OPTIONS.index(default_effort),
            format_func=reasoning_effort_label,
            key=widget_key,
        )
    )
    if selected != default_effort and store is not None and conversation_id is not None:
        store.update_conversation_model_mode(
            conversation_id=conversation_id,
            user_id=user_id,
            model_mode=selected,
        )
    st.sidebar.caption(f"Model: `{configured_model_name() or 'configured default'}`")
    return selected


def _render_memory_manager(
    *,
    store: ConversationHistoryStore | None,
    user_id: str,
    knowledge_store: KnowledgeVectorStore,
    embedding_client: CloudEmbeddingClient,
) -> None:
    """Render knowledge upload and indexed document controls."""
    with st.sidebar.expander("Knowledge (optional)", expanded=False):
        uploaded = st.file_uploader(
            "Upload Knowledge Files",
            type=["pdf", "docx", "pptx", "xls", "xlsx", "txt", "md"],
            accept_multiple_files=True,
            key="knowledge_uploader",
        )
        if st.button("Index Documents", key="fm_index_documents", width="stretch"):
            if not uploaded:
                st.warning("Choose one or more files first.")
            elif store is None:
                st.warning("Document database is unavailable.")
            else:
                for uploaded_file in uploaded:
                    _index_memory_document(
                        store=store,
                        user_id=user_id,
                        uploaded_file=uploaded_file,
                        knowledge_store=knowledge_store,
                        embedding_client=embedding_client,
                    )
                st.success("Documents indexed.")
                st.rerun()
        _render_document_list(store=store, user_id=user_id)


def _index_memory_document(
    *,
    store: ConversationHistoryStore,
    user_id: str,
    uploaded_file,
    knowledge_store: KnowledgeVectorStore,
    embedding_client: CloudEmbeddingClient,
) -> None:
    """Index one uploaded knowledge document and save SQL metadata."""
    file_bytes = uploaded_file.getvalue()
    file_path = _save_uploaded_file(user_id=user_id, filename=uploaded_file.name, data=file_bytes)
    ingest_file = BytesIO(file_bytes)
    ingest_file.name = uploaded_file.name
    indexed = process_file(ingest_file, knowledge_store, embedding_client)
    store.create_document(
        user_id=user_id,
        filename=uploaded_file.name,
        file_type=indexed.get("file_type") or _file_extension(uploaded_file.name),
        file_path=str(file_path),
        summary=f"Indexed with {indexed.get('chunk_count', 0)} chunk(s).",
        qdrant_collection=indexed.get("qdrant_collection"),
        qdrant_point_ids=indexed.get("qdrant_point_ids") or [],
        token_estimate=indexed.get("token_estimate"),
        metadata={"chunk_count": indexed.get("chunk_count", 0)},
    )


def _save_uploaded_file(*, user_id: str, filename: str, data: bytes) -> Path:
    """Save uploaded bytes under local storage and return the path."""
    directory = Path("src/storage/memory_documents") / _safe_path_part(user_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}_{_safe_path_part(filename)}"
    path.write_bytes(data)
    return path


def _render_document_list(
    *,
    store: ConversationHistoryStore | None,
    user_id: str,
) -> None:
    """Render indexed memory documents."""
    if store is None:
        return
    try:
        documents = store.list_documents(user_id=user_id)
    except Exception as exc:
        st.warning(f"Could not load documents: {exc}")
        return
    if not documents:
        st.caption("No indexed documents yet.")
        return
    for document in documents:
        with st.container(border=True):
            st.markdown(f"**{document['filename']}**")
            st.caption(document.get("summary") or document.get("file_type") or "Document")
            if st.button(
                "Deactivate",
                key=f"fm_deactivate_doc_{document['document_id']}",
                width="stretch",
            ):
                store.deactivate_document(document_id=document["document_id"])
                st.rerun()


def _render_skill_manager(
    *,
    store: ConversationHistoryStore | None,
    user_id: str,
    semantic_memory_store: SemanticMemoryStore | None,
) -> None:
    """Render uploaded skills and activation controls."""
    with st.sidebar.expander("Skills", expanded=False):
        if store is None:
            st.warning("Skill database is unavailable.")
            return
        try:
            store.seed_default_skills()
            skills = store.list_skills()
        except Exception as exc:
            st.warning(f"Could not load skills: {exc}")
            return

        _sync_skills_to_vector_store(
            semantic_memory_store=semantic_memory_store,
            skills=skills,
            user_id=user_id,
        )
        _render_skill_upload(store=store, user_id=user_id)
        st.markdown("---")
        for skill in skills:
            _render_skill_card(store=store, skill=skill)


def _sync_skills_to_vector_store(
    *,
    semantic_memory_store: SemanticMemoryStore | None,
    skills: list[dict],
    user_id: str,
) -> None:
    """Upsert SQL skills into Qdrant for semantic skill detection."""
    if semantic_memory_store is None:
        return
    indexed_keys = st.session_state.setdefault("_fm_indexed_skill_vector_keys", set())
    for skill in skills:
        fingerprint = "|".join(
            str(skill.get(key) or "")
            for key in (
                "skill_id",
                "name",
                "description",
                "instruction",
                "is_active",
            )
        )
        if fingerprint in indexed_keys:
            continue
        try:
            semantic_memory_store.upsert_skill(skill=skill, user_id=user_id)
            indexed_keys.add(fingerprint)
        except Exception as exc:
            st.sidebar.warning(f"Could not index skill vectors: {exc}")
            return


def _render_skill_upload(*, store: ConversationHistoryStore, user_id: str) -> None:
    """Render upload flow for new skill files."""
    uploaded = st.file_uploader(
        "Upload Skill Files",
        type=["pdf", "docx", "pptx", "txt", "md"],
        accept_multiple_files=True,
        key="fm_skill_uploader",
    )
    if not st.button("Add Uploaded Skills", key="fm_add_uploaded_skills", width="stretch"):
        return
    if not uploaded:
        st.warning("Choose one or more skill files first.")
        return
    created = 0
    for uploaded_file in uploaded:
        file_bytes = uploaded_file.getvalue()
        skill_file = BytesIO(file_bytes)
        skill_file.name = uploaded_file.name
        file_type, instruction = extract_text_from_file(skill_file)
        if not instruction.strip():
            st.warning(f"{uploaded_file.name} did not contain readable text.")
            continue
        store.create_uploaded_skill(
            user_id=user_id,
            filename=uploaded_file.name,
            file_type=file_type,
            instruction=instruction,
        )
        created += 1
    if created:
        st.success(f"Added {created} skill(s).")
        st.rerun()


def _render_skill_card(*, store: ConversationHistoryStore, skill: dict) -> None:
    """Render one skill with an activate/deactivate button."""
    active = bool(skill.get("is_active"))
    action = "Deactivate" if active else "Activate"
    with st.container(border=True):
        st.markdown(f"**{skill.get('button_label') or skill.get('name')}**")
        st.caption(f"{skill.get('source_type', 'custom').title()} - {'Active' if active else 'Inactive'}")
        if skill.get("description"):
            st.caption(skill["description"])
        if st.button(action, key=f"fm_skill_toggle_{skill['skill_id']}", width="stretch"):
            store.update_skill(skill_id=skill["skill_id"], is_active=not active)
            st.rerun()


def _load_button_skills(store: ConversationHistoryStore | None) -> list[dict]:
    """Load active button skills, falling back to static defaults on failure."""
    if store is None:
        return []
    try:
        store.seed_default_skills()
        return store.list_button_skills()
    except Exception as exc:
        st.sidebar.warning(f"Could not load skill buttons: {exc}")
        return []


def _load_active_skills(store: ConversationHistoryStore | None) -> list[dict]:
    """Load active skills for automatic detection."""
    if store is None:
        return []
    try:
        store.seed_default_skills()
        return [skill for skill in store.list_skills() if skill.get("is_active")]
    except Exception:
        return []


def _detect_skill(query: str, skills: list[dict]) -> dict | None:
    """Return the best matching skill for a typed query using simple word overlap."""
    query_words = {word.lower() for word in query.replace("_", " ").split() if len(word) > 3}
    best_skill: dict | None = None
    best_score = 0
    for skill in skills:
        text = " ".join(
            str(skill.get(key) or "")
            for key in ("name", "description", "instruction")
        ).lower()
        score = sum(1 for word in query_words if word in text)
        if score > best_score:
            best_score = score
            best_skill = skill
    return best_skill if best_score > 0 else None


def _detect_skill_semantic(
    *,
    query: str,
    skills: list[dict],
    semantic_memory_store: SemanticMemoryStore | None,
    user_id: str,
) -> dict | None:
    """Return the best matching active skill using Qdrant semantic search."""
    if semantic_memory_store is None:
        return None
    threshold = float(os.getenv("FURNACEMIND_SKILL_MATCH_THRESHOLD", "0.45"))
    by_id = {str(skill.get("skill_id")): skill for skill in skills if skill.get("skill_id")}
    try:
        results = semantic_memory_store.search_skills(
            user_id=user_id,
            query=query,
            top_k=5,
        )
    except Exception as exc:
        st.sidebar.warning(f"Could not search skill vectors: {exc}")
        return None
    for item in results:
        if float(item.get("score") or 0) < threshold:
            continue
        payload = item.get("payload") or {}
        skill = by_id.get(str(payload.get("skill_id")))
        if skill and skill.get("is_active"):
            return skill
    return None


def _run_button_skill(
    *,
    engine: SkillEngine,
    skill: dict,
    selected_date: date,
    selected_label: str,
) -> tuple[str, str, str]:
    """Return prompt, display text, and skill context ID for a button skill."""
    handler = (skill.get("metadata") or {}).get("handler")
    if handler == "optimise":
        return engine.optimise_prompt(), "💰 **Unit Cost**", "optimise"
    if handler == "shift_to_best":
        return (
            engine.shift_to_best_prompt(str(selected_date), selected_label),
            f"🎯 **Shift to Best**: {selected_date}, Shift {selected_label}",
            "shift_to_best",
        )
    if handler == "heatload":
        return engine.heatload_prompt(), "🌡️ **Heatloads**", "heatload"
    label = skill.get("button_label") or skill.get("name") or "Skill"
    return skill.get("instruction") or label, f"**{label}**", skill.get("skill_id") or "custom"


def _load_knowledge_context(
    *,
    store: ConversationHistoryStore | None,
    user_id: str,
    query: str,
    knowledge_store: KnowledgeVectorStore,
) -> tuple[str, list[dict]]:
    """Retrieve active uploaded document snippets for the prompt."""
    if store is None or not query.strip():
        return "", []
    try:
        documents = store.list_documents(user_id=user_id)
        active_point_ids = {
            str(point_id)
            for document in documents
            for point_id in (document.get("qdrant_point_ids") or [])
        }
        active_sources = {document["filename"] for document in documents}
        if not active_point_ids and not active_sources:
            return "", []
        results = knowledge_store.search(query, top_k=8)
    except Exception as exc:
        st.sidebar.warning(f"Could not retrieve documents: {exc}")
        return "", []
    if active_point_ids:
        selected = [item for item in results if str(item.get("id")) in active_point_ids][:4]
    else:
        selected = [
            item for item in results if (item.get("payload") or {}).get("source") in active_sources
        ][:4]
    if not selected:
        return "", []
    lines = ["UPLOADED KNOWLEDGE CONTEXT:"]
    sources: list[dict] = []
    for item in selected:
        payload = item.get("payload") or {}
        source = payload.get("source") or "document"
        content = str(payload.get("content") or "")[:1200]
        lines.append(f"- Source: {source}\n{content}")
        sources.append({"filename": source, "score": item.get("score")})
    return "\n\n".join(lines), sources


def _chat_history_to_messages(max_messages: int = 14) -> list[dict]:
    """Convert session chat history to OpenAI-compatible messages."""
    messages: list[dict] = []
    for item in (st.session_state.get("chat_history") or [])[-max_messages:]:
        if item.get("type") == "plotly":
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    return messages


def _feedback_snapshot(message_id: str) -> dict:
    """Build a compact feedback snapshot for one assistant message."""
    history = st.session_state.get("chat_history") or []
    target_index = next(
        (index for index, item in enumerate(history) if item.get("message_id") == message_id),
        len(history) - 1,
    )
    return {
        "recent_messages": [
            {"role": item.get("role"), "content": item.get("content")}
            for item in history[max(0, target_index - 4) : target_index + 1]
        ]
    }


def _previous_user_message(message_id: str) -> str:
    """Return the latest user message before an assistant message."""
    history = st.session_state.get("chat_history") or []
    target_index = next(
        (index for index, item in enumerate(history) if item.get("message_id") == message_id),
        len(history) - 1,
    )
    for item in reversed(history[:target_index]):
        if item.get("role") == "user":
            return str(item.get("content") or "")
    return ""


def _last_assistant_message() -> dict | None:
    """Return the latest persisted assistant message from chat history."""
    for item in reversed(st.session_state.get("chat_history") or []):
        if item.get("role") == "assistant" and item.get("message_id"):
            return item
    return None


def _looks_like_chat_feedback(text: str) -> bool:
    """Return True when a chat message appears to critique the prior response."""
    lowered = f" {text.lower()} "
    direct_markers = (
        " feedback ",
        " your answer ",
        " your response ",
        " that answer ",
        " this answer ",
        " previous answer ",
        " next time ",
    )
    correction_markers = (
        " wrong",
        " incorrect",
        " not useful",
        " not helpful",
        " don't ",
        " do not ",
        " should have ",
        " should be ",
        " looks good",
        " good response",
    )
    return any(marker in lowered for marker in direct_markers) and any(
        marker in lowered for marker in correction_markers
    )


def _chat_feedback_polarity(text: str) -> str:
    """Infer feedback polarity from a free-text chat correction."""
    lowered = text.lower()
    if any(word in lowered for word in ("wrong", "incorrect", "not useful", "not helpful", "don't", "do not")):
        return "negative"
    if any(word in lowered for word in ("looks good", "good response", "helpful", "correct")):
        return "positive"
    return "neutral"


def _capture_chat_feedback(
    *,
    store: ConversationHistoryStore | None,
    user_id: str,
    text: str,
) -> None:
    """Persist free-text feedback when the user critiques the prior answer."""
    if store is None or not _looks_like_chat_feedback(text):
        return
    target = _last_assistant_message()
    if target is None:
        return
    try:
        store.add_feedback(
            user_id=user_id,
            source="chat",
            polarity=_chat_feedback_polarity(text),
            message_id=target.get("message_id"),
            conversation_id=target.get("conversation_id"),
            feedback_text=text,
            raw_user_message=_previous_user_message(str(target.get("message_id"))),
            prev_assistant_message=target.get("content"),
            snapshot=_feedback_snapshot(str(target.get("message_id"))),
        )
    except Exception as exc:
        st.sidebar.warning(f"Could not save chat feedback: {exc}")


def _render_feedback_controls(
    *,
    store: ConversationHistoryStore | None,
    message: dict,
    user_id: str,
) -> None:
    """Render like/dislike feedback controls for an assistant message."""
    message_id = message.get("message_id")
    if not message_id:
        return
    existing = None
    if store is not None:
        try:
            existing = store.get_feedback(message_id=message_id, user_id=user_id)
        except Exception:
            existing = None
    if existing:
        st.caption(f"Feedback captured: {existing['polarity']}")
        return
    cols = st.columns(2)
    _render_one_feedback_form(
        col=cols[0],
        store=store,
        message=message,
        user_id=user_id,
        polarity="positive",
        label="👍 Looks good",
    )
    _render_one_feedback_form(
        col=cols[1],
        store=store,
        message=message,
        user_id=user_id,
        polarity="negative",
        label="👎 Needs work",
    )


def _render_one_feedback_form(
    *,
    col,
    store: ConversationHistoryStore | None,
    message: dict,
    user_id: str,
    polarity: str,
    label: str,
) -> None:
    """Render one collapsible feedback form."""
    message_id = message.get("message_id")
    with col:
        with st.expander(label, expanded=False):
            with st.form(f"fm_feedback_{polarity}_{message_id}"):
                comment = st.text_area("Comment", max_chars=1000)
                submitted = st.form_submit_button("Save Feedback")
            if submitted:
                if store is None:
                    st.warning("Feedback database is unavailable.")
                    return
                store.add_feedback(
                    user_id=user_id,
                    source="form",
                    polarity=polarity,
                    message_id=message_id,
                    conversation_id=message.get("conversation_id"),
                    feedback_text=comment,
                    raw_user_message=_previous_user_message(message_id),
                    prev_assistant_message=message.get("content"),
                    snapshot=_feedback_snapshot(message_id),
                )
                st.rerun()


def _estimate_base_prompt_tokens(ctx: SystemPromptContext) -> int:
    """Estimate non-chat prompt tokens sent before any user conversation."""
    return estimate_text_tokens(ctx.build(extra=TOOL_POLICY))


def _estimate_tool_schema_tokens(tools: list[dict]) -> int:
    """Estimate tokens used by tool schemas in the chat-completion request."""
    return estimate_text_tokens(str(tools or []))


def _render_context_status(*, ctx: SystemPromptContext, tools: list[dict]) -> None:
    """Render the context usage status bar."""
    budget = build_context_budget(
        chat_messages=st.session_state.get("chat_history") or [],
        base_tokens=_estimate_base_prompt_tokens(ctx),
        tools_tokens=_estimate_tool_schema_tokens(tools),
    )
    status_col, button_col = st.columns([0.72, 0.28])
    with status_col:
        st.caption(
            f"Context {budget['total_tokens']:,}/{budget['budget_tokens']:,} est. tokens - {budget['status']}"
        )
        st.progress(budget["percent"] / 100)
    with button_col:
        st.session_state["_fm_manual_compress_clicked"] = st.button(
            "Compress Memory",
            key="fm_compress_memory",
            width="stretch",
        )
    cols = st.columns(6)
    cols[0].caption(f"Base: {budget['base_tokens']}")
    cols[1].caption(f"Chat: {budget['chat_tokens']}")
    cols[2].caption(f"Memory: {budget['memory_tokens']}")
    cols[3].caption(f"Feedback: {budget['feedback_tokens']}")
    cols[4].caption(f"Tools: {budget['tools_tokens']}")
    cols[5].caption(f"Docs: {budget['docs_tokens']}")


def _compress_current_conversation(
    *,
    store: ConversationHistoryStore | None,
    conversation_id: str | None,
    user_id: str,
) -> bool:
    """Compress older persisted turns and reload the active chat buffer."""
    if store is None or conversation_id is None:
        st.warning("Conversation database is unavailable.")
        return False
    try:
        compression_llm = OpenRouterClient(reasoning_effort="low")
    except Exception:
        compression_llm = None
    try:
        summary = store.compress_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            llm_client=compression_llm,
        )
    except Exception as exc:
        st.warning(f"Could not compress memory: {exc}")
        return False
    if not summary:
        st.info("Not enough chat history to compress yet.")
        return False
    st.session_state.chat_history = store.load_chat_history(conversation_id=conversation_id)
    return True


def _auto_compress_if_needed(
    *,
    store: ConversationHistoryStore | None,
    conversation_id: str | None,
    user_id: str,
    ctx: SystemPromptContext,
    tools: list[dict],
) -> None:
    """Automatically compress old turns when context usage reaches the threshold."""
    budget = build_context_budget(
        chat_messages=st.session_state.get("chat_history") or [],
        base_tokens=_estimate_base_prompt_tokens(ctx),
        tools_tokens=_estimate_tool_schema_tokens(tools),
    )
    marker = f"{conversation_id}:{budget['total_tokens']}"
    if budget["status"] != "Compressing":
        return
    if st.session_state.get("_fm_last_auto_compress_marker") == marker:
        return
    st.session_state["_fm_last_auto_compress_marker"] = marker
    if _compress_current_conversation(
        store=store,
        conversation_id=conversation_id,
        user_id=user_id,
    ):
        st.toast("Older chat compressed into memory.")
        st.rerun()


def _file_extension(filename: str) -> str:
    """Return a lowercase filename extension without the dot."""
    return Path(filename).suffix.lower().lstrip(".")


def _safe_path_part(value: str) -> str:
    """Return a filesystem-safe path fragment."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe or "file"


def render_ai_cooperate(*, field_labels: dict) -> None:  # noqa: ARG001
    """Render the AI Co-Operate tab."""
    st.header("🤖 FurnaceMind - AI Co-Operate")

    embedding_client = CloudEmbeddingClient()
    knowledge_store = KnowledgeVectorStore(embedding_client)
    shift_store = QdrantVectorStore()
    st.session_state["knowledge_store"] = knowledge_store
    st.session_state["shift_store"] = shift_store

    user_id = _current_user_id()
    store = _load_store()
    semantic_memory_store = _load_semantic_memory_store()
    conversation_id = _render_conversation_selector(store=store, user_id=user_id)
    reasoning_effort = _render_reasoning_selector(
        store=store,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    _hydrate_chat_history(store=store, conversation_id=conversation_id)
    _render_memory_manager(
        store=store,
        user_id=user_id,
        knowledge_store=knowledge_store,
        embedding_client=embedding_client,
    )
    _render_skill_manager(
        store=store,
        user_id=user_id,
        semantic_memory_store=semantic_memory_store,
    )
    if store is not None:
        try:
            store.process_pending_feedback_lessons(
                semantic_memory_store=semantic_memory_store,
                lesson_llm_client=OpenRouterClient(reasoning_effort="low"),
                limit=3,
            )
        except Exception as exc:
            st.sidebar.warning(f"Could not process feedback lessons: {exc}")

    chat_col, artifacts_col = st.columns([0.55, 0.45], gap="large")
    with artifacts_col:
        render_artifacts_panel()

    tools = get_openai_tool_schemas()
    ctx = SystemPromptContext()
    _auto_compress_if_needed(
        store=store,
        conversation_id=conversation_id,
        user_id=user_id,
        ctx=ctx,
        tools=tools,
    )
    engine = SkillEngine()
    button_skills = _load_button_skills(store)
    active_skills = _load_active_skills(store)
    default_date, default_label = _last_completed_shift()

    with chat_col:
        _render_context_status(ctx=ctx, tools=tools)
        if st.session_state.pop("_fm_manual_compress_clicked", False):
            if _compress_current_conversation(
                store=store,
                conversation_id=conversation_id,
                user_id=user_id,
            ):
                st.rerun()
        st.markdown("---")

        for message in st.session_state.chat_history:
            if message.get("type") == "plotly":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message.get("display") or message.get("content"))
                if message.get("role") == "assistant":
                    _render_feedback_controls(store=store, message=message, user_id=user_id)

        st.markdown("---")
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
                labels = ["A", "B", "C"]
                selected_label = st.radio(
                    "Shift",
                    labels,
                    horizontal=True,
                    key="skill_shift",
                    index=labels.index(default_label),
                )
        else:
            selected_date, selected_label = default_date, default_label
            with date_col:
                st.caption(f"Last completed: **{default_date}** Shift **{default_label}**")

        def _fire_skill(prompt: str, display: str, skill_id: str) -> None:
            """Queue a skill prompt for the next render."""
            st.session_state.pop("fm_fig", None)
            st.session_state.pop("fm_df", None)
            st.session_state.pop("fm_df_meta", None)
            st.session_state["pending_skill_prompt"] = {
                "prompt": prompt,
                "display": display,
                "skill_id": skill_id,
                "skill_name": display,
                "source_type": "button",
            }
            st.rerun()

        if button_skills:
            columns = st.columns(min(3, len(button_skills)))
            for index, skill in enumerate(button_skills):
                with columns[index % len(columns)]:
                    if st.button(skill["button_label"], key=f"fm_skill_{skill['skill_id']}", width="stretch"):
                        prompt, display, skill_id = _run_button_skill(
                            engine=engine,
                            skill=skill,
                            selected_date=selected_date,
                            selected_label=selected_label,
                        )
                        st.session_state["_fm_pending_button_skill"] = {
                            "name": skill.get("name"),
                            "source_type": skill.get("source_type"),
                        }
                        _fire_skill(prompt, display, skill_id)

        st.markdown("---")
        typed_query = st.chat_input("Ask about shifts, live trends, documents...")

        user_query = user_display = None
        active_skill_id: str | None = None
        selected_skill: dict | None = None
        auto_skill_context = ""
        if "pending_skill_prompt" in st.session_state:
            pending = st.session_state.pop("pending_skill_prompt")
            user_query = pending["prompt"]
            user_display = pending["display"]
            active_skill_id = pending.get("skill_id")
            pending_button_skill = st.session_state.pop("_fm_pending_button_skill", {})
            selected_skill = {
                "skill_id": active_skill_id,
                "name": pending_button_skill.get("name") or pending.get("skill_name"),
                "source_type": pending_button_skill.get("source_type") or pending.get("source_type"),
                "selection": "button",
            }
        elif typed_query:
            user_query = user_display = typed_query
            _capture_chat_feedback(store=store, user_id=user_id, text=typed_query)
            detected = _detect_skill_semantic(
                query=user_query,
                skills=active_skills,
                semantic_memory_store=semantic_memory_store,
                user_id=user_id,
            ) or _detect_skill(user_query, active_skills)
            if detected:
                active_skill_id = (detected.get("metadata") or {}).get("handler") or detected.get("skill_id")
                selected_skill = {
                    "skill_id": detected.get("skill_id"),
                    "name": detected.get("name"),
                    "source_type": detected.get("source_type"),
                    "selection": "auto",
                    "prompt_skill_id": active_skill_id,
                }
                auto_skill_context = (
                    "AUTO-DETECTED SKILL INSTRUCTION:\n"
                    + str(detected.get("instruction") or "")
                )

        if not user_query:
            return

        user_message_id = None
        if store is not None and conversation_id is not None:
            user_message_id = store.add_user_message(
                conversation_id=conversation_id,
                user_id=user_id,
                content=user_query,
                display=user_display,
            )
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_query,
                "display": user_display,
                "message_id": user_message_id,
                "conversation_id": conversation_id,
            }
        )
        with st.chat_message("user"):
            st.markdown(user_display)

    knowledge_context, knowledge_sources = _load_knowledge_context(
        store=store,
        user_id=user_id,
        query=user_query,
        knowledge_store=knowledge_store,
    )
    feedback_lessons_context = ""
    long_term_memory_context = ""
    if semantic_memory_store is not None:
        try:
            feedback_lessons_context = build_feedback_lessons_context(
                semantic_memory_store.search_lessons(user_id=user_id, query=user_query)
            )
        except Exception as exc:
            st.sidebar.warning(f"Could not retrieve feedback lessons: {exc}")
        try:
            long_term_memory_context = build_long_term_memory_context(
                semantic_memory_store.search_long_term_memories(
                    user_id=user_id,
                    query=user_query,
                )
            )
        except Exception as exc:
            st.sidebar.warning(f"Could not retrieve long-term memory: {exc}")
    extra_parts = [
        TOOL_POLICY,
        knowledge_context,
        auto_skill_context,
        feedback_lessons_context,
        long_term_memory_context,
    ]
    messages = [
        {
            "role": "system",
            "content": ctx.build(
                extra="\n\n".join(part for part in extra_parts if part),
                skill_id=active_skill_id,
            ),
        },
        *_chat_history_to_messages(),
    ]

    llm = OpenRouterClient(reasoning_effort=reasoning_effort)
    with chat_col:
        with st.chat_message("assistant"):
            status_box = st.empty()
            response_box = st.empty()
            status_box.status("Thinking...", expanded=False)
            final_response = run_agent_loop(
                llm=llm,
                messages=messages,
                tools=tools,
                status_box=status_box,
                response_box=response_box,
            )

    assistant_message_id = None
    if store is not None and conversation_id is not None:
        assistant_metadata = {"knowledge_sources": knowledge_sources}
        if selected_skill:
            assistant_metadata["selected_skill"] = selected_skill
        assistant_message_id = store.add_assistant_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=final_response,
            model=getattr(llm, "model", None),
            metadata=assistant_metadata,
        )
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": final_response,
            "display": final_response,
            "message_id": assistant_message_id,
            "conversation_id": conversation_id,
            "knowledge_sources": knowledge_sources,
            "selected_skill": selected_skill,
        }
    )
    if store is not None:
        try:
            store.store_turn_long_term_memories(
                user_id=user_id,
                user_text=user_query,
                assistant_text=final_response,
                semantic_memory_store=semantic_memory_store,
                memory_llm_client=OpenRouterClient(reasoning_effort="low"),
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
            )
        except Exception as exc:
            st.sidebar.warning(f"Could not store long-term memory: {exc}")
    save_fm_memory(add_recent_turn(ctx.memory, user=user_query, assistant=final_response))
    st.rerun()
