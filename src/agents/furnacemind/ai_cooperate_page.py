"""Streamlit page renderer for the FurnaceMind AI Co-Operate experience.

This module wires together the FurnaceMind chat UI, PostgreSQL conversation
persistence, rolling memory summaries, skill shortcuts, vector stores, and the
agent loop. It intentionally keeps page orchestration here while reusable UI
pieces live under ``ui.furnacemind`` and persistence logic lives under
``agents.memory``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import streamlit as st

import agents.furnace_tools as furnace_tools
from agents.embeddings.cloud_embedding import CloudEmbeddingClient
from agents.furnacemind import prompts
from agents.furnacemind.agent import run_agent_loop
from agents.furnacemind.context import SystemPromptContext
from agents.furnacemind.skill_registry import SkillRegistry
from agents.furnacemind.skill_vector_store import SkillVectorStore
from agents.furnacemind.skills import SkillEngine
from agents.llm.llm_client import OpenRouterClient
from agents.memory import fm_memory
from agents.memory.conversation_history import ConversationHistoryStore
from agents.memory.knowledge_vector_store import KnowledgeVectorStore
from agents.memory.semantic_memory import SemanticMemoryService
from agents.memory.vector_store import QdrantVectorStore
from furnace_data import relational
from ui.furnacemind import chat_interface, feedback_flow
from utils.furnacemind.feedback_service import FurnaceMindFeedbackService
from utils.session import current_user_id
from utils.settings import (
    REASONING_LEVELS,
    normalize_openrouter_reasoning_level,
    settings,
)
from utils.shift_windows import last_completed_shift


@dataclass(frozen=True)
class MemoryFlushResult:
    """
    Report the outcome of a memory-summary flush.

    The page uses this to decide whether New Chat can safely reset the visible
    conversation after persisting summary and semantic-memory data.
    """

    attempted: bool = False
    summary_saved: bool = False
    semantic_fact_ids: tuple[str, ...] = ()
    error: str | None = None


@st.cache_resource(show_spinner=False)
def _cached_embedding_client() -> CloudEmbeddingClient:
    """
    Build or reuse the embedding client used by FurnaceMind knowledge features.

    Streamlit reruns this page after every chat submission, so the embedding
    client is cached as a resource to avoid rebuilding provider clients on every
    render. The returned client is shared by document search and knowledge
    sidebar operations during the current app process.

    Args:
         - None

    Returns:
         - return: CloudEmbeddingClient - Singleton embedding client instance.
    """
    return CloudEmbeddingClient()


@st.cache_resource(show_spinner=False)
def _cached_knowledge_store() -> KnowledgeVectorStore:
    """
    Build or reuse the knowledge vector store for uploaded FurnaceMind context.

    The store is backed by the cached embedding client and is used by the
    sidebar upload/search flow. Keeping it cached prevents repeated vector-store
    setup work during Streamlit reruns while still exposing the same store to the
    chat submission handler.

    Args:
         - None

    Returns:
         - return: KnowledgeVectorStore - Singleton knowledge vector store.
    """
    return KnowledgeVectorStore(_cached_embedding_client())


@st.cache_resource(show_spinner=False)
def _cached_shift_store() -> QdrantVectorStore:
    """
    Build or reuse the Qdrant store used for shift-history retrieval.

    FurnaceMind tools can query shift-history embeddings through this store.
    Caching keeps the Qdrant client stable across page reruns and avoids
    reconnecting when the user sends another message.

    Args:
         - None

    Returns:
         - return: QdrantVectorStore - Singleton shift vector store.
    """
    return QdrantVectorStore()


@st.cache_resource(show_spinner=False)
def _cached_context() -> SystemPromptContext:
    """
    Build or reuse the system prompt context assembler.

    The context object loads static prompt files and refreshes per-conversation
    memory before each answer. It is cached so expensive static file reads happen
    once, while ``refresh_session_context`` still reloads memory and tool-error
    state for the active conversation.

    Args:
         - None

    Returns:
         - return: SystemPromptContext - Singleton prompt context.
    """
    return SystemPromptContext()


@st.cache_resource(show_spinner=False)
def _cached_skill_engine() -> SkillEngine:
    """
    Build or reuse the FurnaceMind quick-skill engine.

    The skill engine owns the quick skill definitions displayed above the chat
    input. Caching keeps the skill registry stable across reruns and prevents
    repeated initialization when Streamlit refreshes after each message.

    Args:
         - None

    Returns:
         - return: SkillEngine - Singleton FurnaceMind skill engine.
    """
    return SkillEngine()


@st.cache_resource(show_spinner=False)
def _cached_skill_vector_store() -> SkillVectorStore | None:
    """
    Build the Qdrant-backed semantic index for FurnaceMind skills.

    The app can still run without this store: selected skills are injected from
    SQL directly, and the registry can fall back to runtime embedding if Qdrant
    is unavailable. When available, this store is used for persisted skill
    retrieval and sidebar create/edit indexing.
    """
    try:
        return SkillVectorStore(_cached_embedding_client())
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_skill_repository() -> Any | None:
    """
    Build the PostgreSQL repository for database-driven quick skills.

    Skill configuration is optional at runtime. When PostgreSQL is unavailable or
    the skills table is empty, ``SkillRegistry`` falls back to the built-in skill
    definitions so the existing FurnaceMind buttons continue to work.
    """
    try:
        engine = relational.build_relational_engine()
        session_factory = relational.build_relational_session_factory(engine)
        return relational.SkillRepository(session_factory)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_history_store() -> ConversationHistoryStore | None:
    """
    Build the PostgreSQL conversation-history adapter when the database is ready.

    This function is intentionally tolerant of database connection failures. If
    PostgreSQL is unavailable, the page continues with session-only chat history
    and shows a sidebar notice instead of breaking the FurnaceMind UI.

    Args:
         - None

    Returns:
         - return: ConversationHistoryStore | None - Store instance when available.
    """
    try:
        return ConversationHistoryStore()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_document_repository() -> Any | None:
    """
    Build the PostgreSQL metadata repository for uploaded MRAG documents.

    The upload flow remains usable when PostgreSQL is unavailable; in that case
    Qdrant indexing still runs and document metadata persistence is skipped.
    """
    try:
        engine = relational.build_relational_engine()
        session_factory = relational.build_relational_session_factory(engine)
        return relational.MemoryDocumentRepository(session_factory)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_chunk_repository() -> Any | None:
    """
    Build the PostgreSQL chunk metadata repository for MRAG document parts.

    The repository reflects the live ``memory_chunks`` table, so it can be
    enabled without adding a strict ORM model for every deployed schema variant.
    """
    try:
        engine = relational.build_relational_engine()
        session_factory = relational.build_relational_session_factory(engine)
        return relational.MemoryChunkRepository(session_factory)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_retrieval_trace_repository() -> Any | None:
    """
    Build the PostgreSQL repository for MRAG retrieval traces when available.

    Trace persistence is best-effort; the chat page continues normally if the
    table is absent or the database is unavailable.
    """
    try:
        engine = relational.build_relational_engine()
        session_factory = relational.build_relational_session_factory(engine)
        return relational.RetrievalTraceRepository(session_factory)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_feedback_service() -> FurnaceMindFeedbackService | None:
    """
    Build the feedback service used for SQL and Qdrant lesson persistence.

    Feedback should never break the chat page. If the database, embedding
    provider, or Qdrant is unavailable during startup, the page continues
    without feedback persistence and shows a sidebar notice when needed.

    Args:
         - None

    Returns:
         - return: FurnaceMindFeedbackService | None - Feedback service when ready.
    """
    try:
        return FurnaceMindFeedbackService(embedding_client=_cached_embedding_client())
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_feedback_llm() -> OpenRouterClient | None:
    """
    Build the LLM client used for feedback detection and lesson extraction.

    Feedback work uses the configured memory-compression model so lightweight
    classification and lesson generation stay separate from the main answering
    model while reusing the same OpenRouter connection settings.

    Args:
         - None

    Returns:
         - return: OpenRouterClient | None - LLM client for feedback helper tasks.
    """
    try:
        return OpenRouterClient(
            model_name=settings.llm.openrouter.memory_compression_model_name
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _cached_semantic_memory_service() -> SemanticMemoryService:
    """
    Build or reuse the SQL-backed, Qdrant-indexed semantic memory service.

    The service is failure-tolerant: when PostgreSQL, Qdrant, the extraction
    LLM, or embedding credentials are unavailable it reports itself as
    unavailable and FurnaceMind continues with the rolling conversation summary.

    Args:
         - None

    Returns:
         - return: SemanticMemoryService - Cached semantic memory adapter.
    """
    return SemanticMemoryService()


def _flush_conversation_memory(
    *,
    chat_history: list[dict[str, Any]],
    memory: dict[str, Any],
    user_id: str | None,
    conversation_id: str | None,
    semantic_memory_service: SemanticMemoryService | None,
    memory_summary_window: int,
    memory_summary_token_limit: int,
    trigger: str,
    force: bool = False,
    retry_saved_summary: bool = False,
    memory_llm: Any | None = None,
) -> MemoryFlushResult:
    """
    Generate, save, and semantic-index the next due conversation summary.

    This is shared by the normal summary window and the explicit New Chat
    finalization path. A forced call includes a short leftover tail so the last
    messages in a conversation are not missed when the user starts a new chat.

    Args:
         - chat_history: list[dict[str, Any]] - Current Streamlit chat messages.
         - memory: dict[str, Any] - Latest saved cumulative summary payload.
         - user_id: str | None - Current FurnaceMind user id.
         - conversation_id: str | None - Active conversation id.
         - semantic_memory_service: SemanticMemoryService | None - Long-term memory service.
         - memory_summary_window: int - Regular summary window size.
         - memory_summary_token_limit: int - Requested summary token limit.
         - trigger: str - Event that caused the flush, such as ``window`` or ``new_chat``.
         - force: bool - Whether to summarize a partial trailing window.
         - retry_saved_summary: bool - Whether to retry semantic extraction for saved memory.
         - memory_llm: Any | None - Optional test override for the summary LLM.

    Returns:
         - return: MemoryFlushResult - Summary/fact persistence outcome.
    """
    if not user_id or not conversation_id:
        return MemoryFlushResult()
    if not fm_memory.should_generate_memory_summary(
        chat_history,
        window=memory_summary_window,
        memory=memory,
        force=force,
    ):
        if retry_saved_summary:
            return _index_saved_summary_memory(
                memory=memory,
                user_id=user_id,
                conversation_id=conversation_id,
                semantic_memory_service=semantic_memory_service,
                memory_summary_window=memory_summary_window,
                trigger=f"{trigger}_retry",
            )
        return MemoryFlushResult()

    pending_start, pending_end = fm_memory.summary_source_message_ids(
        chat_history,
        window=memory_summary_window,
        memory=memory,
        force=force,
    )
    memory_llm = memory_llm or OpenRouterClient(
        model_name=settings.llm.openrouter.memory_compression_model_name
    )
    updated_memory = fm_memory.generate_memory_summary(
        memory,
        chat_history=chat_history,
        llm=memory_llm,
        summary_system_prompt=prompts.memory_summary_system_prompt(
            memory_summary_token_limit
        ),
        summary_token_limit=memory_summary_token_limit,
        window=memory_summary_window,
        force=force,
    )

    generated_end = str(updated_memory.get("source_message_id_end") or "").strip()
    if pending_end and generated_end != pending_end:
        return MemoryFlushResult(
            attempted=True,
            error="Conversation summary did not complete. Please retry New Chat.",
        )

    updated_summary = str(updated_memory.get("conversation_summary") or "").strip()
    if not updated_summary:
        return MemoryFlushResult(
            attempted=True,
            error="Conversation summary was empty. Please retry New Chat.",
        )

    source_message_id_start = str(
        updated_memory.get("source_message_id_start") or pending_start or ""
    ).strip()
    source_message_id_end = str(
        updated_memory.get("source_message_id_end") or pending_end or ""
    ).strip()
    summary_saved = fm_memory.save_fm_memory(
        updated_memory,
        user_id=user_id,
        conversation_id=conversation_id,
        source_message_id_start=source_message_id_start or None,
        source_message_id_end=source_message_id_end or None,
    )
    if not summary_saved:
        return MemoryFlushResult(
            attempted=True,
            error="Conversation summary could not be saved. Please retry New Chat.",
        )

    fact_ids: list[str] = []
    if semantic_memory_service is not None and updated_summary:
        if not semantic_memory_service.storage_available:
            error = (
                semantic_memory_service.last_error
                or "Semantic memory storage is unavailable."
            )
            return MemoryFlushResult(
                attempted=True,
                summary_saved=True,
                error=f"Long-term memory could not start: {error}",
            )
        semantic_error_before = semantic_memory_service.last_error
        fact_ids = semantic_memory_service.add_summary(
            user_id=user_id,
            conversation_id=conversation_id,
            summary=updated_summary,
            source_message_id_start=source_message_id_start or None,
            source_message_id_end=source_message_id_end or None,
            summarized_message_count=updated_memory.get("summarized_message_count"),
            metadata={
                "summary_trigger": trigger,
                "memory_summary_window": memory_summary_window,
            },
        )
        semantic_error_after = semantic_memory_service.last_error
        if not fact_ids and semantic_error_after != semantic_error_before:
            return MemoryFlushResult(
                attempted=True,
                summary_saved=True,
                error=f"Long-term memory failed: {semantic_error_after}",
            )

    return MemoryFlushResult(
        attempted=True,
        summary_saved=True,
        semantic_fact_ids=tuple(fact_ids),
    )


def _index_saved_summary_memory(
    *,
    memory: dict[str, Any],
    user_id: str,
    conversation_id: str,
    semantic_memory_service: SemanticMemoryService | None,
    memory_summary_window: int,
    trigger: str,
) -> MemoryFlushResult:
    """
    Retry semantic extraction for an already-saved conversation summary.

    This supports the New Chat retry path. If the final cumulative summary was
    saved but semantic extraction or Qdrant indexing failed, the next New Chat
    click can retry long-term memory from the saved summary instead of treating
    the conversation as already complete.

    Args:
         - memory: dict[str, Any] - Latest saved cumulative summary payload.
         - user_id: str - Current FurnaceMind user id.
         - conversation_id: str - Active conversation id.
         - semantic_memory_service: SemanticMemoryService | None - Long-term memory service.
         - memory_summary_window: int - Regular summary window size.
         - trigger: str - Event label stored in semantic-memory metadata.

    Returns:
         - return: MemoryFlushResult - Semantic retry outcome.
    """
    summary = str(memory.get("conversation_summary") or "").strip()
    if not summary:
        return MemoryFlushResult()
    if semantic_memory_service is None:
        return MemoryFlushResult(
            attempted=True,
            summary_saved=True,
            error="Long-term memory service is unavailable.",
        )
    if not semantic_memory_service.storage_available:
        error = (
            semantic_memory_service.last_error
            or "Semantic memory storage is unavailable."
        )
        return MemoryFlushResult(
            attempted=True,
            summary_saved=True,
            error=f"Long-term memory could not start: {error}",
        )

    semantic_error_before = semantic_memory_service.last_error
    fact_ids = semantic_memory_service.add_summary(
        user_id=user_id,
        conversation_id=conversation_id,
        summary=summary,
        source_message_id_start=memory.get("source_message_id_start"),
        source_message_id_end=memory.get("source_message_id_end"),
        summarized_message_count=memory.get("summarized_message_count"),
        metadata={
            "summary_trigger": trigger,
            "memory_summary_window": memory_summary_window,
        },
    )
    semantic_error_after = semantic_memory_service.last_error
    if not fact_ids and semantic_error_after != semantic_error_before:
        return MemoryFlushResult(
            attempted=True,
            summary_saved=True,
            error=f"Long-term memory failed: {semantic_error_after}",
        )
    return MemoryFlushResult(
        attempted=True,
        semantic_fact_ids=tuple(fact_ids),
    )


def _reset_chat_session() -> None:
    """
    Clear active FurnaceMind chat state so the next rerun starts fresh.

    Persisted conversation rows remain in PostgreSQL. This only resets the
    Streamlit session pointers and transient artifacts for the browser session.

    Args:
         - None

    Returns:
         - return: None - This function does not return a value.
    """
    st.session_state["chat_history"] = []
    st.session_state["fm_artifact_store"] = {}
    for key in (
        "fm_conversation_id",
        "_fm_loaded_conversation_id",
        "pending_skill_prompt",
        "fm_fig",
        "fm_fig_turn_id",
        "fm_df",
        "fm_df_turn_id",
        "fm_df_meta",
        "fm_current_artifact_turn_id",
    ):
        st.session_state.pop(key, None)


def _render_reasoning_selector() -> str:
    """Render the user-facing LLM reasoning selector.

    Returns:
        Normalized reasoning level: ``Low``, ``Medium``, or ``High``. Missing
        or invalid values are corrected to the configured Medium profile so the
        downstream LLM client and SQL conversation metadata stay consistent.
    """
    current_level = normalize_openrouter_reasoning_level(
        st.session_state.get("fm_reasoning_level")
        or settings.llm.openrouter.default_reasoning_level
    )
    st.session_state["fm_reasoning_level"] = current_level
    options = list(REASONING_LEVELS)
    selected = st.sidebar.selectbox(
        "Reasoning",
        options,
        index=options.index(current_level),
        key="fm_reasoning_level",
        help="Low uses the faster/lower-cost model, Medium balances quality and latency, High uses the deeper-quality model.",
    )
    return normalize_openrouter_reasoning_level(selected)


def _render_conversation_controls(
    *,
    context: SystemPromptContext,
    semantic_memory_service: SemanticMemoryService | None,
    user_id: str | None,
    conversation_id: str | None,
    memory_summary_window: int,
    memory_summary_token_limit: int,
) -> None:
    """
    Render controls that operate on the active FurnaceMind conversation.

    New Chat is treated as an explicit conversation-finalization event. It
    force-flushes any unsummarized tail before resetting the UI to a new
    conversation id on the next Streamlit rerun.

    Args:
         - context: SystemPromptContext - Current prompt/memory context.
         - semantic_memory_service: SemanticMemoryService | None - Long-term memory service.
         - user_id: str | None - Current FurnaceMind user id.
         - conversation_id: str | None - Active conversation id.
         - memory_summary_window: int - Regular summary window size.
         - memory_summary_token_limit: int - Requested summary token limit.

    Returns:
         - return: None - This function does not return a value.
    """
    notice = st.session_state.pop("fm_new_chat_notice", "")
    if notice:
        st.sidebar.success(notice)

    if not st.sidebar.button(
        "New Chat",
        key="fm_new_chat",
        width="stretch",
    ):
        return

    with st.spinner("Saving conversation memory..."):
        result = _flush_conversation_memory(
            chat_history=st.session_state.get("chat_history") or [],
            memory=context.memory,
            user_id=user_id,
            conversation_id=conversation_id,
            semantic_memory_service=semantic_memory_service,
            memory_summary_window=memory_summary_window,
            memory_summary_token_limit=memory_summary_token_limit,
            trigger="new_chat",
            force=True,
            retry_saved_summary=bool(
                st.session_state.get("fm_new_chat_retry_semantic")
            ),
        )

    if result.error:
        if result.summary_saved:
            st.session_state["fm_new_chat_retry_semantic"] = True
        st.sidebar.error(result.error)
        return

    st.session_state.pop("fm_new_chat_retry_semantic", None)
    _reset_chat_session()
    if result.summary_saved or result.semantic_fact_ids:
        st.session_state["fm_new_chat_notice"] = "Conversation memory saved."
    else:
        st.session_state["fm_new_chat_notice"] = "New chat started."
    st.rerun()


def render_ai_cooperate(*, field_labels: dict) -> None:  # noqa: ARG001
    """
    Render the FurnaceMind AI Co-Operate chat page and handle one chat turn.

    The function initializes cached clients, restores the active conversation
    from PostgreSQL, refreshes prompt memory, renders chat UI controls, persists
    the new user and assistant messages, runs the agent, and updates the rolling
    memory summary when the configured message window is reached.

    Args:
         - field_labels: dict - Page field labels passed by the app shell.

    Returns:
         - return: None - This function does not return a value.
    """
    embedding_client = _cached_embedding_client()
    knowledge_store = _cached_knowledge_store()
    context = _cached_context()
    engine = _cached_skill_engine()
    skill_repository = _cached_skill_repository()
    skill_vector_store = _cached_skill_vector_store()
    skill_registry = SkillRegistry(
        engine=engine,
        repository=skill_repository,
        vector_store=skill_vector_store,
    )
    history_store = _cached_history_store()
    document_repository = _cached_document_repository()
    chunk_repository = _cached_chunk_repository()
    retrieval_trace_repository = _cached_retrieval_trace_repository()
    feedback_service = _cached_feedback_service()
    feedback_llm = _cached_feedback_llm()
    semantic_memory_service = _cached_semantic_memory_service()
    user_id = current_user_id()
    memory_summary_window = settings.memory_summary_message_window
    memory_summary_token_limit = settings.memory_summary_token_limit

    st.session_state["knowledge_store"] = knowledge_store
    st.session_state["fm_user_id"] = user_id
    if document_repository is not None:
        st.session_state["knowledge_document_repository"] = document_repository
    else:
        st.session_state.pop("knowledge_document_repository", None)
    if chunk_repository is not None:
        st.session_state["knowledge_chunk_repository"] = chunk_repository
    else:
        st.session_state.pop("knowledge_chunk_repository", None)
    if retrieval_trace_repository is not None:
        st.session_state["knowledge_retrieval_trace_repository"] = (
            retrieval_trace_repository
        )
    else:
        st.session_state.pop("knowledge_retrieval_trace_repository", None)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.setdefault("fm_artifact_store", {})
    selected_reasoning_level = _render_reasoning_selector()

    if not user_id:
        history_store = None
        document_repository = None
        chunk_repository = None
        retrieval_trace_repository = None
        feedback_service = None
        st.session_state.pop("knowledge_document_repository", None)
        st.session_state.pop("knowledge_chunk_repository", None)
        st.session_state.pop("knowledge_retrieval_trace_repository", None)
        st.sidebar.caption("FurnaceMind user id unavailable for SQL persistence.")

    conversation_id: str | None = st.session_state.get("fm_conversation_id")
    if history_store is not None and user_id:
        try:
            conversation_id = history_store.ensure_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                model_mode=selected_reasoning_level,
            )
            st.session_state["fm_conversation_id"] = conversation_id
            if st.session_state.get("_fm_loaded_conversation_id") != conversation_id:
                st.session_state.chat_history = history_store.load_chat_history(
                    conversation_id=conversation_id
                )
                st.session_state["_fm_loaded_conversation_id"] = conversation_id
        except Exception as exc:
            history_store = None
            st.sidebar.caption(f"Chat history database unavailable: {exc}")
    else:
        st.sidebar.caption("Chat history database unavailable.")

    if user_id and document_repository is None:
        st.sidebar.caption("Knowledge document metadata database unavailable.")
    if skill_repository is None:
        st.sidebar.caption(
            "Skill registry database unavailable; using built-in quick skills."
        )
    if skill_vector_store is None:
        st.sidebar.caption(
            "Skill vector index unavailable; selected skills still work."
        )

    feedback_flow.process_pending_explicit_feedback(
        feedback_service=feedback_service,
        feedback_llm=feedback_llm,
        user_id=user_id or "",
    )

    context.refresh_session_context(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    _render_conversation_controls(
        context=context,
        semantic_memory_service=semantic_memory_service,
        user_id=user_id,
        conversation_id=conversation_id,
        memory_summary_window=memory_summary_window,
        memory_summary_token_limit=memory_summary_token_limit,
    )
    chat_interface.render_skill_sidebar(
        skill_repository=skill_repository,
        skill_vector_store=skill_vector_store,
        user_id=user_id,
    )
    chat_interface.render_knowledge_sidebar(
        knowledge_store=knowledge_store,
        embedding_client=embedding_client,
        user_id=user_id,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        semantic_memory_service=semantic_memory_service,
    )

    default_date, default_label = last_completed_shift()

    chat_interface.render_chat_history()

    chat_interface.render_quick_skills(skill_registry, default_date, default_label)
    chat_submission = st.chat_input(
        "Ask FurnaceMind or attach a document...",
        accept_file="multiple",
        file_type=chat_interface.KNOWLEDGE_FILE_TYPES,
        key="furnacemind_chat_input",
    )

    user_query: str | None = None
    user_display: str | None = None
    active_skill_id: str | None = None
    active_skill_context: str | None = None
    if "pending_skill_prompt" in st.session_state:
        pending = st.session_state.pop("pending_skill_prompt")
        user_query = pending.get("prompt")
        user_display = pending.get("display")
        active_skill_id = pending.get("skill_id")
        active_skill_context = pending.get("skill_context")
    else:
        user_query, user_display = chat_interface.extract_submission(
            chat_submission,
            knowledge_store=knowledge_store,
            embedding_client=embedding_client,
            user_id=user_id,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
        )

    if not user_query:
        return

    recent_skill_messages = chat_interface.chat_history_to_messages(max_messages=3)
    turn_skill_context = skill_registry.turn_skill_context(
        query=user_query,
        recent_messages=recent_skill_messages,
        selected_skill_id=active_skill_id,
        selected_skill_context=active_skill_context,
        embedding_client=embedding_client,
    )
    if turn_skill_context:
        active_skill_context = turn_skill_context

    skill_context_skill_ids = list(skill_registry.last_context_skill_ids)
    skill_context_skill_slugs = list(skill_registry.last_context_skill_slugs)
    skill_message_metadata: dict[str, Any] = {}
    if skill_context_skill_ids or skill_context_skill_slugs:
        skill_message_metadata = {
            "exclude_from_memory": True,
            "skill_context_skill_ids": skill_context_skill_ids,
            "skill_context_skill_slugs": skill_context_skill_slugs,
            "skill_context_source": "furnacemind_skills",
        }
    feedback_flow.detect_and_save_chat_feedback(
        feedback_service=feedback_service,
        feedback_llm=feedback_llm,
        user_id=user_id or "",
        conversation_id=conversation_id,
        user_query=user_query,
    )
    feedback_lesson_context = (
        feedback_service.feedback_context(query=user_query, user_id=user_id)
        if feedback_service is not None and user_id
        else ""
    )
    semantic_memory_context = (
        semantic_memory_service.context_for_query(query=user_query, user_id=user_id)
        if semantic_memory_service is not None and user_id
        else ""
    )
    turn_message_metadata = {
        "reasoning_level": selected_reasoning_level,
        **skill_message_metadata,
    }
    user_message_id: str | None = None
    if history_store is not None and conversation_id and user_id:
        try:
            user_message_id = history_store.add_user_message(
                conversation_id=conversation_id,
                user_id=user_id,
                content=user_query,
                display=user_display or user_query,
                metadata=turn_message_metadata or None,
            )
        except Exception as exc:
            st.sidebar.caption(f"Could not save user message: {exc}")

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_query,
            "display": user_display or user_query,
            "type": "text",
            "message_id": user_message_id,
            "conversation_id": conversation_id,
            "metadata": turn_message_metadata,
        }
    )
    with st.chat_message("user"):
        st.markdown(user_display or user_query)

    llm = OpenRouterClient(reasoning_level=selected_reasoning_level)
    if st.session_state.get("shift_store") is None:
        st.session_state["shift_store"] = _cached_shift_store()
    tools = furnace_tools.get_openai_tool_schemas()
    extra_context = prompts.TOOL_POLICY
    if feedback_lesson_context:
        extra_context = f"{extra_context}\n\n{feedback_lesson_context}"
    if semantic_memory_context:
        extra_context = f"{extra_context}\n\n{semantic_memory_context}"
    messages = [
        {
            "role": "system",
            "content": context.build(
                extra=extra_context,
                skill_id=active_skill_id,
                skill_context=active_skill_context,
            ),
        },
        *chat_interface.chat_history_to_messages(
            max_messages=memory_summary_window,
        ),
    ]
    history_len_before = len(st.session_state.chat_history)
    artifact_turn_id = uuid.uuid4().hex
    st.session_state["fm_current_artifact_turn_id"] = artifact_turn_id

    try:
        with st.chat_message("assistant"):
            status_box = st.empty()
            response_box = st.empty()
            status_box.status("Thinking...", expanded=False)
            try:
                final_response = run_agent_loop(
                    llm=llm,
                    messages=messages,
                    tools=tools,
                    status_box=status_box,
                    response_box=response_box,
                )
            except Exception as exc:
                llm.record_failure(exc)
                final_response = llm.unavailable_message()
                status_box.empty()
                response_box.markdown(final_response)
    finally:
        st.session_state.pop("fm_current_artifact_turn_id", None)

    chat_interface.inject_artifacts(
        history_len_before,
        artifact_turn_id=artifact_turn_id,
    )

    assistant_metadata = {
        "source": "furnacemind",
        **turn_message_metadata,
        **llm.usage_metadata(),
    }
    assistant_message_id: str | None = None
    if history_store is not None and conversation_id and user_id:
        try:
            assistant_message_id = history_store.add_assistant_message(
                conversation_id=conversation_id,
                user_id=user_id,
                content=final_response,
                model=assistant_metadata.get("actual_model")
                or getattr(llm, "model", None),
                metadata=assistant_metadata,
            )
        except Exception as exc:
            st.sidebar.caption(f"Could not save assistant message: {exc}")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": final_response,
            "display": final_response,
            "type": "text",
            "message_id": assistant_message_id,
            "conversation_id": conversation_id,
            "metadata": assistant_metadata,
        }
    )

    _flush_conversation_memory(
        chat_history=st.session_state.chat_history,
        memory=context.memory,
        user_id=user_id,
        conversation_id=conversation_id,
        semantic_memory_service=semantic_memory_service,
        memory_summary_window=memory_summary_window,
        memory_summary_token_limit=memory_summary_token_limit,
        trigger="window",
    )
    st.rerun()
