"""Streamlit page renderer for the FurnaceMind AI Co-Operate experience.

This module wires together the FurnaceMind chat UI, PostgreSQL conversation
persistence, rolling memory summaries, skill shortcuts, vector stores, and the
agent loop. It intentionally keeps page orchestration here while reusable UI
pieces live under ``ui.furnacemind`` and persistence logic lives under
``agents.memory``.
"""

from __future__ import annotations

import streamlit as st

from agents.embeddings.cloud_embedding import CloudEmbeddingClient
from agents.furnace_tools import get_openai_tool_schemas
from agents.furnacemind import prompts
from agents.furnacemind.agent import run_agent_loop
from agents.furnacemind.context import SystemPromptContext
from agents.furnacemind.skills import SkillEngine
from agents.llm.llm_client import OpenRouterClient
from agents.memory import fm_memory
from agents.memory.conversation_history import ConversationHistoryStore
from agents.memory.knowledge_vector_store import KnowledgeVectorStore
from agents.memory.vector_store import QdrantVectorStore
from ui.furnacemind import chat_interface
from utils.settings import settings
from utils.shift_windows import last_completed_shift


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


def _current_user_id() -> str:
    """
    Resolve the user id used for FurnaceMind persistence records.

    The relational tables link conversations, messages, and summaries by user.
    When the app does not provide an authenticated user in Streamlit session
    state, this function returns a stable ``anonymous`` fallback so local testing
    and unauthenticated sessions still work.

    Args:
         - None

    Returns:
         - return: str - Current user id or anonymous fallback.
    """
    user_id = str(st.session_state.get("auth_user") or "anonymous").strip()
    return user_id or "anonymous"


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
    history_store = _cached_history_store()
    user_id = _current_user_id()
    memory_summary_window = settings.memory_summary_message_window
    memory_summary_token_limit = settings.memory_summary_token_limit

    st.session_state["knowledge_store"] = knowledge_store

    chat_interface.render_knowledge_sidebar(
        knowledge_store=knowledge_store,
        embedding_client=embedding_client,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.setdefault("fm_artifact_store", {})
    conversation_id: str | None = st.session_state.get("fm_conversation_id")
    if history_store is not None:
        try:
            conversation_id = history_store.ensure_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
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

    context.refresh_session_context(
        user_id=user_id,
        conversation_id=conversation_id,
    )

    default_date, default_label = last_completed_shift()

    chat_interface.render_chat_history()

    chat_interface.render_quick_skills(engine, default_date, default_label)
    chat_submission = st.chat_input(
        "Ask FurnaceMind or attach a document...",
        accept_file="multiple",
        file_type="document",
        key="furnacemind_chat_input",
    )

    user_query: str | None = None
    user_display: str | None = None
    active_skill_id: str | None = None

    if "pending_skill_prompt" in st.session_state:
        pending = st.session_state.pop("pending_skill_prompt")
        user_query = pending.get("prompt")
        user_display = pending.get("display")
        active_skill_id = pending.get("skill_id")
    else:
        user_query, user_display = chat_interface.extract_submission(
            chat_submission,
            knowledge_store=knowledge_store,
            embedding_client=embedding_client,
        )

    if not user_query:
        return

    user_message_id: str | None = None
    if history_store is not None and conversation_id:
        try:
            user_message_id = history_store.add_user_message(
                conversation_id=conversation_id,
                user_id=user_id,
                content=user_query,
                display=user_display or user_query,
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
        }
    )
    with st.chat_message("user"):
        st.markdown(user_display or user_query)

    llm = OpenRouterClient()
    if st.session_state.get("shift_store") is None:
        st.session_state["shift_store"] = _cached_shift_store()
    tools = get_openai_tool_schemas()
    messages = [
        {
            "role": "system",
            "content": context.build(
                extra=prompts.TOOL_POLICY,
                skill_id=active_skill_id,
            ),
        },
        *chat_interface.chat_history_to_messages(
            max_messages=memory_summary_window,
        ),
    ]
    history_len_before = len(st.session_state.chat_history)

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

    chat_interface.inject_artifacts(history_len_before)

    assistant_message_id: str | None = None
    if history_store is not None and conversation_id:
        try:
            assistant_message_id = history_store.add_assistant_message(
                conversation_id=conversation_id,
                user_id=user_id,
                content=final_response,
                model=getattr(llm, "model", None),
                metadata={"source": "furnacemind"},
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
        }
    )

    if fm_memory.should_generate_memory_summary(
        st.session_state.chat_history,
        window=memory_summary_window,
    ):
        memory_llm = OpenRouterClient(
            model_name=settings.llm.openrouter.memory_compression_model_name
        )
        updated_memory = fm_memory.generate_memory_summary(
            context.memory,
            chat_history=st.session_state.chat_history,
            llm=memory_llm,
            summary_system_prompt=prompts.memory_summary_system_prompt(
                memory_summary_token_limit
            ),
            summary_token_limit=memory_summary_token_limit,
            window=memory_summary_window,
        )
        source_message_id_start, source_message_id_end = (
            fm_memory.summary_source_message_ids(
                st.session_state.chat_history,
                window=memory_summary_window,
            )
        )
        fm_memory.save_fm_memory(
            updated_memory,
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id_start=source_message_id_start or user_message_id,
            source_message_id_end=source_message_id_end or assistant_message_id,
        )
    st.rerun()
