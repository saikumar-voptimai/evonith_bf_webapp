"""Render AI Co-Operate tab for FurnaceMind.

Artifacts (plots/dataframes) are rendered inline in the chat stream.
No separate right-side artifact pane is used.
"""

from __future__ import annotations

import streamlit as st

from agents.embeddings.cloud_embedding import CloudEmbeddingClient
from agents.furnace_tools import get_openai_tool_schemas
from agents.furnacemind.agent import run_agent_loop
from agents.furnacemind.context import SystemPromptContext
from agents.furnacemind.prompts import TOOL_POLICY
from agents.furnacemind.skills import SkillEngine
from agents.llm.llm_client import OpenRouterClient
from agents.memory.conversation_history import ConversationHistoryStore
from agents.memory.fm_memory import add_recent_turn, save_fm_memory
from agents.memory.knowledge_vector_store import KnowledgeVectorStore
from agents.memory.vector_store import QdrantVectorStore
from ui.furnacemind import chat_interface
from utils.shift_windows import last_completed_shift


@st.cache_resource(show_spinner=False)
def _cached_embedding_client() -> CloudEmbeddingClient:
    """
    Return the cached embedding client.

    Args:
         - None

    Returns:
         - return: CloudEmbeddingClient - Singleton embedding client instance.
    """
    return CloudEmbeddingClient()


@st.cache_resource(show_spinner=False)
def _cached_knowledge_store() -> KnowledgeVectorStore:
    """
    Return the cached knowledge vector store.

    Args:
         - None

    Returns:
         - return: KnowledgeVectorStore - Singleton knowledge vector store.
    """
    return KnowledgeVectorStore(_cached_embedding_client())


@st.cache_resource(show_spinner=False)
def _cached_shift_store() -> QdrantVectorStore:
    """
    Return the cached shift vector store.

    Args:
         - None

    Returns:
         - return: QdrantVectorStore - Singleton shift vector store.
    """
    return QdrantVectorStore()


@st.cache_resource(show_spinner=False)
def _cached_context() -> SystemPromptContext:
    """
    Return the cached system prompt context.

    Args:
         - None

    Returns:
         - return: SystemPromptContext - Singleton prompt context.
    """
    return SystemPromptContext()


@st.cache_resource(show_spinner=False)
def _cached_skill_engine() -> SkillEngine:
    """
    Return the cached skill engine.

    Args:
         - None

    Returns:
         - return: SkillEngine - Singleton FurnaceMind skill engine.
    """
    return SkillEngine()


@st.cache_resource(show_spinner=False)
def _cached_history_store() -> ConversationHistoryStore | None:
    """
    Return a singleton PostgreSQL conversation history store when configured.

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
    Return the current authenticated user id for persistence.

    Args:
         - None

    Returns:
         - return: str - Current user id or anonymous fallback.
    """
    user_id = str(st.session_state.get("auth_user") or "anonymous").strip()
    return user_id or "anonymous"


def render_ai_cooperate(*, field_labels: dict) -> None:  # noqa: ARG001
    """
    Render FurnaceMind AI Co-Operate tab.

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
            "content": context.build(extra=TOOL_POLICY, skill_id=active_skill_id),
        },
        *chat_interface.chat_history_to_messages(),
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

    updated_memory = add_recent_turn(
        context.memory, user=user_query, assistant=final_response
    )
    save_fm_memory(
        updated_memory,
        user_id=user_id,
        conversation_id=conversation_id,
        source_message_id_start=user_message_id,
        source_message_id_end=assistant_message_id,
    )
    st.rerun()
