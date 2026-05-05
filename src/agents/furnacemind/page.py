"""Render AI Co-Operate tab for FurnaceMind.

Artifacts (plots/dataframes) are rendered inline in the chat stream.
No separate right-side artifact pane is used.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

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
from agents.multimodal.ingestion import process_file

_IST = timezone(timedelta(hours=5, minutes=30))
_ARTIFACT_TYPES = {"plotly", "dataframe"}


@st.cache_resource(show_spinner=False)
def _cached_embedding_client() -> CloudEmbeddingClient:
    """Return singleton embedding client."""
    return CloudEmbeddingClient()


@st.cache_resource(show_spinner=False)
def _cached_knowledge_store() -> KnowledgeVectorStore:
    """Return singleton knowledge vector store."""
    return KnowledgeVectorStore(_cached_embedding_client())


@st.cache_resource(show_spinner=False)
def _cached_shift_store() -> QdrantVectorStore:
    """Return singleton shift vector store."""
    return QdrantVectorStore()


@st.cache_resource(show_spinner=False)
def _cached_context() -> SystemPromptContext:
    """Return singleton system prompt context."""
    return SystemPromptContext()


@st.cache_resource(show_spinner=False)
def _cached_skill_engine() -> SkillEngine:
    """Return singleton skill engine."""
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


def _last_completed_shift() -> tuple[date, str]:
    """Return date/label for most recently completed IST shift."""
    now = datetime.now(_IST)
    if now.hour < 6:
        return (now.date() - timedelta(days=1)), "C"
    if now.hour < 14:
        return now.date(), "A"
    if now.hour < 22:
        return now.date(), "B"
    return now.date(), "C"


def _chat_history_to_messages(max_messages: int = 14) -> list[dict]:
    """Convert chat history to OpenAI messages, skipping artifact entries."""
    messages: list[dict] = []
    for item in (st.session_state.get("chat_history") or [])[-max_messages:]:
        if item.get("type") in _ARTIFACT_TYPES:
            continue
        role = item.get("role")
        content = item.get("content")
        if (
            role in {"user", "assistant"}
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({"role": role, "content": content})
    return messages


def _render_message(item: dict) -> None:
    """Render one chat message including inline artifacts."""
    message_type = item.get("type") or "text"
    artifact_store: dict = st.session_state.get("fm_artifact_store") or {}

    if message_type == "text":
        st.markdown(item.get("display") or item.get("content", ""))
        return

    if message_type == "plotly":
        artifact_key = item.get("artifact_key", "")
        figure = artifact_store.get(artifact_key)
        if figure is not None:
            st.plotly_chart(
                figure, use_container_width=True, key=f"chat_{artifact_key}"
            )
        else:
            st.caption("_Chart no longer in session._")
        return

    if message_type == "dataframe":
        artifact_key = item.get("artifact_key", "")
        dataframe = artifact_store.get(artifact_key)
        if dataframe is not None and not dataframe.empty:
            meta = item.get("meta", {})
            dataset_id = meta.get("dataset_id", "")
            st.caption(
                f"Dataset: {dataframe.shape[0]:,} rows x {dataframe.shape[1]} cols"
                + (f" | {dataset_id}" if dataset_id else "")
            )
            st.dataframe(
                dataframe.head(50),
                use_container_width=True,
                hide_index=True,
                key=f"chat_{artifact_key}_df",
            )
            st.download_button(
                "Download CSV",
                dataframe.to_csv(index=True).encode(),
                file_name=f"{dataset_id or artifact_key}.csv",
                mime="text/csv",
                key=f"chat_{artifact_key}_dl",
            )
        else:
            st.caption("_Data no longer in session._")


def _inject_artifacts(history_len_before: int) -> None:
    """Append inline artifact messages for plot/dataframe produced this turn."""
    artifact_store: dict = st.session_state.setdefault("fm_artifact_store", {})
    history: list = st.session_state.chat_history

    new_figure = st.session_state.get("fm_fig")
    if new_figure is not None:
        key = f"fm_fig_{history_len_before}"
        if not any(item.get("artifact_key") == key for item in history):
            artifact_store[key] = new_figure
            history.append(
                {
                    "role": "assistant",
                    "type": "plotly",
                    "artifact_key": key,
                    "content": "",
                }
            )

    new_dataframe = st.session_state.get("fm_df")
    if new_dataframe is not None and not new_dataframe.empty:
        key = f"fm_df_{history_len_before}"
        if not any(item.get("artifact_key") == key for item in history):
            artifact_store[key] = new_dataframe
            history.append(
                {
                    "role": "assistant",
                    "type": "dataframe",
                    "artifact_key": key,
                    "content": "",
                    "meta": st.session_state.get("fm_df_meta", {}),
                }
            )


def _queue_skill(prompt: str, display: str, skill_id: str) -> None:
    """Queue quick-skill execution and rerun."""
    st.session_state.pop("fm_fig", None)
    st.session_state.pop("fm_df", None)
    st.session_state.pop("fm_df_meta", None)
    st.session_state["pending_skill_prompt"] = {
        "prompt": prompt,
        "display": display,
        "skill_id": skill_id,
    }
    st.rerun()


def _render_quick_skills(
    engine: SkillEngine, shift_date: date, shift_label: str
) -> None:
    """Render quick-skill buttons directly above chat input."""
    st.caption("Quick skills")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Unit Cost", use_container_width=True, key="fm_skill_cost"):
            _queue_skill(
                engine.optimise_prompt(),
                "Unit Cost - last 30 days vs best-shift targets",
                "optimise",
            )
    with c2:
        if st.button("Shift to Best", use_container_width=True, key="fm_skill_shift"):
            _queue_skill(
                engine.shift_to_best_prompt(str(shift_date), shift_label),
                f"Shift to Best: {shift_date}, Shift {shift_label}",
                "shift_to_best",
            )
    with c3:
        if st.button("Heatloads", use_container_width=True, key="fm_skill_heatloads"):
            _queue_skill(
                engine.heatload_prompt(),
                "Heatloads - last 8h vs 2-month baseline",
                "heatload",
            )


def _extract_submission(chat_submission: object) -> tuple[str | None, str | None]:
    """Extract user query/display from chat_input object."""
    if not chat_submission:
        return None, None

    if hasattr(chat_submission, "text"):
        typed_query = chat_submission.text
        files = getattr(chat_submission, "files", None) or []
    elif isinstance(chat_submission, dict):
        typed_query = chat_submission.get("text", "")
        files = chat_submission.get("files", []) or []
    else:
        typed_query = str(chat_submission)
        files = []

    knowledge_store = st.session_state.get("knowledge_store")
    embedding_client = _cached_embedding_client()
    if files and knowledge_store and embedding_client:
        for uploaded in files:
            process_file(uploaded, knowledge_store, embedding_client)

    if typed_query and str(typed_query).strip():
        text = str(typed_query).strip()
        return text, text
    if files:
        file_label = ", ".join(getattr(file_obj, "name", "file") for file_obj in files)
        display = f"Attached files: {file_label}"
        return display, display
    return None, None


def render_ai_cooperate(*, field_labels: dict) -> None:  # noqa: ARG001
    """Render FurnaceMind AI Co-Operate tab."""
    embedding_client = _cached_embedding_client()
    knowledge_store = _cached_knowledge_store()
    context = _cached_context()
    engine = _cached_skill_engine()
    history_store = _cached_history_store()
    user_id = _current_user_id()

    context.refresh_session_context()
    st.session_state["knowledge_store"] = knowledge_store

    with st.sidebar.expander(
        "Knowledge (optional)", expanded=False, key="fm_knowledge"
    ):
        uploaded_files = st.file_uploader(
            "Upload Knowledge Files",
            type="document",
            accept_multiple_files=True,
            key="knowledge_uploader",
        )
        upload_status = st.empty()
        if uploaded_files:
            for uploaded in uploaded_files:
                process_file(uploaded, knowledge_store, embedding_client)
            upload_status.success("Documents indexed successfully.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.setdefault("fm_artifact_store", {})
    if history_store is not None:
        try:
            conversation_id = history_store.ensure_conversation(
                user_id=user_id,
                conversation_id=st.session_state.get("fm_conversation_id"),
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

    default_date, default_label = _last_completed_shift()

    with st.container(height=560, border=False):
        for item in st.session_state.chat_history:
            role = item.get("role", "assistant")
            with st.chat_message(role):
                _render_message(item)

    _render_quick_skills(engine, default_date, default_label)
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
        user_query, user_display = _extract_submission(chat_submission)

    if not user_query:
        return

    conversation_id = st.session_state.get("fm_conversation_id")
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
        *_chat_history_to_messages(),
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

    _inject_artifacts(history_len_before)

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
    save_fm_memory(updated_memory)
    st.rerun()
