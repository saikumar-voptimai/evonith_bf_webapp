"""Render the FurnaceMind AI Co-Operate page."""

from __future__ import annotations

import streamlit as st

from agents.furnace_tools import get_openai_tool_schemas
from agents.furnacemind.agent import run_agent_loop
from agents.furnacemind.chat_ui import (
    inject_artifacts,
    render_message,
    render_quick_skills,
)
from agents.furnacemind.prompts import TOOL_POLICY
from agents.furnacemind.resources import (
    cached_context,
    cached_embedding_client,
    cached_history_store,
    cached_ingestion_service,
    cached_knowledge_store,
    cached_shift_store,
    cached_skill_engine,
)
from agents.furnacemind.state import (
    chat_history_to_messages,
    current_user_id,
    last_completed_shift,
)
from agents.furnacemind.submissions import (
    extract_submission,
    ingest_uploaded_knowledge_files,
)
from agents.llm.llm_client import OpenRouterClient
from agents.memory.fm_memory import add_recent_turn


def render_ai_cooperate(*, field_labels: dict) -> None:  # noqa: ARG001
    """
    Render the FurnaceMind AI Co-Operate tab.

    Args:
         - field_labels: dict - Page field labels supplied by the app router.

    Returns:
         - return: None - Renders the Streamlit page.
    """
    embedding_client = cached_embedding_client()
    knowledge_store = cached_knowledge_store()
    context = cached_context()
    engine = cached_skill_engine()
    history_store = cached_history_store()
    ingestion_service = cached_ingestion_service()
    user_id = current_user_id()

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
            ingest_uploaded_knowledge_files(
                uploaded_files,
                user_id=user_id,
                knowledge_store=knowledge_store,
                embedding_client=embedding_client,
                ingestion_service=ingestion_service,
            )
            upload_status.success("Documents indexed successfully.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.setdefault("fm_artifact_store", {})
    conversation_id: str | None = None
    prompt_memory: dict | None = None
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
            prompt_memory = history_store.load_memory(conversation_id=conversation_id)
        except Exception as exc:
            history_store = None
            st.sidebar.caption(f"Chat history database unavailable: {exc}")
    else:
        st.sidebar.caption("Chat history database unavailable.")
    context.refresh_session_context(memory=prompt_memory)

    default_date, default_label = last_completed_shift()

    with st.container(height=560, border=False):
        for item in st.session_state.chat_history:
            role = item.get("role", "assistant")
            with st.chat_message(role):
                render_message(item)

    render_quick_skills(engine, default_date, default_label)
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
        user_query, user_display = extract_submission(
            chat_submission,
            user_id=user_id,
            knowledge_store=knowledge_store,
            embedding_client=embedding_client,
            ingestion_service=ingestion_service,
        )

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
        st.session_state["shift_store"] = cached_shift_store()
    tools = get_openai_tool_schemas()
    messages = [
        {
            "role": "system",
            "content": context.build(extra=TOOL_POLICY, skill_id=active_skill_id),
        },
        *chat_history_to_messages(),
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

    inject_artifacts(history_len_before)

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

    if history_store is not None and conversation_id:
        try:
            context.refresh_memory(
                memory=history_store.load_memory(conversation_id=conversation_id)
            )
        except Exception:
            pass
    else:
        context.refresh_memory(
            memory=add_recent_turn(
                context.memory, user=user_query, assistant=final_response
            )
        )
    st.rerun()
