"""Streamlit chat UI helpers for FurnaceMind."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import streamlit as st

from agents.multimodal.ingestion import process_file

if TYPE_CHECKING:
    from agents.furnacemind.skills import SkillEngine

_ARTIFACT_TYPES = {"plotly", "dataframe"}
_CHAT_HISTORY_LIMIT = 14
_CHAT_HISTORY_HEIGHT = 560
_KNOWLEDGE_FILE_TYPE = "document"


def chat_history_to_messages(
    max_messages: int = _CHAT_HISTORY_LIMIT,
) -> list[dict]:
    """
    Convert chat history to OpenAI messages, skipping artifact entries.

    Args:
         - max_messages: int - Maximum recent messages to include.

    Returns:
         - return: list[dict] - Chat messages compatible with the LLM client.
    """
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


def render_chat_history(*, height: int = _CHAT_HISTORY_HEIGHT) -> None:
    """
    Render persisted and in-session chat messages.

    Args:
         - height: int - Streamlit chat history container height.

    Returns:
         - return: None - This function does not return a value.
    """
    with st.container(height=height, border=False):
        for item in st.session_state.chat_history:
            role = item.get("role", "assistant")
            with st.chat_message(role):
                render_message(item)


def render_message(item: dict) -> None:
    """
    Render one chat message including inline artifacts.

    Args:
         - item: dict - Chat history item to render.

    Returns:
         - return: None - This function does not return a value.
    """
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


def inject_artifacts(history_len_before: int) -> None:
    """
    Append inline artifact messages for plot/dataframe produced this turn.

    Args:
         - history_len_before: int - Chat history length before agent execution.

    Returns:
         - return: None - This function does not return a value.
    """
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


def render_knowledge_sidebar(*, knowledge_store: Any, embedding_client: Any) -> None:
    """
    Render optional knowledge upload controls.

    Args:
         - knowledge_store: Any - Vector store used to index uploaded files.
         - embedding_client: Any - Embedding client used for uploaded files.

    Returns:
         - return: None - This function does not return a value.
    """
    with st.sidebar.expander(
        "Knowledge (optional)", expanded=False, key="fm_knowledge"
    ):
        uploaded_files = st.file_uploader(
            "Upload Knowledge Files",
            type=_KNOWLEDGE_FILE_TYPE,
            accept_multiple_files=True,
            key="knowledge_uploader",
        )
        upload_status = st.empty()
        if uploaded_files:
            for uploaded in uploaded_files:
                process_file(uploaded, knowledge_store, embedding_client)
            upload_status.success("Documents indexed successfully.")


def _queue_skill(prompt: str, display: str, skill_id: str) -> None:
    """
    Queue quick-skill execution and rerun the Streamlit page.

    Args:
         - prompt: str - Prompt generated by the selected quick skill.
         - display: str - User-facing text shown in chat history.
         - skill_id: str - Skill identifier used for prompt context.

    Returns:
         - return: None - This function does not return a value.
    """
    st.session_state.pop("fm_fig", None)
    st.session_state.pop("fm_df", None)
    st.session_state.pop("fm_df_meta", None)
    st.session_state["pending_skill_prompt"] = {
        "prompt": prompt,
        "display": display,
        "skill_id": skill_id,
    }
    st.rerun()


def render_quick_skills(
    engine: SkillEngine, shift_date: date, shift_label: str
) -> None:
    """
    Render quick-skill buttons directly above chat input.

    Args:
         - engine: SkillEngine - FurnaceMind quick-skill prompt engine.
         - shift_date: date - Shift date used for shift-based prompts.
         - shift_label: str - Shift label used for shift-based prompts.

    Returns:
         - return: None - This function does not return a value.
    """
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


def extract_submission(
    chat_submission: object,
    *,
    knowledge_store: Any,
    embedding_client: Any,
) -> tuple[str | None, str | None]:
    """
    Extract user query and display text from the Streamlit chat input.

    Args:
         - chat_submission: object - Streamlit chat input payload.
         - knowledge_store: Any - Vector store used to index attached files.
         - embedding_client: Any - Embedding client used for attached files.

    Returns:
         - return: tuple[str | None, str | None] - Query and display text.
    """
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
