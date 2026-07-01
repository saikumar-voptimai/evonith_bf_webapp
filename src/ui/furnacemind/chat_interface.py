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
KNOWLEDGE_FILE_TYPES = [
    "pdf",
    "txt",
    "md",
    "markdown",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "pptx",
    "xlsx",
    "xls",
    "csv",
    "docx",
]


def _message_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Return normalized metadata for a chat-history item."""
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _message_knowledge_document_ids(item: dict[str, Any]) -> set[str]:
    """Return MRAG document ids attached to a chat-history item."""
    metadata = _message_metadata(item)
    raw_ids = metadata.get("knowledge_document_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, (list, tuple, set)):
        return set()
    return {
        str(document_id).strip() for document_id in raw_ids if str(document_id).strip()
    }


def _is_revoked_knowledge_message(item: dict[str, Any]) -> bool:
    """Return True when a message depends on a removed knowledge document."""
    revoked_ids = st.session_state.get("fm_revoked_knowledge_document_ids") or set()
    if not isinstance(revoked_ids, (set, list, tuple)):
        return False
    return bool(_message_knowledge_document_ids(item) & set(revoked_ids))


def _remember_revoked_knowledge_document(document_id: str) -> None:
    """Remember a removed MRAG document id for this Streamlit session."""
    normalized = str(document_id or "").strip()
    if not normalized:
        return
    revoked_ids = st.session_state.setdefault(
        "fm_revoked_knowledge_document_ids", set()
    )
    if not isinstance(revoked_ids, set):
        revoked_ids = set(revoked_ids or [])
        st.session_state["fm_revoked_knowledge_document_ids"] = revoked_ids
    revoked_ids.add(normalized)


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
        if _is_revoked_knowledge_message(item):
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
        history = st.session_state.chat_history
        for index, item in enumerate(history):
            role = item.get("role", "assistant")
            with st.chat_message(role):
                render_message(item)
                if _can_collect_feedback(item):
                    render_feedback_controls(
                        item,
                        raw_user_message=_previous_user_message(history, index),
                    )


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
            st.plotly_chart(figure, width="stretch", key=f"chat_{artifact_key}")
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
                width="stretch",
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


def _can_collect_feedback(item: dict) -> bool:
    """
    Check whether an assistant chat item can receive explicit feedback.

    Feedback must be linked to a persisted assistant message. Artifact entries
    are skipped because the feedback table expects the assistant answer text and
    message id, not transient chart or dataframe render entries.

    Args:
         - item: dict - Chat history item being rendered.

    Returns:
         - return: bool - True when thumbs feedback can be shown.
    """
    return (
        item.get("role") == "assistant"
        and item.get("type", "text") == "text"
        and bool(item.get("message_id"))
        and bool(item.get("conversation_id"))
        and bool(str(item.get("content") or "").strip())
    )


def _previous_user_message(history: list[dict], assistant_index: int) -> str:
    """
    Return the user message that produced an assistant response.

    Args:
         - history: list[dict] - Current Streamlit chat history.
         - assistant_index: int - Index of the assistant message being rendered.

    Returns:
         - return: str - Previous user message text, or an empty string.
    """
    for index in range(assistant_index - 1, -1, -1):
        item = history[index]
        if item.get("type", "text") != "text":
            continue
        if item.get("role") == "user":
            return str(item.get("content") or "").strip()
    return ""


def render_feedback_controls(item: dict, *, raw_user_message: str) -> None:
    """
    Render thumbs feedback controls below one assistant response.

    The first click only selects positive or negative feedback. The user must
    then write a short comment and submit it. The UI does not write to
    PostgreSQL directly; it stores a pending feedback event in Streamlit
    session state so the page renderer can process it through the feedback
    service on the next rerun.

    Args:
         - item: dict - Assistant chat-history item receiving feedback.
         - raw_user_message: str - User question that produced the answer.

    Returns:
         - return: None - This function does not return a value.
    """
    message_id = str(item.get("message_id") or "")
    saved = st.session_state.setdefault("fm_feedback_saved_message_ids", set())
    if message_id in saved:
        st.caption("Feedback saved.")
        return

    feedback_key = f"fm_feedback_{message_id}"
    selection = st.feedback("thumbs", key=feedback_key)
    if selection is None:
        return

    polarity = "negative" if selection == 0 else "positive"
    placeholder = (
        "What should FurnaceMind repeat next time?"
        if polarity == "positive"
        else "What was wrong or missing in this answer?"
    )

    feedback_text = st.text_input(
        "Feedback details",
        placeholder=placeholder,
        key=f"fm_feedback_text_{message_id}",
        label_visibility="collapsed",
    )
    _, save_col = st.columns([0.82, 0.18])
    with save_col:
        submitted = st.button(
            "Save",
            key=f"fm_feedback_save_{message_id}",
            width="stretch",
        )

    if not submitted:
        return

    feedback_text = feedback_text.strip()
    if not feedback_text:
        st.warning("Please add feedback details before saving.")
        return

    st.session_state["pending_fm_feedback"] = {
        "source": "explicit",
        "polarity": polarity,
        "feedback_text": feedback_text,
        "message_id": message_id,
        "conversation_id": item.get("conversation_id"),
        "raw_user_message": raw_user_message,
        "assistant_response": str(item.get("content") or "").strip(),
    }
    st.rerun()


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


def render_knowledge_sidebar(
    *,
    knowledge_store: Any,
    embedding_client: Any,
    user_id: str | None = None,
    document_repository: Any | None = None,
    chunk_repository: Any | None = None,
    semantic_memory_service: Any | None = None,
) -> None:
    """Render the FurnaceMind MRAG document-library controls in the sidebar.

    The sidebar lets a user upload one or more supported knowledge files, wait
    until they are ready, then explicitly trigger chunking and indexing with the
    ``Chunk & Index`` button. Successful indexing stores vectors in Qdrant and,
    when repositories are available, stores document/chunk metadata in SQL so the
    same files can be listed, inspected, filtered, and removed later.

    Args:
        knowledge_store: Vector store used to write and delete MRAG embeddings.
        embedding_client: Embedding client used by ``process_file`` to create
            multimodal vectors for uploaded file parts.
        user_id: Current FurnaceMind user id. Required for user-owned SQL
            metadata and active-document filtering.
        document_repository: SQL repository for knowledge document rows. When it
            is missing, upload still works but the library list is hidden.
        chunk_repository: SQL repository for knowledge chunk rows. Passed through
            to ingestion so chunk metadata can be persisted with the document.
        semantic_memory_service: Optional long-term memory service used to delete
            document-derived memory facts when a document is removed.

    Returns:
        None. The function writes Streamlit UI elements and uses session state for
        transient upload/remove messages and file-uploader reset behavior.
    """
    with st.sidebar.expander(
        "Multimodal Knowledge", expanded=False, key="fm_knowledge"
    ):
        st.caption("PDF, images, PPTX, Excel, CSV, DOCX, TXT, and Markdown")
        upload_message = st.session_state.pop("fm_knowledge_upload_result", None)
        remove_message = st.session_state.pop("fm_knowledge_remove_result", None)
        memory_warning = st.session_state.pop(
            "fm_knowledge_memory_cleanup_warning", None
        )
        if upload_message:
            st.success(upload_message)
        if remove_message:
            st.success(remove_message)
        if memory_warning:
            st.warning(memory_warning)

        uploader_nonce = st.session_state.setdefault("fm_knowledge_uploader_nonce", 0)
        uploaded_files = st.file_uploader(
            "Upload MRAG files",
            type=KNOWLEDGE_FILE_TYPES,
            accept_multiple_files=True,
            key=f"knowledge_uploader_{uploader_nonce}",
            help="Indexes text, tables, pages, slides, and images into the FurnaceMind MRAG library.",
        )
        upload_status = st.empty()
        selected_files = list(uploaded_files or [])
        if selected_files:
            file_count = len(selected_files)
            st.caption(f"{file_count} file(s) ready to chunk and index.")
        else:
            st.caption("Upload files first, then chunk when you are ready.")
        should_index = st.button(
            "Chunk & Index",
            key="fm_chunk_index_knowledge",
            type="primary",
            use_container_width=True,
            disabled=not selected_files,
        )
        if should_index and selected_files:
            indexed_files = 0
            indexed_parts = 0
            skipped_files: list[str] = []
            with st.spinner("Chunking and indexing knowledge files..."):
                for uploaded in selected_files:
                    parts = process_file(
                        uploaded,
                        knowledge_store,
                        embedding_client,
                        user_id=user_id,
                        document_repository=document_repository,
                        chunk_repository=chunk_repository,
                    )
                    if parts:
                        indexed_files += 1
                        indexed_parts += len(parts)
                    else:
                        skipped_files.append(getattr(uploaded, "name", "file"))
            if indexed_files:
                st.session_state["fm_knowledge_upload_result"] = (
                    f"Indexed {indexed_files} file(s), {indexed_parts} searchable part(s)."
                )
                st.session_state["fm_knowledge_uploader_nonce"] = uploader_nonce + 1
                st.rerun()
            if skipped_files:
                upload_status.warning(
                    f"Skipped unsupported or empty files: {', '.join(skipped_files)}"
                )
        _render_knowledge_documents(
            knowledge_store=knowledge_store,
            user_id=user_id,
            document_repository=document_repository,
            semantic_memory_service=semantic_memory_service,
        )


def _document_metadata(document: Any) -> dict[str, Any]:
    """Return metadata from a knowledge document row as a dictionary.

    SQL repository objects may expose ``metadata_json`` as a dict, ``None``, or a
    backend-specific value. This helper normalizes that shape before the sidebar
    reads MRAG fields such as ``document_id``, ``chunk_count``, ``modalities``,
    and ``qdrant_point_ids``.

    Args:
        document: Knowledge document object returned by the SQL repository.

    Returns:
        The document metadata dictionary, or an empty dict when metadata is not
        available in the expected shape.
    """
    metadata = getattr(document, "metadata_json", None)
    return metadata if isinstance(metadata, dict) else {}


def _document_point_ids(document: Any) -> list[str]:
    """Extract Qdrant point ids recorded for one indexed document.

    Newer document objects may expose point ids through a ``qdrant_point_ids``
    property or method. Older rows may keep the same ids inside ``metadata_json``.
    The remove flow uses these ids to delete exactly the embeddings created for
    the selected document.

    Args:
        document: Knowledge document object selected in the MRAG library UI.

    Returns:
        Clean string point ids. Invalid, blank, or missing ids are ignored.
    """
    point_ids = getattr(document, "qdrant_point_ids", None)
    if callable(point_ids):
        point_ids = point_ids()
    if not isinstance(point_ids, list):
        point_ids = _document_metadata(document).get("qdrant_point_ids")
    if not isinstance(point_ids, list):
        return []
    return [str(point_id) for point_id in point_ids if str(point_id).strip()]


def _remove_knowledge_document(
    *,
    document: Any,
    knowledge_store: Any,
    document_repository: Any,
    user_id: str | None,
    semantic_memory_service: Any | None = None,
) -> tuple[int, int]:
    """Remove one MRAG document and revoke document-derived memories.

    The Qdrant knowledge embeddings are deleted first, then the SQL document row
    is deactivated. Semantic-memory cleanup is best-effort: if it fails, the
    document still stays removed, and the sidebar shows a separate warning.

    Returns:
        ``(deleted_embedding_points, deleted_memory_facts)``.
    """
    metadata = _document_metadata(document)
    mrag_document_id = str(metadata.get("document_id") or "").strip()
    point_ids = _document_point_ids(document)
    deleted_points = 0
    if point_ids:
        deleted_points = knowledge_store.delete_points(point_ids)
    else:
        if not mrag_document_id:
            raise ValueError("No Qdrant point ids found for this document.")
        knowledge_store.delete_document(document_id=mrag_document_id, user_id=user_id)

    document_repository.deactivate_document(document.document_id)
    _remember_revoked_knowledge_document(mrag_document_id)
    st.session_state.pop("fm_last_knowledge_document_refs", None)
    st.session_state.pop("fm_mrag_image_results", None)

    deleted_memories = 0
    if semantic_memory_service is not None and user_id:
        try:
            deleted_memories = int(
                semantic_memory_service.delete_document_related_memories(
                    user_id=user_id,
                    sql_document_id=str(document.document_id),
                    mrag_document_id=mrag_document_id,
                    filename=str(getattr(document, "filename", "") or ""),
                )
                or 0
            )
        except Exception as exc:
            st.session_state["fm_knowledge_memory_cleanup_warning"] = (
                f"Document removed, but related memory cleanup failed: {exc}"
            )
    return deleted_points, deleted_memories


def _render_knowledge_documents(
    *,
    knowledge_store: Any,
    user_id: str | None,
    document_repository: Any | None,
    semantic_memory_service: Any | None = None,
) -> None:
    """Render active MRAG documents and their inspect/remove controls.

    The document repository is treated as the source of truth for what the user
    can currently retrieve from. Active documents are shown in a compact selector,
    with metadata details for debugging and a removal button that clears Qdrant
    embeddings before deactivating the SQL row.

    Args:
        knowledge_store: Vector store used when the selected document is removed.
        user_id: Current FurnaceMind user id. Without it, no user-scoped library
            can be listed.
        document_repository: SQL repository used to fetch active documents and
            deactivate the selected document on removal.

    Returns:
        None. The function renders Streamlit controls and stores success messages
        in session state before rerunning the page.
    """
    if not user_id or document_repository is None:
        return

    try:
        documents = document_repository.list_documents(
            user_id=user_id, active_only=True
        )
    except Exception as exc:
        st.caption(f"Knowledge library unavailable: {exc}")
        return

    if not documents:
        st.caption("No active multimodal knowledge documents.")
        return

    st.caption("MRAG library")
    documents = documents[:20]
    document_by_id = {document.document_id: document for document in documents}
    selected_document_id = st.selectbox(
        "Indexed documents",
        options=list(document_by_id.keys()),
        format_func=lambda document_id: document_by_id[document_id].filename,
        key="fm_selected_knowledge_document",
        label_visibility="collapsed",
    )
    document = document_by_id[selected_document_id]
    metadata = _document_metadata(document)
    point_ids = _document_point_ids(document)
    chunk_count = metadata.get("chunk_count", 0)
    modalities = ", ".join(metadata.get("modalities") or [])
    details = f"{chunk_count} chunks"
    if modalities:
        details = f"{details} | {modalities}"

    st.markdown(f"**{document.filename}**")
    st.caption(details)
    with st.expander("Document details", expanded=False):
        st.caption(f"SQL document: {document.document_id}")
        if metadata.get("document_id"):
            st.caption(f"MRAG document: {metadata['document_id']}")
        st.caption(f"Qdrant: {getattr(document, 'qdrant_collection', '') or 'unknown'}")
        st.caption(f"Embeddings: {len(point_ids)} point(s)")

    if st.button(
        "Remove Selected Document",
        key="fm_remove_selected_knowledge_document",
        help="Delete this document's Qdrant embeddings and remove it from active knowledge.",
        use_container_width=True,
    ):
        try:
            deleted_points, deleted_memories = _remove_knowledge_document(
                document=document,
                knowledge_store=knowledge_store,
                document_repository=document_repository,
                user_id=user_id,
                semantic_memory_service=semantic_memory_service,
            )
            message = (
                f"Removed {document.filename} and {deleted_points} embedding point(s)."
            )
            if deleted_memories:
                message = (
                    f"{message} Deleted {deleted_memories} related memory fact(s)."
                )
            st.session_state["fm_knowledge_remove_result"] = message
            st.rerun()
        except Exception as exc:
            st.warning(f"Could not remove document: {exc}")


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
        if st.button("Unit Cost", width="stretch", key="fm_skill_cost"):
            _queue_skill(
                engine.optimise_prompt(),
                "Unit Cost - last 30 days vs best-shift targets",
                "optimise",
            )
    with c2:
        if st.button("Shift to Best", width="stretch", key="fm_skill_shift"):
            _queue_skill(
                engine.shift_to_best_prompt(str(shift_date), shift_label),
                f"Shift to Best: {shift_date}, Shift {shift_label}",
                "shift_to_best",
            )
    with c3:
        if st.button("Heatloads", width="stretch", key="fm_skill_heatloads"):
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
    user_id: str | None = None,
    document_repository: Any | None = None,
    chunk_repository: Any | None = None,
) -> tuple[str | None, str | None]:
    """Normalize Streamlit chat input into the query used by FurnaceMind.

    Streamlit can return chat input as a simple string, an object with ``text``
    and ``files`` attributes, or a dictionary with the same fields. This helper
    extracts the typed text, indexes any attached files through the MRAG ingestion
    path, and returns both the model query and the user-facing display text.

    Args:
        chat_submission: Raw value returned from ``st.chat_input``.
        knowledge_store: Vector store used to index files attached directly to
            the chat input.
        embedding_client: Embedding client used by ``process_file`` for attached
            files.
        user_id: Current FurnaceMind user id for SQL document/chunk ownership.
        document_repository: Optional SQL document repository passed to
            ingestion when attachments are indexed.
        chunk_repository: Optional SQL chunk repository passed to ingestion when
            attachments are indexed.

    Returns:
        ``(query, display_text)`` when the user submitted text or files.
        ``(None, None)`` when there is no usable submission. If the submission
        contains files but no text, the returned query/display text is an
        attachment summary such as ``"Attached files: report.pdf"``.
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
            process_file(
                uploaded,
                knowledge_store,
                embedding_client,
                user_id=user_id,
                document_repository=document_repository,
                chunk_repository=chunk_repository,
            )

    if typed_query and str(typed_query).strip():
        text = str(typed_query).strip()
        return text, text
    if files:
        file_label = ", ".join(getattr(file_obj, "name", "file") for file_obj in files)
        display = f"Attached files: {file_label}"
        return display, display
    return None, None
