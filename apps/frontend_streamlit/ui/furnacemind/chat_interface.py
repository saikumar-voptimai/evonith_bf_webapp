"""Streamlit chat UI helpers for FurnaceMind.

This module owns the chat-facing controls: rendering messages, capturing
feedback, indexing/removing MRAG knowledge documents, and managing the
DB-backed quick-skill sidebar. It intentionally keeps UI state and Streamlit
widgets here while delegating execution to the agent, repositories, ingestion
pipeline, and skill registry.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from apps.frontend_streamlit.agents.multimodal.ingestion import process_file
from apps.frontend_streamlit.utils.furnacemind.skill_ui import (
    build_skill_context_preview,
    display_source_type,
    format_skill_option,
    format_skill_timestamp,
    html_text,
    is_built_in_skill,
    next_skill_order,
    skill_metadata,
    skill_order,
    skill_slug,
    skill_source_type,
    skill_symbol,
    skill_symbol_conflicts,
    store_skill_markdown,
    unique_skill_slug,
)

if TYPE_CHECKING:
    from apps.frontend_streamlit.agents.furnacemind.skill_registry import SkillDefinition, SkillRegistry
    from apps.frontend_streamlit.agents.furnacemind.skills import SkillEngine

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


def _metadata_values(metadata: dict[str, Any], *keys: str) -> list[str]:
    """Return normalized string values from scalar/list metadata fields."""
    values: list[str] = []
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, str):
            raw_values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            raw_values = list(raw)
        elif raw is None:
            raw_values = []
        else:
            raw_values = [raw]
        values.extend(str(item).strip() for item in raw_values if str(item).strip())
    return values


def _skill_history_key(value: Any) -> str:
    """Normalize skill ids/slugs for current-session history filtering."""
    return str(value or "").strip().lower()


def _message_skill_keys(item: dict[str, Any]) -> set[str]:
    """Return skill identifiers attached to one chat-history item."""
    metadata = _message_metadata(item)
    return {
        _skill_history_key(value)
        for value in _metadata_values(
            metadata,
            "skill_context_skill_ids",
            "skill_context_skill_slugs",
            "skill_ids",
            "skill_slugs",
            "selected_skill_id",
        )
        if _skill_history_key(value)
    }


def _message_mentions_inactive_skill_context(item: dict[str, Any]) -> bool:
    """Return True when old text cites a file from an inactive skill."""
    filenames = st.session_state.get("fm_inactive_skill_context_files") or set()
    if not isinstance(filenames, (set, list, tuple)) or not filenames:
        return False
    text = f"{item.get('content') or ''} {item.get('display') or ''}".lower()
    return any(str(filename).lower() in text for filename in filenames if filename)


def _is_inactive_skill_message(item: dict[str, Any]) -> bool:
    """Return True when a prior message depends on a now-inactive skill."""
    inactive_keys = st.session_state.get("fm_inactive_skill_keys") or set()
    if isinstance(inactive_keys, (set, list, tuple)):
        normalized_inactive = {_skill_history_key(key) for key in inactive_keys}
        if _message_skill_keys(item) & normalized_inactive:
            return True
    return _message_mentions_inactive_skill_context(item)


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
        if _is_inactive_skill_message(item):
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


def _render_selected_skill_details(
    *,
    skill: Any,
    slug: str,
    handler: str,
    context_files: list[Any],
    symbol: str,
    metadata: dict[str, Any],
) -> None:
    """Render the selected skill summary before the user enters edit mode.

    The details panel shows the operator-facing configuration, context files, and
    technical metadata without exposing editable inputs by default. This keeps the
    sidebar scannable and avoids accidental edits when users only want to inspect
    an active skill.
    """
    name = str(getattr(skill, "name", "") or "Skill").strip() or "Skill"
    description = str(getattr(skill, "description", "") or "").strip()
    instruction = str(getattr(skill, "instruction", "") or "").strip()
    display_text = str(metadata.get("display_template") or name).strip()
    is_active = bool(getattr(skill, "is_active", False))
    status = "Active" if is_active else "Inactive"
    status_color = "#16a34a" if is_active else "#94a3b8"
    source_type = display_source_type(skill_source_type(skill))
    updated_at = format_skill_timestamp(getattr(skill, "updated_at", None))
    symbol_text = symbol or "\u2022"

    st.markdown(
        f"""
<div style="border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 8px; padding: 10px 12px; margin: 6px 0 10px 0;">
  <div style="display: flex; gap: 10px; align-items: center;">
    <div style="width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; background: rgba(255, 75, 75, 0.12); color: #ff4b4b; font-weight: 700; flex: 0 0 auto;">{html_text(symbol_text)}</div>
    <div style="min-width: 0;">
      <div style="font-weight: 700; line-height: 1.2;">{html_text(name)}</div>
      <div style="font-size: 0.76rem; opacity: 0.72; margin-top: 3px;"><span style="color: {status_color}; font-weight: 700;">{status}</span> &middot; {html_text(source_type)}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if description:
        st.caption("Description")
        st.markdown(html_text(description), unsafe_allow_html=True)
    else:
        st.caption("No description saved.")

    st.caption("Instruction")
    if instruction:
        with st.expander("View instruction", expanded=False):
            st.code(instruction, language="markdown")
    else:
        st.caption("No instruction saved.")

    if context_files:
        st.caption("Context files")
        files = [f"- `{Path(str(item)).name}`" for item in context_files]
        st.markdown("\n".join(files))

    preview = build_skill_context_preview(
        instruction=instruction,
        context_files=context_files,
    )
    if preview:
        with st.expander("Prompt / context preview", expanded=False):
            st.caption("This is the context added when the skill runs.")
            st.code(preview, language="markdown")

    with st.expander("Technical metadata", expanded=False):
        st.caption(f"Skill id: {skill.skill_id}")
        st.caption(f"Slug: {slug}")
        st.caption(f"Handler: {handler or 'prompt-only'}")
        if updated_at:
            st.caption(f"Last updated: {updated_at}")
        st.caption(f"Chat display: {display_text}")


def _skill_context_file_names_from_metadata(metadata: dict[str, Any]) -> set[str]:
    """Return safe context filenames configured for one skill metadata payload."""
    raw_files: list[Any] = []
    for key in ("context_files", "skill_files", "skill_file"):
        value = metadata.get(key)
        if isinstance(value, str):
            raw_files.append(value)
        elif isinstance(value, (list, tuple, set)):
            raw_files.extend(value)
    return {
        Path(str(filename or "")).name.lower()
        for filename in raw_files
        if Path(str(filename or "")).name
    }


def _skill_session_keys(skill: Any) -> set[str]:
    """Return identifiers used to mark chat messages tied to one skill."""
    keys = {
        _skill_history_key(getattr(skill, "skill_id", "")),
        _skill_history_key(skill_slug(skill)),
        _skill_history_key(getattr(skill, "name", "")),
    }
    return {key for key in keys if key}


def _remember_inactive_skill_filters(skills: list[Any]) -> None:
    """Cache inactive skill ids/files so old skill messages stop feeding prompts."""
    inactive_keys: set[str] = set()
    inactive_files: set[str] = set()
    for skill in skills:
        if bool(getattr(skill, "is_active", False)):
            continue
        metadata = skill_metadata(skill)
        inactive_keys.update(_skill_session_keys(skill))
        inactive_files.update(_skill_context_file_names_from_metadata(metadata))
    st.session_state["fm_inactive_skill_keys"] = inactive_keys
    st.session_state["fm_inactive_skill_context_files"] = inactive_files


def render_skill_sidebar(
    *,
    skill_repository: Any | None,
    skill_vector_store: Any | None = None,
    user_id: str | None = None,
) -> None:
    """Render the sidebar panel for managing DB-backed FurnaceMind skills.

    Users can add prompt-only skills, inspect existing rows, edit allowed fields,
    and activate/deactivate skills. The UI deliberately does not expose arbitrary
    Python handler editing; built-in handler metadata is preserved so execution
    remains limited to the whitelist enforced by ``SkillRegistry``.
    """
    with st.sidebar.expander("Skills", expanded=False, key="fm_skills"):
        result_message = st.session_state.pop("fm_skill_result", None)
        error_message = st.session_state.pop("fm_skill_error", None)
        if result_message:
            st.success(result_message)
        if error_message:
            st.warning(error_message)

        if skill_repository is None:
            st.session_state["fm_inactive_skill_keys"] = set()
            st.session_state["fm_inactive_skill_context_files"] = set()
            st.caption(
                "Skill database unavailable. Built-in quick skills are still available."
            )
            return

        try:
            skills = list(skill_repository.list_skills(active_only=False))
        except Exception as exc:
            st.session_state["fm_inactive_skill_keys"] = set()
            st.session_state["fm_inactive_skill_context_files"] = set()
            st.caption(f"Skill library unavailable: {exc}")
            return

        _remember_inactive_skill_filters(skills)

        _ensure_skills_vectorized(
            skill_vector_store=skill_vector_store,
            skills=skills,
        )

        _render_create_skill_form(
            skill_repository=skill_repository,
            skill_vector_store=skill_vector_store,
            user_id=user_id,
            existing_skills=skills,
        )
        _render_skill_library(
            skill_repository=skill_repository,
            skill_vector_store=skill_vector_store,
            skills=skills,
        )


def _render_create_skill_form(
    *,
    skill_repository: Any,
    skill_vector_store: Any | None,
    user_id: str | None,
    existing_skills: list[Any],
) -> None:
    """Render the Add Skill button and create form.

    The form is hidden until the user clicks ``Add Skill``. Creating a skill
    stores its markdown context file, writes the SQL row, records safe metadata
    used by the registry, and reruns the page with a success or error message.
    """
    show_form = bool(st.session_state.get("fm_show_create_skill_form", False))
    if not show_form:
        if st.button("Add Skill", key="fm_add_skill", use_container_width=True):
            st.session_state["fm_show_create_skill_form"] = True
            st.rerun()
        return

    heading_cols = st.columns([1, 1])
    heading_cols[0].caption("Add skill")
    if heading_cols[1].button(
        "Cancel",
        key="fm_cancel_create_skill",
        use_container_width=True,
    ):
        st.session_state["fm_show_create_skill_form"] = False
        st.rerun()

    with st.form("fm_create_skill_form", clear_on_submit=True):
        symbol = st.text_input(
            "Symbol",
            placeholder="Example: â‚¹",
            max_chars=8,
            help="Unique short symbol shown before the skill name.",
        )
        name = st.text_input("Skill name", placeholder="Example: Campaign Checker")
        description = st.text_input(
            "Description",
            placeholder="Short helper text shown in the skill library",
        )
        instruction = st.text_area(
            "Instruction",
            placeholder="Tell FurnaceMind how this skill should behave.",
            height=120,
        )
        skill_file = st.file_uploader(
            "Skill.md file",
            type=["md"],
            accept_multiple_files=False,
            help="Upload the markdown context used by the system when this skill runs.",
        )
        is_active = st.checkbox("Active", value=True)
        submitted = st.form_submit_button("Create Skill", use_container_width=True)

    if not submitted:
        return

    clean_symbol = str(symbol or "").strip()
    clean_name = str(name or "").strip()
    clean_instruction = str(instruction or "").strip()
    if (
        not clean_symbol
        or not clean_name
        or not clean_instruction
        or skill_file is None
    ):
        st.session_state["fm_skill_error"] = (
            "Symbol, skill name, instruction, and Skill.md file are required."
        )
        st.rerun()
    if skill_symbol_conflicts(clean_symbol, existing_skills):
        st.session_state["fm_skill_error"] = f"Symbol already used: {clean_symbol}"
        st.rerun()

    slug = unique_skill_slug(clean_name, existing_skills)
    try:
        stored_filename = store_skill_markdown(skill_file, slug=slug)
        metadata = {
            "slug": slug,
            "symbol": clean_symbol,
            "order": next_skill_order(existing_skills),
            "display_template": clean_name,
            "context_files": [stored_filename],
            "skill_file": stored_filename,
        }
        created_skill = skill_repository.create_skill(
            name=clean_name,
            symbol=clean_symbol,
            description=str(description or "").strip() or None,
            instruction=clean_instruction,
            source_type="uploaded",
            qdrant_collection=getattr(skill_vector_store, "collection_name", None),
            is_active=bool(is_active),
            created_by=str(user_id) if user_id else None,
            metadata=metadata,
        )
        vector_count, vector_error = _sync_skill_vector_index(
            skill_vector_store=skill_vector_store,
            skill=created_skill,
        )
    except Exception as exc:
        st.session_state["fm_skill_error"] = f"Could not create skill: {exc}"
    else:
        st.session_state["fm_show_create_skill_form"] = False
        result = f"Created skill: {clean_name}"
        if vector_count:
            result += f". Indexed {vector_count} skill vector point(s)."
        st.session_state["fm_skill_result"] = result
        if vector_error:
            st.session_state["fm_skill_error"] = (
                f"Created skill, but vector indexing failed: {vector_error}"
            )
    st.rerun()


def _render_skill_library(
    *,
    skill_repository: Any,
    skill_vector_store: Any | None,
    skills: list[Any],
) -> None:
    """Render the skill library selector, details panel, and edit form.

    The selected row is shown read-only first. Clicking ``Edit Skill`` opens the
    form. Built-in rows can only change symbol and active state; uploaded/custom
    rows can also change name, description, instruction, and chat display text.
    """
    if not skills:
        st.caption(
            "No database skills yet. Built-ins will be used until you seed or create skills."
        )
        return

    st.caption("Skill library")
    ordered_skills = sorted(
        skills,
        key=lambda skill: (
            skill_order(skill),
            str(getattr(skill, "name", "")).lower(),
        ),
    )
    skill_by_id = {str(skill.skill_id): skill for skill in ordered_skills}
    selected_skill_id = st.selectbox(
        "Configured skills",
        options=list(skill_by_id.keys()),
        format_func=lambda skill_id: format_skill_option(skill_by_id[skill_id]),
        key="fm_selected_skill",
        label_visibility="collapsed",
    )
    skill = skill_by_id[selected_skill_id]
    metadata = {**skill_metadata(skill)}
    slug = skill_slug(skill)
    handler = str(metadata.get("handler") or "").strip()
    context_files = metadata.get("context_files") or []
    if not isinstance(context_files, list):
        context_files = []

    symbol = skill_symbol(skill)
    is_built_in = is_built_in_skill(skill)
    _render_selected_skill_details(
        skill=skill,
        slug=slug,
        handler=handler,
        context_files=context_files,
        symbol=symbol,
        metadata=metadata,
    )

    is_editing = st.session_state.get("fm_edit_skill_id") == selected_skill_id
    if not is_editing:
        if st.button(
            "Edit Skill",
            key=f"fm_edit_skill_{selected_skill_id}",
            use_container_width=True,
        ):
            st.session_state["fm_edit_skill_id"] = selected_skill_id
            st.rerun()
        return

    edit_cols = st.columns([1, 1])
    edit_cols[0].caption("Editing skill")
    if edit_cols[1].button(
        "Cancel",
        key=f"fm_cancel_edit_skill_{selected_skill_id}",
        use_container_width=True,
    ):
        st.session_state.pop("fm_edit_skill_id", None)
        st.rerun()

    original_name = str(getattr(skill, "name", "") or "")
    original_description = str(getattr(skill, "description", "") or "")
    original_instruction = str(getattr(skill, "instruction", "") or "")
    original_display = str(metadata.get("display_template") or original_name)
    if is_built_in:
        st.caption("Built-in skill: only symbol and active state can be edited.")

    with st.form(f"fm_update_skill_form_{selected_skill_id}"):
        updated_symbol = st.text_input(
            "Symbol",
            value=symbol,
            max_chars=8,
            help="Unique short symbol shown before the skill name.",
        )
        updated_name = st.text_input(
            "Skill name",
            value=original_name,
            disabled=is_built_in,
        )
        updated_description = st.text_input(
            "Description",
            value=original_description,
            disabled=is_built_in,
        )
        updated_instruction = st.text_area(
            "Instruction",
            value=original_instruction,
            height=140,
            disabled=is_built_in,
        )
        updated_display = st.text_input(
            "Chat display text",
            value=original_display,
            disabled=is_built_in,
        )
        updated_active = st.checkbox(
            "Active", value=bool(getattr(skill, "is_active", False))
        )
        submitted = st.form_submit_button("Save Skill", use_container_width=True)

    if not submitted:
        return

    clean_symbol = str(updated_symbol or "").strip()
    clean_name = str(original_name if is_built_in else updated_name or "").strip()
    clean_description = str(
        original_description if is_built_in else updated_description or ""
    ).strip()
    clean_instruction = str(
        original_instruction if is_built_in else updated_instruction or ""
    ).strip()
    clean_display = str(
        original_display if is_built_in else updated_display or clean_name
    ).strip()
    if (
        not clean_symbol
        or not clean_name
        or (not is_built_in and not clean_instruction)
    ):
        st.session_state["fm_skill_error"] = (
            "Symbol, skill name, and instruction are required."
        )
        st.rerun()
    if skill_symbol_conflicts(
        clean_symbol,
        skills,
        exclude_skill_id=selected_skill_id,
    ):
        st.session_state["fm_skill_error"] = f"Symbol already used: {clean_symbol}"
        st.rerun()

    metadata.update(
        {
            "slug": slug,
            "symbol": clean_symbol,
            "order": skill_order(skill),
            "display_template": clean_display or clean_name,
        }
    )
    try:
        updated_skill = skill_repository.update_skill(
            skill_id=selected_skill_id,
            name=clean_name,
            symbol=clean_symbol,
            description=clean_description or None,
            instruction=clean_instruction,
            qdrant_collection=getattr(skill_vector_store, "collection_name", None),
            is_active=bool(updated_active),
            metadata=metadata,
        )
        vector_count, vector_error = _sync_skill_vector_index(
            skill_vector_store=skill_vector_store,
            skill=updated_skill or skill,
        )
    except Exception as exc:
        st.session_state["fm_skill_error"] = f"Could not update skill: {exc}"
    else:
        state = "active" if updated_active else "inactive"
        st.session_state.pop("fm_edit_skill_id", None)
        result = f"Saved {clean_name} ({state})."
        if vector_count:
            result += f" Indexed {vector_count} skill vector point(s)."
        st.session_state["fm_skill_result"] = result
        if vector_error:
            st.session_state["fm_skill_error"] = (
                f"Saved skill, but vector indexing failed: {vector_error}"
            )
    st.rerun()


def _sync_skill_vector_index(
    *,
    skill_vector_store: Any | None,
    skill: Any | None,
) -> tuple[int, str | None]:
    """Upsert Qdrant skill vectors after SQL has been saved.

    SQL is the source of truth, so vector sync is best-effort. Active and
    inactive skills are both embedded; the active flag is stored in payload, and
    runtime retrieval still filters against active SQL skill ids.
    """
    if skill_vector_store is None or skill is None:
        return 0, None
    try:
        point_ids = skill_vector_store.index_skill(skill)
        return len(point_ids), None
    except Exception as exc:
        return 0, str(exc)


def _skill_vector_sync_signature(
    *,
    skill_vector_store: Any,
    skills: list[Any],
) -> tuple[Any, ...]:
    """Return the skill state used to avoid repeated automatic reindexing.

    The sidebar should backfill vectors for SQL skills that existed before the
    Qdrant skill index was introduced, but embedding every skill on every
    Streamlit rerun would be wasteful. This signature changes when the active
    flag, prompt-facing fields, context files, or target collection changes.
    """
    collection_name = str(getattr(skill_vector_store, "collection_name", "") or "")
    signature: list[tuple[Any, ...]] = []
    for skill in skills:
        metadata = skill_metadata(skill)
        context_files = (
            metadata.get("context_files") or metadata.get("skill_files") or []
        )
        if isinstance(context_files, str):
            safe_context_files = (context_files,)
        elif isinstance(context_files, (list, tuple, set)):
            safe_context_files = tuple(str(item) for item in context_files)
        else:
            safe_context_files = ()
        signature.append(
            (
                str(getattr(skill, "skill_id", "") or ""),
                bool(getattr(skill, "is_active", False)),
                str(getattr(skill, "name", "") or ""),
                skill_symbol(skill),
                str(getattr(skill, "description", "") or ""),
                str(getattr(skill, "instruction", "") or ""),
                str(getattr(skill, "source_type", "") or ""),
                str(getattr(skill, "updated_at", "") or ""),
                tuple(sorted(safe_context_files)),
                collection_name,
            )
        )
    return tuple(sorted(signature))


def _ensure_skills_vectorized(
    *,
    skill_vector_store: Any | None,
    skills: list[Any],
) -> None:
    """Backfill Qdrant vectors for existing SQL skills once per skill state.

    Create/edit actions already sync one skill immediately. This function covers
    the other case: rows that were already present in ``furnace_mind.skills``
    before the vector-backed skill retrieval feature was added. Active and
    inactive rows are embedded; runtime retrieval filters against active SQL
    skill ids so disabled skills are not injected.
    """
    if skill_vector_store is None or not skills:
        return

    signature = _skill_vector_sync_signature(
        skill_vector_store=skill_vector_store,
        skills=skills,
    )
    collection_name = str(
        getattr(skill_vector_store, "collection_name", "default") or "default"
    )
    state_key = f"fm_skill_vector_sync_signature_{collection_name}"
    if st.session_state.get(state_key) == signature:
        return

    indexed_points = 0
    first_error: str | None = None
    for skill in skills:
        point_count, error = _sync_skill_vector_index(
            skill_vector_store=skill_vector_store,
            skill=skill,
        )
        indexed_points += point_count
        if error and first_error is None:
            first_error = error

    st.session_state[state_key] = signature
    if first_error:
        st.caption(f"Skill vector sync incomplete: {first_error}")
    elif indexed_points:
        st.caption(f"Skill vector index ready: {indexed_points} point(s).")


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


def _queue_skill(
    prompt: str,
    display: str,
    skill_id: str,
    *,
    skill_context: str | None = None,
) -> None:
    """Store a selected quick skill as the next pending chat action.

    The actual agent call happens in the page renderer after Streamlit reruns.
    This helper clears stale chart/dataframe artifacts, records the prompt shown
    to the agent, the display text shown to the user, the skill id, and optional
    DB-provided context from ``SkillRegistry``.
    """
    st.session_state.pop("fm_fig", None)
    st.session_state.pop("fm_df", None)
    st.session_state.pop("fm_df_meta", None)
    st.session_state["pending_skill_prompt"] = {
        "prompt": prompt,
        "display": display,
        "skill_id": skill_id,
        "skill_context": skill_context,
    }
    st.rerun()


def _skill_button_key(skill: "SkillDefinition") -> str:
    """Return a Streamlit-safe widget key for a DB-backed skill button."""
    safe_slug = "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in str(skill.slug)
    )
    return f"fm_skill_{safe_slug or 'skill'}"


def _render_database_skills(
    registry: "SkillRegistry", shift_date: date, shift_label: str
) -> None:
    """Render quick-skill buttons from ``SkillRegistry`` definitions.

    Each button asks the registry to build a ``SkillExecution`` payload. That
    keeps button rendering in the UI layer while preserving registry ownership of
    DB fallback rules, handler dispatch, and skill-context construction.
    """
    skills = registry.available_skills()
    if not skills:
        st.caption("No active quick skills.")
        return

    columns = st.columns(min(3, len(skills)))
    for index, skill in enumerate(skills):
        with columns[index % len(columns)]:
            if st.button(
                skill.button_label,
                width="stretch",
                key=_skill_button_key(skill),
                help=skill.description or None,
            ):
                try:
                    execution = registry.execute_definition(
                        skill,
                        shift_date=shift_date,
                        shift_label=shift_label,
                    )
                except Exception as exc:
                    st.warning(f"Could not start skill: {exc}")
                    return
                _queue_skill(
                    execution.prompt,
                    execution.display,
                    execution.skill_id,
                    skill_context=execution.skill_context,
                )


def render_quick_skills(
    engine: "SkillEngine | SkillRegistry", shift_date: date, shift_label: str
) -> None:
    """Render quick-skill buttons directly above the chat input.

    Newer pages pass a ``SkillRegistry`` so buttons are database-driven. The
    fallback branch accepts the older ``SkillEngine`` directly, preserving the
    original three hardcoded buttons when the registry is not available.
    """
    st.caption("Quick skills")
    if hasattr(engine, "available_skills") and hasattr(engine, "execute_definition"):
        _render_database_skills(engine, shift_date, shift_label)
        return

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
