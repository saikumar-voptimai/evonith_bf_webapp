"""API-backed FurnaceMind AI Co-Operate renderer."""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.services import furnacemind_api as api
from apps.frontend_streamlit.utils.shift_windows import last_completed_shift

_KNOWLEDGE_FILE_TYPES = ["pdf", "txt", "md", "markdown", "csv", "json"]


def _token() -> str | None:
    value = str(st.session_state.get("auth_access_token") or "").strip()
    return value or None


def _show_api_error(title: str, exc: FrontendApiError) -> None:
    request_id = f" Request ID: `{exc.request_id}`" if exc.request_id else ""
    code = f" `{exc.error_code}`" if getattr(exc, "error_code", None) else ""
    st.error(f"{title}{code}: {exc.message}.{request_id}")


def _conversation_label(item: dict[str, Any]) -> str:
    return f"{item.get('title') or 'Conversation'} ({str(item.get('id') or '')[-6:]})"


def _ensure_conversation(token: str | None) -> str | None:
    try:
        conversations = api.list_conversations(token=token).get("items", [])
    except FrontendApiError as exc:
        _show_api_error("Could not load conversations", exc)
        return None

    if not conversations:
        try:
            created = api.create_conversation({"title": "FurnaceMind conversation"}, token=token)
        except FrontendApiError as exc:
            _show_api_error("Could not create conversation", exc)
            return None
        conversations = [created]
        st.session_state["fm_api_conversation_id"] = created["id"]

    current_id = st.session_state.get("fm_api_conversation_id")
    labels = {_conversation_label(item): item["id"] for item in conversations}
    values = list(labels.values())
    index = values.index(current_id) if current_id in values else 0
    selected = st.sidebar.selectbox("Conversation", list(labels), index=index, key="fm_api_conversation_select")
    conversation_id = labels[selected]
    st.session_state["fm_api_conversation_id"] = conversation_id

    cols = st.sidebar.columns(2)
    if cols[0].button("New Chat", key="fm_api_new_chat", width="stretch"):
        try:
            created = api.create_conversation({"title": "FurnaceMind conversation"}, token=token)
            st.session_state["fm_api_conversation_id"] = created["id"]
            st.rerun()
        except FrontendApiError as exc:
            _show_api_error("Could not create conversation", exc)
    if cols[1].button("Finalize", key="fm_api_finalize_chat", width="stretch"):
        try:
            api.finalize_conversation(conversation_id, token=token)
            st.rerun()
        except FrontendApiError as exc:
            _show_api_error("Could not finalize conversation", exc)
    return conversation_id


def _render_documents_sidebar(token: str | None) -> list[str]:
    selected_ids: list[str] = []
    with st.sidebar.expander("Documents", expanded=False):
        uploads = st.file_uploader("Upload", type=_KNOWLEDGE_FILE_TYPES, accept_multiple_files=True, key="fm_api_sidebar_upload")
        if uploads and st.button("Upload & Index", key="fm_api_upload_index", width="stretch"):
            for file in uploads:
                try:
                    uploaded = api.upload_document(file, token=token)
                    task = api.index_document(uploaded["id"], token=token)
                    st.caption(f"{uploaded.get('filename')} -> {task.get('status')}")
                except FrontendApiError as exc:
                    _show_api_error("Document upload failed", exc)
        try:
            docs = api.list_documents(token=token).get("items", [])
        except FrontendApiError as exc:
            _show_api_error("Could not load documents", exc)
            docs = []
        options = {f"{doc.get('filename')} [{doc.get('status')}]	{doc.get('id')[-6:]}": doc.get("id") for doc in docs}
        if options:
            selected_ids = st.multiselect("Attach to next run", list(options), key="fm_api_doc_select")
            selected_ids = [options[label] for label in selected_ids]
        if docs:
            st.dataframe(docs, width="stretch", hide_index=True)
    return selected_ids


def _render_skills_sidebar(token: str | None) -> str | None:
    selected_instruction = None
    with st.sidebar.expander("Skills", expanded=False):
        try:
            skills = api.list_skills(token=token).get("items", [])
        except FrontendApiError as exc:
            _show_api_error("Could not load skills", exc)
            skills = []
        if skills:
            labels = {f"{item.get('symbol') or '*'} {item.get('name')}": item for item in skills if item.get("is_active")}
            if labels:
                label = st.selectbox("Configured skills", ["None", *labels], key="fm_api_skill_select")
                if label != "None":
                    selected_instruction = str(labels[label].get("instruction") or "")
                    if st.button("Reindex Skill", key="fm_api_reindex_skill", width="stretch"):
                        try:
                            api.index_skill(labels[label]["id"], token=token)
                            st.toast("Skill indexing queued.")
                        except FrontendApiError as exc:
                            _show_api_error("Could not index skill", exc)
        with st.form("fm_api_create_skill_form", clear_on_submit=True):
            st.caption("Add skill")
            name = st.text_input("Name", key="fm_api_skill_name")
            symbol = st.text_input("Symbol", max_chars=8, key="fm_api_skill_symbol")
            instruction = st.text_area("Instruction", height=100, key="fm_api_skill_instruction")
            submitted = st.form_submit_button("Create", width="stretch")
        if submitted:
            try:
                created = api.create_skill({"name": name, "symbol": symbol, "instruction": instruction}, token=token)
                api.index_skill(created["id"], token=token)
                st.success("Skill created.")
                st.rerun()
            except FrontendApiError as exc:
                _show_api_error("Could not create skill", exc)
    return selected_instruction


def _render_tools_sidebar(token: str | None) -> None:
    with st.sidebar.expander("Tools", expanded=False):
        try:
            st.dataframe(api.list_tools(token=token), width="stretch", hide_index=True)
        except FrontendApiError as exc:
            _show_api_error("Could not load tools", exc)


def _render_message(item: dict[str, Any], token: str | None) -> None:
    role = item.get("role") if item.get("role") in {"user", "assistant"} else "assistant"
    with st.chat_message(role):
        st.markdown(item.get("content") or "")
        for warning in item.get("warnings", []):
            st.warning(warning.get("message", warning))
        for artifact in item.get("artifacts", []):
            artifact_id = artifact.get("artifact_id")
            if artifact_id:
                st.link_button(f"Download {artifact.get('filename', 'artifact')}", api.download_artifact_url(artifact_id))
        if role == "assistant" and item.get("id"):
            cols = st.columns([0.2, 0.2, 0.6])
            if cols[0].button("Helpful", key=f"fm_api_helpful_{item['id']}"):
                try:
                    api.submit_message_feedback(item["id"], {"helpful": True}, token=token)
                    st.toast("Feedback saved.")
                except FrontendApiError as exc:
                    _show_api_error("Could not save feedback", exc)
            if cols[1].button("Issue", key=f"fm_api_issue_{item['id']}"):
                st.session_state["fm_api_feedback_target"] = item["id"]
        if st.session_state.get("fm_api_feedback_target") == item.get("id"):
            comment = st.text_input("Feedback details", key=f"fm_api_feedback_text_{item['id']}", label_visibility="collapsed")
            if st.button("Save feedback", key=f"fm_api_feedback_save_{item['id']}"):
                try:
                    api.submit_message_feedback(item["id"], {"helpful": False, "comment": comment}, token=token)
                    st.session_state.pop("fm_api_feedback_target", None)
                    st.rerun()
                except FrontendApiError as exc:
                    _show_api_error("Could not save feedback", exc)


def render_ai_cooperate_api(*, field_labels: dict) -> None:  # noqa: ARG001
    token = _token()
    try:
        config = api.get_furnacemind_config(token=token)
    except FrontendApiError as exc:
        if getattr(exc, "error_code", None) == "AUTH_REQUIRED":
            st.warning("Login is required to use FurnaceMind.")
        else:
            _show_api_error("Could not load FurnaceMind configuration", exc)
        return

    st.sidebar.caption(f"LLM: {'ready' if config.get('provider_configured') else 'offline'}")
    for item in config.get("warnings", [])[:4]:
        st.sidebar.caption(item.get("message", item))

    conversation_id = _ensure_conversation(token)
    if not conversation_id:
        return
    selected_doc_ids = _render_documents_sidebar(token)
    skill_instruction = _render_skills_sidebar(token)
    _render_tools_sidebar(token)

    try:
        messages = api.list_messages(conversation_id, token=token)
    except FrontendApiError as exc:
        _show_api_error("Could not load messages", exc)
        messages = []
    for item in messages:
        if item.get("role") in {"user", "assistant"}:
            _render_message(item, token)

    default_date, default_label = last_completed_shift()
    skill_prefix = f"Use this skill context for the answer:\n{skill_instruction}\n\n" if skill_instruction else ""
    prompt = st.chat_input("Ask FurnaceMind or attach a document...", accept_file="multiple", file_type=_KNOWLEDGE_FILE_TYPES, key="furnacemind_api_chat_input")
    if not prompt:
        return

    text = getattr(prompt, "text", None) if not isinstance(prompt, str) else prompt
    files = getattr(prompt, "files", None) if not isinstance(prompt, str) else []
    document_ids = list(selected_doc_ids)
    for file in files or []:
        try:
            uploaded = api.upload_document(file, token=token)
            document_ids.append(uploaded["id"])
            api.index_document(uploaded["id"], token=token)
        except FrontendApiError as exc:
            _show_api_error("Chat attachment upload failed", exc)
            return

    message = (skill_prefix + str(text or "")).strip()
    if not message:
        return
    allow_llm = bool(config.get("llm_enabled") and config.get("provider_configured"))
    try:
        run = api.start_run(
            conversation_id,
            {
                "message": message,
                "document_ids": document_ids,
                "allow_llm": allow_llm,
                "tool_mode": "auto",
                "options": {"export": False, "default_shift_date": default_date.isoformat(), "default_shift_label": default_label},
            },
            token=token,
        )
        status = api.get_run(run["id"], token=token)
    except FrontendApiError as exc:
        _show_api_error("FurnaceMind run failed", exc)
        return

    with st.chat_message("assistant"):
        if status.get("status") == "waiting_for_documents":
            st.warning("Selected documents are still indexing. Try again after indexing completes.")
        elif status.get("result_message"):
            st.markdown(status["result_message"].get("content") or "")
        else:
            st.caption(f"Run status: {status.get('status')}")
        for event in status.get("events", [])[-5:]:
            st.caption(event.get("event_type"))
    st.rerun()