"""FurnaceMind page — AI co-pilot for blast furnace operations.

Two tabs are available:
  AI Co-Operate — conversational agent with tool-calling and skill buttons.
  Reports       — live and saved shift handover reports.
"""

from __future__ import annotations

import streamlit as st

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.services.furnacemind_api import (
    create_conversation,
    download_artifact_url,
    get_furnacemind_config,
    get_run,
    list_conversations,
    list_documents,
    list_messages,
    list_tools,
    start_run,
    submit_message_feedback,
    upload_document,
)


def _api_token() -> str | None:
    token = str(st.session_state.get("auth_access_token") or "").strip()
    return token or None


def _show_api_error(title: str, exc: FrontendApiError) -> None:
    request_id = f" Request ID: `{exc.request_id}`" if exc.request_id else ""
    code = f" `{exc.error_code}`" if getattr(exc, "error_code", None) else ""
    st.error(f"{title}{code}: {exc.message}.{request_id}")


def _render_api_mode() -> None:
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title("FurnaceMind")
    st.caption("FurnaceMind mode: Backend API")
    token = _api_token()

    try:
        config = get_furnacemind_config(token=token)
    except FrontendApiError as exc:
        if getattr(exc, "error_code", None) == "AUTH_REQUIRED":
            st.warning("Login is required to use backend FurnaceMind mode.")
        else:
            _show_api_error("Could not load FurnaceMind configuration", exc)
        return

    with st.sidebar:
        st.header("FurnaceMind")
        st.caption(f"LLM: {'enabled' if config.get('llm_enabled') else 'disabled'}")
        st.caption(f"Memory: {'enabled' if config.get('memory_enabled') else 'disabled'}")
        st.caption(f"Tools: {'enabled' if config.get('tools_enabled') else 'disabled'}")
        if config.get("warnings"):
            for item in config["warnings"][:4]:
                st.caption(item.get("message", item))

    try:
        conversations = list_conversations(token=token)
    except FrontendApiError as exc:
        _show_api_error("Could not load conversations", exc)
        return

    items = conversations.get("items", [])
    current_id = st.session_state.get("fm_api_conversation_id")
    if not items:
        try:
            created = create_conversation({"title": "FurnaceMind API conversation"}, token=token)
            current_id = created["id"]
            st.session_state["fm_api_conversation_id"] = current_id
            items = [created]
        except FrontendApiError as exc:
            _show_api_error("Could not create conversation", exc)
            return

    labels = {f"{item.get('title') or item['id']} ({item['id'][-6:]})": item["id"] for item in items}
    selected_label = st.selectbox(
        "Conversation",
        list(labels),
        index=max(0, list(labels.values()).index(current_id)) if current_id in labels.values() else 0,
    )
    conversation_id = labels[selected_label]
    st.session_state["fm_api_conversation_id"] = conversation_id

    if st.button("New Conversation", key="fm_api_new_conversation"):
        try:
            created = create_conversation({"title": "FurnaceMind API conversation"}, token=token)
            st.session_state["fm_api_conversation_id"] = created["id"]
            st.rerun()
        except FrontendApiError as exc:
            _show_api_error("Could not create conversation", exc)

    tab_chat, tab_docs, tab_tools = st.tabs(["Chat", "Documents", "Tools"])

    with tab_chat:
        try:
            messages = list_messages(conversation_id, token=token)
        except FrontendApiError as exc:
            _show_api_error("Could not load messages", exc)
            messages = []
        for item in messages:
            if item.get("role") not in {"user", "assistant"}:
                continue
            with st.chat_message(item["role"]):
                st.markdown(item.get("content") or "")
                for warning in item.get("warnings", []):
                    st.warning(warning.get("message", warning))
                for artifact in item.get("artifacts", []):
                    artifact_id = artifact.get("artifact_id")
                    if artifact_id:
                        st.link_button(
                            f"Download {artifact.get('filename', 'artifact')}",
                            download_artifact_url(artifact_id),
                        )
                if item.get("role") == "assistant":
                    if st.button("Helpful", key=f"fm_api_helpful_{item['id']}"):
                        try:
                            submit_message_feedback(item["id"], {"helpful": True}, token=token)
                            st.toast("Feedback saved.")
                        except FrontendApiError as exc:
                            _show_api_error("Could not save feedback", exc)

        prompt = st.chat_input("Ask FurnaceMind...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                run = start_run(
                    conversation_id,
                    {"message": prompt, "allow_llm": False, "tool_mode": "auto"},
                    token=token,
                )
                status = get_run(run["id"], token=token)
                result_message = status.get("result_message")
                with st.chat_message("assistant"):
                    st.markdown((result_message or {}).get("content") or "Run completed.")
                    for event in status.get("events", []):
                        st.caption(event.get("event_type"))
                    for warning in (result_message or {}).get("warnings", []):
                        st.warning(warning.get("message", warning))
                st.rerun()
            except FrontendApiError as exc:
                _show_api_error("FurnaceMind run failed", exc)

    with tab_docs:
        uploaded = st.file_uploader(
            "Upload document",
            type=["txt", "md", "csv", "json", "pdf"],
            key="fm_api_document_upload",
        )
        if uploaded and st.button("Upload", key="fm_api_upload_button"):
            try:
                result = upload_document(uploaded, token=token)
                st.success(f"Uploaded {result.get('filename')}")
                for warning_item in result.get("warnings", []):
                    st.warning(warning_item.get("message", warning_item))
            except FrontendApiError as exc:
                _show_api_error("Document upload failed", exc)
        try:
            docs = list_documents(token=token)
            st.dataframe(docs.get("items", []), width="stretch")
        except FrontendApiError as exc:
            _show_api_error("Could not load documents", exc)

    with tab_tools:
        try:
            tools = list_tools(token=token)
            st.dataframe(tools, width="stretch")
        except FrontendApiError as exc:
            _show_api_error("Could not load tools", exc)


def main() -> None:
    """Render the FurnaceMind page."""
    if is_backend_api_enabled("furnacemind"):
        _render_api_mode()
        return

    from apps.frontend_streamlit.agents.furnacemind.ai_cooperate_page import render_ai_cooperate
    from apps.frontend_streamlit.config.config_loader import load_config
    from apps.frontend_streamlit.ui.furnacemind_sections import select_nav_tab
    from apps.frontend_streamlit.ui.furnacemind.reports import render_reports
    from apps.frontend_streamlit.ui.styles import apply_styles
    from apps.frontend_streamlit.utils.dataset_refresher import (
        get_version as _ds_get_version,
        maybe_refresh as _ds_maybe_refresh,
    )

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    apply_styles()

    st.title("FurnaceMind")
    st.caption("Blast Furnace Operational Intelligence")
    st.caption("FurnaceMind mode: Direct")

    # ── ML dataset auto-refresh ────────────────────────────────────────────
    _config = load_config("setting_ds_dv.yml")
    if _ds_maybe_refresh(_config):
        st.sidebar.caption("⏳ Refreshing dataset in background…")

    _current_ds_version = _ds_get_version()
    if st.session_state.get("_ds_version") != _current_ds_version:
        st.session_state.pop("fm_ml_df_cache", None)
        st.session_state["_ds_version"] = _current_ds_version

    app_mode = select_nav_tab()

    if app_mode == "🤖 AI Co-Operate":
        render_ai_cooperate(field_labels={})
        return

    if app_mode == "📊 Reports":
        render_reports()
        return


if __name__ == "__main__":
    main()
