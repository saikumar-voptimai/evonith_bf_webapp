from __future__ import annotations

import streamlit as st

from ui.furnacemind import chat_interface


def test_chat_history_to_messages_skips_inactive_skill_metadata() -> None:
    """Old skill-backed answers should not feed prompts after deactivation."""
    st.session_state.clear()
    st.session_state["fm_inactive_skill_keys"] = {"skill_cast", "cast_house_sop"}
    st.session_state["fm_inactive_skill_context_files"] = {
        "cast_house_sop_furnacemind.md"
    }
    st.session_state["chat_history"] = [
        {
            "role": "user",
            "type": "text",
            "content": "normal question",
            "metadata": {},
        },
        {
            "role": "assistant",
            "type": "text",
            "content": "normal answer",
            "metadata": {},
        },
        {
            "role": "assistant",
            "type": "text",
            "content": "Answer from Cast House SOP",
            "metadata": {
                "skill_context_skill_ids": ["skill_cast"],
                "skill_context_skill_slugs": ["cast_house_sop"],
            },
        },
        {
            "role": "assistant",
            "type": "text",
            "content": "Evidence: cast_house_sop_furnacemind.md",
            "metadata": {},
        },
    ]

    messages = chat_interface.chat_history_to_messages(max_messages=10)

    assert messages == [
        {"role": "user", "content": "normal question"},
        {"role": "assistant", "content": "normal answer"},
    ]
    st.session_state.clear()
