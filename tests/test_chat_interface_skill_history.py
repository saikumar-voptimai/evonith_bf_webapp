from __future__ import annotations

import pandas as pd
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


def test_inject_artifacts_skips_stale_artifacts_from_previous_turn() -> None:
    """A later text-only answer should not show old chart/table artifacts."""
    st.session_state.clear()
    st.session_state["chat_history"] = [
        {"role": "user", "type": "text", "content": "summarize BMO"}
    ]
    st.session_state["fm_artifact_store"] = {}
    st.session_state["fm_fig"] = object()
    st.session_state["fm_fig_turn_id"] = "heatload_turn"
    st.session_state["fm_df"] = pd.DataFrame({"heatload": [1.2]})
    st.session_state["fm_df_turn_id"] = "heatload_turn"

    chat_interface.inject_artifacts(1, artifact_turn_id="bmo_turn")

    assert st.session_state["chat_history"] == [
        {"role": "user", "type": "text", "content": "summarize BMO"}
    ]
    assert st.session_state["fm_artifact_store"] == {}
    st.session_state.clear()


def test_inject_artifacts_adds_artifacts_from_current_turn() -> None:
    """New chart/table artifacts should render when tagged with this turn id."""
    st.session_state.clear()
    figure = object()
    dataframe = pd.DataFrame({"eta_co": [42.1]})
    st.session_state["chat_history"] = []
    st.session_state["fm_artifact_store"] = {}
    st.session_state["fm_fig"] = figure
    st.session_state["fm_fig_turn_id"] = "eta_turn"
    st.session_state["fm_df"] = dataframe
    st.session_state["fm_df_turn_id"] = "eta_turn"
    st.session_state["fm_df_meta"] = {"dataset_id": "eta_recent"}

    chat_interface.inject_artifacts(0, artifact_turn_id="eta_turn")

    assert st.session_state["chat_history"] == [
        {
            "role": "assistant",
            "type": "plotly",
            "artifact_key": "fm_fig_0",
            "content": "",
        },
        {
            "role": "assistant",
            "type": "dataframe",
            "artifact_key": "fm_df_0",
            "content": "",
            "meta": {"dataset_id": "eta_recent"},
        },
    ]
    assert st.session_state["fm_artifact_store"] == {
        "fm_fig_0": figure,
        "fm_df_0": dataframe,
    }
    st.session_state.clear()
