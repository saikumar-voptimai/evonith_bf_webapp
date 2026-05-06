"""Chat rendering helpers for the FurnaceMind page."""

from __future__ import annotations

from datetime import date

import streamlit as st

from agents.furnacemind.skills import SkillEngine


def render_message(item: dict) -> None:
    """
    Render one FurnaceMind chat message or inline artifact.

    Args:
         - item: dict - Chat history item to render.

    Returns:
         - return: None - Renders directly into the Streamlit page.
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
    Append inline artifact entries for outputs created during the current turn.

    Args:
         - history_len_before: int - Chat history length before agent execution.

    Returns:
         - return: None - Mutates Streamlit session chat history.
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


def queue_skill(prompt: str, display: str, skill_id: str) -> None:
    """
    Queue a quick-skill prompt for execution on the next rerun.

    Args:
         - prompt: str - Prompt sent to the agent.
         - display: str - User-facing prompt label shown in chat.
         - skill_id: str - Skill identifier used by prompt context.

    Returns:
         - return: None - Updates session state and triggers rerun.
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
    Render quick-skill buttons near the FurnaceMind chat input.

    Args:
         - engine: SkillEngine - Skill prompt builder.
         - shift_date: date - Last completed shift date.
         - shift_label: str - Last completed shift label.

    Returns:
         - return: None - Renders controls directly into the Streamlit page.
    """
    st.caption("Quick skills")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Unit Cost", use_container_width=True, key="fm_skill_cost"):
            queue_skill(
                engine.optimise_prompt(),
                "Unit Cost - last 30 days vs best-shift targets",
                "optimise",
            )
    with c2:
        if st.button("Shift to Best", use_container_width=True, key="fm_skill_shift"):
            queue_skill(
                engine.shift_to_best_prompt(str(shift_date), shift_label),
                f"Shift to Best: {shift_date}, Shift {shift_label}",
                "shift_to_best",
            )
    with c3:
        if st.button("Heatloads", use_container_width=True, key="fm_skill_heatloads"):
            queue_skill(
                engine.heatload_prompt(),
                "Heatloads - last 8h vs 2-month baseline",
                "heatload",
            )
