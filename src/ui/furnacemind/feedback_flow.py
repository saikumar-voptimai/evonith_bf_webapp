"""Streamlit feedback event flow for FurnaceMind AI Co-Operate.

This module keeps feedback UI events out of the main Streamlit page renderer.
It consumes explicit thumbs feedback queued by the chat UI, detects feedback
written directly in chat, and delegates persistence/lesson generation to the
feedback service in ``utils.furnacemind.feedback_service``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from utils.furnacemind.feedback_service import (
    FurnaceMindFeedbackService,
    latest_assistant_exchange,
)


def process_pending_explicit_feedback(
    *,
    feedback_service: FurnaceMindFeedbackService | None,
    feedback_llm: Any | None,
    user_id: str,
) -> None:
    """
    Persist thumbs feedback queued by the chat UI.

    The chat UI stores the selected polarity and user-written feedback text in
    Streamlit session state. This function consumes that event, saves it through
    the feedback service, and marks the assistant message as handled for the
    current session.

    Args:
         - feedback_service: FurnaceMindFeedbackService | None - Feedback service.
         - feedback_llm: Any | None - LLM used to generate feedback lessons.
         - user_id: str - Current user id.

    Returns:
         - return: None - This function does not return a value.
    """
    event = st.session_state.pop("pending_fm_feedback", None)
    if not event:
        return
    if feedback_service is None:
        st.sidebar.caption("Feedback service unavailable.")
        return
    if feedback_llm is None:
        st.sidebar.caption("Feedback LLM unavailable.")
        return

    try:
        feedback_id = feedback_service.save_explicit_feedback(
            user_id=user_id,
            conversation_id=str(event.get("conversation_id") or ""),
            message_id=str(event.get("message_id") or ""),
            raw_user_message=str(event.get("raw_user_message") or ""),
            assistant_response=str(event.get("assistant_response") or ""),
            polarity=str(event.get("polarity") or "negative"),
            feedback_text=str(event.get("feedback_text") or ""),
            llm=feedback_llm,
        )
    except Exception as exc:
        st.sidebar.caption(f"Could not save feedback: {exc}")
        return

    if feedback_id:
        saved = st.session_state.setdefault("fm_feedback_saved_message_ids", set())
        saved.add(str(event.get("message_id") or ""))


def detect_and_save_chat_feedback(
    *,
    feedback_service: FurnaceMindFeedbackService | None,
    feedback_llm: Any | None,
    user_id: str,
    conversation_id: str | None,
    user_query: str,
) -> None:
    """
    Detect and persist correction feedback written as a normal chat message.

    Each new user message is compared with the latest assistant response. When
    the detector classifies the message as a correction or evaluation, the
    original question, assistant answer, and correction text are saved as a
    feedback item and converted into a reusable lesson.

    Args:
         - feedback_service: FurnaceMindFeedbackService | None - Feedback service.
         - feedback_llm: Any | None - LLM used for detection and lesson generation.
         - user_id: str - Current user id.
         - conversation_id: str | None - Active conversation id.
         - user_query: str - Latest user message.

    Returns:
         - return: None - This function does not return a value.
    """
    if feedback_service is None or feedback_llm is None or not conversation_id:
        return

    previous_user, previous_assistant = latest_assistant_exchange(
        st.session_state.chat_history
    )
    if previous_user is None or previous_assistant is None:
        return

    assistant_message_id = str(previous_assistant.get("message_id") or "")
    if not assistant_message_id:
        return

    detection = feedback_service.detect_chat_feedback(
        user_message=user_query,
        raw_user_message=str(previous_user.get("content") or ""),
        assistant_response=str(previous_assistant.get("content") or ""),
        llm=feedback_llm,
    )
    if detection is None:
        return

    try:
        feedback_service.save_chat_feedback(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            raw_user_message=str(previous_user.get("content") or ""),
            assistant_response=str(previous_assistant.get("content") or ""),
            feedback_text=user_query,
            polarity=str(detection.get("polarity") or "negative"),
            llm=feedback_llm,
        )
    except Exception as exc:
        st.sidebar.caption(f"Could not save chat feedback: {exc}")
