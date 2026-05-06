"""Session-state helpers for the FurnaceMind page."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import streamlit as st

_IST = timezone(timedelta(hours=5, minutes=30))
_ARTIFACT_TYPES = {"plotly", "dataframe"}


def current_user_id() -> str:
    """
    Return the current user id for persisted FurnaceMind records.

    Args:
         - None

    Returns:
         - return: str - Authenticated user id or anonymous fallback.
    """
    user_id = str(st.session_state.get("auth_user") or "anonymous").strip()
    return user_id or "anonymous"


def last_completed_shift() -> tuple[date, str]:
    """
    Return the most recently completed IST shift.

    Args:
         - None

    Returns:
         - return: tuple[date, str] - Shift date and shift label.
    """
    now = datetime.now(_IST)
    if now.hour < 6:
        return (now.date() - timedelta(days=1)), "C"
    if now.hour < 14:
        return now.date(), "A"
    if now.hour < 22:
        return now.date(), "B"
    return now.date(), "C"


def chat_history_to_messages(max_messages: int = 14) -> list[dict]:
    """
    Convert Streamlit chat history into LLM message objects.

    Args:
         - max_messages: int - Maximum recent text messages to include.

    Returns:
         - return: list[dict] - OpenAI-compatible chat messages.
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
