"""Manage FurnaceMind AI Co-Operate conversation memory.

This module owns the PostgreSQL-backed memory summary flow for FurnaceMind.
It loads the latest saved conversation summary, generates a new rolling summary
with the LLM whenever the chat reaches the configured message window, and saves
the updated summary back to the ``memory_summaries`` table.

The raw chat messages remain stored separately in ``conversation_messages``.
This module stores only the compressed conversation summary that is later
injected into the FurnaceMind system prompt along with the recent chat window.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from furnace_data import relational

DEFAULT_MEMORY: dict[str, Any] = {
    "conversation_summary": "",
    "last_updated_utc": None,
}

MEMORY_SUMMARY_MESSAGE_WINDOW = 8

_MEMORY_SOURCE = "furnacemind_memory"
_SUMMARY_KIND = "conversation_summary"


def _utc_now_iso() -> str:
    """
    Create a consistent timestamp for FurnaceMind memory metadata.

    Memory summaries are stored in PostgreSQL, but the in-memory payload also
    carries a lightweight update marker. UTC keeps this value independent of the
    operator's local timezone and makes it safe to compare across environments.

    Args:
         - None

    Returns:
         - return: str - Current UTC timestamp formatted as ISO-8601 text.
    """
    return datetime.now(timezone.utc).isoformat()


def _estimate_text_tokens(text: str | None) -> int:
    """
    Estimate summary token count for the ``memory_summaries`` table.

    The relational table stores a token count for reporting and future context
    budgeting. This helper uses a small character-based estimate so saving a
    summary does not need an extra tokenizer dependency or model call.

    Args:
         - text: str | None - Summary text whose approximate token count is needed.

    Returns:
         - return: int - Approximate token count, returning zero for blank text.
    """
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _normalize_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    """
    Convert any partial memory payload into the FurnaceMind memory shape.

    Callers may pass ``None``, an empty dictionary, or a dictionary loaded from
    PostgreSQL. This helper guarantees that downstream code can safely read the
    expected keys without repeating defensive checks at every call site.

    Args:
         - memory: dict[str, Any] | None - Raw memory payload from storage or caller code.

    Returns:
         - return: dict[str, Any] - Memory payload with default keys and safe value types.
    """
    normalized = dict(DEFAULT_MEMORY)
    if isinstance(memory, dict):
        for key in DEFAULT_MEMORY:
            if key in memory:
                normalized[key] = memory[key]
    if not isinstance(normalized.get("conversation_summary"), str):
        normalized["conversation_summary"] = ""
    return normalized


def _is_text_chat_message(item: dict[str, Any]) -> bool:
    """
    Decide whether a Streamlit chat-history item belongs in memory compression.

    FurnaceMind chat history can contain text messages as well as rendered
    artifacts such as plots or dataframes. Only non-empty user and assistant
    text messages are safe to send to the memory-summary LLM.

    Args:
         - item: dict[str, Any] - Streamlit chat-history entry.

    Returns:
         - return: bool - True when the item is a text user/assistant message.
    """
    if item.get("type") in {"plotly", "dataframe"}:
        return False
    role = item.get("role")
    content = item.get("content")
    return (
        role in {"user", "assistant"}
        and isinstance(content, str)
        and bool(content.strip())
    )


def _text_chat_messages(chat_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return only the chat messages that can be used for summary generation.

    This keeps the summary window aligned to real conversation turns and removes
    UI artifacts before message counts, source message ids, and LLM input text
    are calculated.

    Args:
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.

    Returns:
         - return: list[dict[str, Any]] - Summarizable chat messages.
    """
    return [item for item in chat_history if _is_text_chat_message(item)]


def _format_summary_window(messages: list[dict[str, Any]]) -> str:
    """
    Format the latest message window into readable input for the summary LLM.

    The summary model receives the previous cumulative summary plus a numbered
    list of recent text messages. Numbering and role labels make the input clear
    without asking the model to parse Streamlit's internal dictionary structure.

    Args:
         - messages: list[dict[str, Any]] - Text chat messages to summarize.

    Returns:
         - return: str - Prompt-ready message window.
    """
    formatted: list[str] = []
    for index, item in enumerate(messages, start=1):
        role = str(item.get("role") or "unknown").upper()
        content = str(item.get("content") or "").strip()
        formatted.append(f"{index}. {role}: {content}")
    return "\n\n".join(formatted)


def should_generate_memory_summary(
    chat_history: list[dict[str, Any]],
    *,
    window: int = MEMORY_SUMMARY_MESSAGE_WINDOW,
) -> bool:
    """
    Check whether the chat has reached the next memory-summary boundary.

    FurnaceMind summarizes on fixed text-message windows so the LLM does not run
    after every turn. A summary is due only when the number of summarizable text
    messages is an exact multiple of the configured window size.

    Args:
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - window: int - Number of text messages per summary window.

    Returns:
         - return: bool - True when an LLM summary should be generated.
    """
    if window <= 0:
        return False
    message_count = len(_text_chat_messages(chat_history))
    return message_count >= window and message_count % window == 0


def summary_source_message_ids(
    chat_history: list[dict[str, Any]],
    *,
    window: int = MEMORY_SUMMARY_MESSAGE_WINDOW,
) -> tuple[str | None, str | None]:
    """
    Find the message id range represented by the latest summary window.

    The memory summary row stores the first and last source message ids so a
    reviewer can trace which chat messages were compressed into that summary.
    If message persistence was unavailable, either id may be missing.

    Args:
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - window: int - Number of text messages per summary window.

    Returns:
         - return: tuple[str | None, str | None] - Start and end message ids.
    """
    messages = _text_chat_messages(chat_history)
    if not messages:
        return None, None
    window_messages = messages[-window:]
    start_id = window_messages[0].get("message_id")
    end_id = window_messages[-1].get("message_id")
    return (
        str(start_id) if start_id else None,
        str(end_id) if end_id else None,
    )


def generate_memory_summary(
    memory: dict[str, Any],
    *,
    chat_history: list[dict[str, Any]],
    llm: Any,
    summary_system_prompt: str,
    summary_token_limit: int,
    window: int = MEMORY_SUMMARY_MESSAGE_WINDOW,
) -> dict[str, Any]:
    """
    Generate the next cumulative memory summary when a window is complete.

    The function keeps the previous saved summary and sends it together with the
    latest text-message window to the memory-compression model. The returned
    summary replaces the prior summary, so each saved row should carry forward
    useful older facts while adding durable new facts from the latest window.

    Args:
         - memory: dict[str, Any] - Current persistent memory payload.
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - llm: Any - LLM client used to create the updated summary.
         - summary_system_prompt: str - System prompt used for summary generation.
         - summary_token_limit: int - Maximum summary length requested from the LLM.
         - window: int - Number of text messages per summary window.

    Returns:
         - return: dict[str, Any] - Memory payload with an updated summary when due.
    """
    normalized = _normalize_memory(memory)
    if not should_generate_memory_summary(chat_history, window=window):
        return normalized

    messages = _text_chat_messages(chat_history)[-window:]
    previous_summary = normalized.get("conversation_summary") or "(none yet)"
    user_prompt = (
        "Previous cumulative memory summary:\n"
        f"{previous_summary}\n\n"
        f"New message window ({len(messages)} messages):\n"
        f"{_format_summary_window(messages)}\n\n"
        "Create the next cumulative memory summary by preserving useful facts "
        "from the previous summary and adding durable facts from the new "
        "message window. Return only the replacement cumulative summary. "
        f"Keep it under {summary_token_limit} tokens."
    )

    try:
        summary_text = llm.generate(
            system_prompt=summary_system_prompt,
            user_prompt=user_prompt,
        ).strip()
    except Exception:
        return normalized

    if summary_text:
        normalized["conversation_summary"] = summary_text
        normalized["last_updated_utc"] = _utc_now_iso()
    return normalized


class FurnaceMindMemoryStore:
    """Database adapter for FurnaceMind conversation summaries.

    This class hides the repository details from the Streamlit page. It loads
    and saves only FurnaceMind conversation-summary rows, identified by the
    metadata source/kind pair used by this module.
    """

    def __init__(self) -> None:
        """
        Create the repository needed for SQL-backed memory summaries.

        The relational engine reads the normal application database settings.
        A session factory is passed into ``MemorySummaryRepository`` so the rest
        of this adapter can load and write summary rows without owning session
        management details.

        Args:
             - None

        Returns:
             - return: None - This function does not return a value.
        """
        engine = relational.build_relational_engine()
        session_factory = relational.build_relational_session_factory(engine)
        self._summaries = relational.MemorySummaryRepository(session_factory)

    def load_memory(self, *, user_id: str, conversation_id: str) -> dict[str, Any]:
        """
        Load the latest FurnaceMind summary for one user conversation.

        The repository returns recent summaries for the conversation. This
        method filters them to the current user and the FurnaceMind summary
        metadata so unrelated summary rows cannot be injected into the prompt.

        Args:
             - user_id: str - User that owns the memory.
             - conversation_id: str - Conversation whose summary should be loaded.

        Returns:
             - return: dict[str, Any] - Normalized memory payload.
        """
        summaries = self._summaries.list_summaries(
            conversation_id=conversation_id,
            limit=20,
        )

        for summary in summaries:
            metadata = (
                summary.metadata_json if isinstance(summary.metadata_json, dict) else {}
            )
            same_user = summary.user_id == user_id
            same_source = metadata.get("source") == _MEMORY_SOURCE
            same_kind = metadata.get("kind") == _SUMMARY_KIND

            if same_user and same_source and same_kind:
                return _normalize_memory(
                    {
                        "conversation_summary": summary.summary_text or "",
                        "last_updated_utc": summary.created_at.isoformat(),
                    }
                )

        return _normalize_memory(None)

    def save_memory(
        self,
        memory: dict[str, Any],
        *,
        user_id: str,
        conversation_id: str,
        source_message_id_start: str | None = None,
        source_message_id_end: str | None = None,
    ) -> None:
        """
        Save a new FurnaceMind summary row when the summary has changed.

        Empty summaries are ignored, and duplicate text is not written again.
        When a new summary is saved, the row is linked to the conversation,
        user, source message range, approximate token count, and FurnaceMind
        metadata used later by ``load_memory``.

        Args:
             - memory: dict[str, Any] - Memory payload containing the summary.
             - user_id: str - User that owns the memory.
             - conversation_id: str - Conversation that owns the summary.
             - source_message_id_start: str | None - First source message id.
             - source_message_id_end: str | None - Last source message id.

        Returns:
             - return: None - This function does not return a value.
        """
        normalized = _normalize_memory(memory)
        existing = self.load_memory(user_id=user_id, conversation_id=conversation_id)

        summary_text = str(normalized.get("conversation_summary") or "").strip()
        if not summary_text:
            return

        if summary_text == str(existing.get("conversation_summary") or "").strip():
            return

        self._summaries.create_summary(
            conversation_id=conversation_id,
            user_id=user_id,
            summary_text=summary_text,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
            token_count=_estimate_text_tokens(summary_text),
            metadata={"source": _MEMORY_SOURCE, "kind": _SUMMARY_KIND},
        )


@lru_cache(maxsize=1)
def _memory_store() -> FurnaceMindMemoryStore:
    """
    Return the cached PostgreSQL adapter for FurnaceMind summaries.

    The page can call load/save helpers multiple times during Streamlit reruns.
    Caching the adapter avoids rebuilding the relational engine and repository
    for every call while still using repository-managed sessions per operation.

    Args:
         - None

    Returns:
         - return: FurnaceMindMemoryStore - Cached DB-backed memory adapter.
    """
    return FurnaceMindMemoryStore()


def load_fm_memory(
    *,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """
    Load normalized FurnaceMind memory for prompt injection.

    This is the page-facing read helper. It returns an empty default memory
    object when the user or conversation is not known, or when PostgreSQL is not
    available, so the chat page can continue without breaking the UI.

    Args:
         - user_id: str | None - User that owns the memory.
         - conversation_id: str | None - Conversation whose summary should be loaded.

    Returns:
         - return: dict[str, Any] - Normalized memory payload.
    """
    if not user_id or not conversation_id:
        return _normalize_memory(None)

    try:
        return _memory_store().load_memory(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except Exception:
        return _normalize_memory(None)


def save_fm_memory(
    memory: dict[str, Any],
    *,
    user_id: str | None = None,
    conversation_id: str | None = None,
    source_message_id_start: str | None = None,
    source_message_id_end: str | None = None,
) -> None:
    """
    Persist the generated FurnaceMind memory summary if it is safe to save.

    This is the page-facing write helper. It requires both user and conversation
    ids, refreshes the payload timestamp, and delegates duplicate detection plus
    row creation to ``FurnaceMindMemoryStore``.

    Args:
         - memory: dict[str, Any] - Memory payload containing the summary.
         - user_id: str | None - User that owns the memory.
         - conversation_id: str | None - Conversation that owns the summary.
         - source_message_id_start: str | None - First source message id.
         - source_message_id_end: str | None - Last source message id.

    Returns:
         - return: None - This function does not return a value.
    """
    if not user_id or not conversation_id:
        return
    try:
        normalized = _normalize_memory(memory)
        normalized["last_updated_utc"] = _utc_now_iso()
        _memory_store().save_memory(
            normalized,
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
        )
    except Exception:
        return


def build_persistent_context(memory: dict[str, Any]) -> str:
    """
    Convert stored FurnaceMind memory into a system-prompt context block.

    The agent should receive compressed durable memory, not the raw SQL row.
    This helper extracts the saved conversation summary and labels it clearly so
    the LLM can treat it as prior context instead of a new user instruction.

    Args:
         - memory: dict[str, Any] - Persistent FurnaceMind memory payload.

    Returns:
         - return: str - Prompt-ready memory context block.
    """
    if not memory:
        return ""

    parts: list[str] = []

    summary = (memory.get("conversation_summary") or "").strip()
    if summary:
        parts.append("PERSISTENT CONVERSATION SUMMARY (compressed):\n" + summary)

    return "\n\n".join(parts).strip()
