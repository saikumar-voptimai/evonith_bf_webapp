"""Manage FurnaceMind AI Co-Operate conversation memory.

This module owns the PostgreSQL-backed memory summary flow for FurnaceMind.
It loads the latest saved conversation summary, generates a new rolling summary
with the LLM whenever the chat reaches the configured message window, supports
an explicit final flush for short unsummarized tails, and saves the updated
summary back to the ``memory_summaries`` table.

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
    "source_message_id_start": None,
    "source_message_id_end": None,
    "summarized_message_count": 0,
}

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
    if normalized.get("source_message_id_start") is not None:
        normalized["source_message_id_start"] = str(
            normalized["source_message_id_start"]
        )
    if normalized.get("source_message_id_end") is not None:
        normalized["source_message_id_end"] = str(normalized["source_message_id_end"])
    try:
        normalized["summarized_message_count"] = int(
            normalized.get("summarized_message_count") or 0
        )
    except (TypeError, ValueError):
        normalized["summarized_message_count"] = 0
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


def _message_id(item: dict[str, Any]) -> str | None:
    """
    Return a normalized chat message id, when one is available.

    Args:
         - item: dict[str, Any] - Streamlit chat-history entry.

    Returns:
         - return: str | None - Stable message id or None.
    """
    message_id = item.get("message_id")
    if not message_id:
        return None
    return str(message_id)


def _last_summarized_index(
    messages: list[dict[str, Any]],
    memory: dict[str, Any],
) -> int:
    """
    Find the index of the last message covered by the saved cumulative summary.

    Args:
         - messages: list[dict[str, Any]] - Summarizable chat messages.
         - memory: dict[str, Any] - Normalized persistent memory payload.

    Returns:
         - return: int - Zero-based index, or -1 when no summarized message is known.
    """
    source_message_id_end = memory.get("source_message_id_end")
    if source_message_id_end:
        for index, item in enumerate(messages):
            if _message_id(item) == source_message_id_end:
                return index
    return -1


def _pending_summary_messages(
    chat_history: list[dict[str, Any]],
    *,
    memory: dict[str, Any] | None,
    window: int,
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Select the next unsummarized message window.

    Normal operation returns a full unsummarized window. When ``force`` is true,
    a shorter tail is also returned so callers can finalize a conversation
    without leaving the last few messages outside the cumulative summary.

    Args:
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - memory: dict[str, Any] | None - Current persistent memory payload.
         - window: int - Number of text messages per regular summary window.
         - force: bool - Whether to summarize a partial trailing window.

    Returns:
         - return: list[dict[str, Any]] - Messages to send to the summary LLM.
    """
    if window <= 0:
        return []

    messages = _text_chat_messages(chat_history)
    normalized = _normalize_memory(memory)
    pending = messages[_last_summarized_index(messages, normalized) + 1 :]

    if len(pending) >= window:
        return pending[:window]
    if force and pending:
        return pending
    return []


def _summary_message_count(
    chat_history: list[dict[str, Any]],
    selected_messages: list[dict[str, Any]],
) -> int:
    """
    Count text messages through the final selected summary message.

    Args:
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - selected_messages: list[dict[str, Any]] - Messages chosen for summary.

    Returns:
         - return: int - Count of text messages covered by the cumulative summary.
    """
    if not selected_messages:
        return 0

    messages = _text_chat_messages(chat_history)
    final_message = selected_messages[-1]
    final_message_id = _message_id(final_message)
    for index, item in enumerate(messages, start=1):
        if item is final_message:
            return index
        if final_message_id and _message_id(item) == final_message_id:
            return index
    return 0


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
    window: int,
    memory: dict[str, Any] | None = None,
    force: bool = False,
) -> bool:
    """
    Check whether the chat has reached the next memory-summary boundary.

    FurnaceMind summarizes on fixed text-message windows so the LLM does not run
    after every turn. A summary is due when there is a full unsummarized message
    window. When ``force`` is true, a shorter unsummarized tail is also due.

    Args:
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - window: int - Number of text messages per summary window.
         - memory: dict[str, Any] | None - Current persistent memory payload.
         - force: bool - Whether to treat a partial trailing window as due.

    Returns:
         - return: bool - True when an LLM summary should be generated.
    """
    return bool(
        _pending_summary_messages(
            chat_history,
            memory=memory,
            window=window,
            force=force,
        )
    )


def summary_source_message_ids(
    chat_history: list[dict[str, Any]],
    *,
    window: int,
    memory: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[str | None, str | None]:
    """
    Find the message id range represented by the next summary window.

    The memory summary row stores the first and last source message ids so a
    reviewer can trace which chat messages were compressed into that summary.
    If message persistence was unavailable, either id may be missing.

    Args:
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - window: int - Number of text messages per summary window.
         - memory: dict[str, Any] | None - Current persistent memory payload.
         - force: bool - Whether to use a partial trailing window.

    Returns:
         - return: tuple[str | None, str | None] - Start and end message ids.
    """
    messages = _pending_summary_messages(
        chat_history,
        memory=memory,
        window=window,
        force=force,
    )
    if not messages:
        return None, None
    start_id = messages[0].get("message_id")
    end_id = messages[-1].get("message_id")
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
    window: int,
    force: bool = False,
) -> dict[str, Any]:
    """
    Generate the next cumulative memory summary when a window is due.

    The function keeps the previous saved summary and sends it together with the
    next unsummarized text-message window to the memory-compression model. The
    returned summary replaces the prior summary, so each saved row should carry
    forward useful older facts while adding durable new facts from that window.

    Args:
         - memory: dict[str, Any] - Current persistent memory payload.
         - chat_history: list[dict[str, Any]] - Raw Streamlit chat-history items.
         - llm: Any - LLM client used to create the updated summary.
         - summary_system_prompt: str - System prompt used for summary generation.
         - summary_token_limit: int - Maximum summary length requested from the LLM.
         - window: int - Number of text messages per summary window.
         - force: bool - Whether to summarize a partial trailing window.

    Returns:
         - return: dict[str, Any] - Memory payload with an updated summary when due.
    """
    normalized = _normalize_memory(memory)
    messages = _pending_summary_messages(
        chat_history,
        memory=normalized,
        window=window,
        force=force,
    )
    if not messages:
        return normalized

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
        normalized["source_message_id_start"] = _message_id(messages[0])
        normalized["source_message_id_end"] = _message_id(messages[-1])
        normalized["summarized_message_count"] = _summary_message_count(
            chat_history,
            messages,
        )
    return normalized


def add_recent_turn(
    memory: dict[str, Any], *, user: str, assistant: str
) -> dict[str, Any]:
    """
    Append one recent turn to the memory payload for legacy UI callers.

    The active AI Co-Operate page uses ``generate_memory_summary`` so summaries
    are produced by the configured memory-compression LLM. This helper remains
    as a small compatibility path for older FurnaceMind page code that imports
    it directly.

    Args:
         - memory: dict[str, Any] - Existing memory payload.
         - user: str - Latest user message.
         - assistant: str - Latest assistant response.

    Returns:
         - return: dict[str, Any] - Memory payload with a lightweight recent-turn note.
    """
    normalized = _normalize_memory(memory)
    user_text = str(user or "").strip()
    assistant_text = str(assistant or "").strip()
    if not user_text and not assistant_text:
        return normalized

    existing_summary = str(normalized.get("conversation_summary") or "").strip()
    turn_note = (
        "Latest user request: "
        f"{user_text}\n"
        "Latest assistant response: "
        f"{assistant_text}"
    ).strip()
    normalized["conversation_summary"] = "\n\n".join(
        part for part in (existing_summary, turn_note) if part
    )
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
            same_user = str(summary.user_id) == user_id
            same_source = metadata.get("source") == _MEMORY_SOURCE
            same_kind = metadata.get("kind") == _SUMMARY_KIND

            if same_user and same_source and same_kind:
                return _normalize_memory(
                    {
                        "conversation_summary": summary.summary_text or "",
                        "last_updated_utc": summary.created_at.isoformat(),
                        "source_message_id_start": summary.source_message_id_start,
                        "source_message_id_end": summary.source_message_id_end,
                        "summarized_message_count": metadata.get(
                            "summarized_message_count",
                            0,
                        ),
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
    ) -> bool:
        """
        Save a new FurnaceMind summary row when the summary has changed.

        Empty summaries are ignored, and exact duplicate coverage is not written
        again.
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
             - return: bool - True when a new summary row was inserted.
        """
        normalized = _normalize_memory(memory)
        existing = self.load_memory(user_id=user_id, conversation_id=conversation_id)

        summary_text = str(normalized.get("conversation_summary") or "").strip()
        if not summary_text:
            return False

        existing_summary_text = str(existing.get("conversation_summary") or "").strip()
        existing_source_end = str(existing.get("source_message_id_end") or "").strip()
        new_source_end = str(
            source_message_id_end or normalized.get("source_message_id_end") or ""
        ).strip()
        if (
            summary_text == existing_summary_text
            and new_source_end == existing_source_end
        ):
            return False
        metadata = {"source": _MEMORY_SOURCE, "kind": _SUMMARY_KIND}
        summarized_message_count = normalized.get("summarized_message_count")
        if isinstance(summarized_message_count, int) and summarized_message_count > 0:
            metadata["summarized_message_count"] = summarized_message_count

        self._summaries.create_summary(
            conversation_id=conversation_id,
            user_id=user_id,
            summary_text=summary_text,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
            token_count=_estimate_text_tokens(summary_text),
            metadata=metadata,
        )
        return True


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
) -> bool:
    """
    Persist the generated FurnaceMind memory summary if it is safe to save.

    This is the page-facing write helper. It requires both user and conversation
    ids, refreshes the payload timestamp, and delegates duplicate detection plus
    row creation to ``FurnaceMindMemoryStore``. Failures return ``False`` so the
    page can avoid advancing workflows that depend on the summary row.

    Args:
         - memory: dict[str, Any] - Memory payload containing the summary.
         - user_id: str | None - User that owns the memory.
         - conversation_id: str | None - Conversation that owns the summary.
         - source_message_id_start: str | None - First source message id.
         - source_message_id_end: str | None - Last source message id.

    Returns:
         - return: bool - True when a new summary row was inserted.
    """
    if not user_id or not conversation_id:
        return False
    try:
        normalized = _normalize_memory(memory)
        source_message_id_start = source_message_id_start or normalized.get(
            "source_message_id_start"
        )
        source_message_id_end = source_message_id_end or normalized.get(
            "source_message_id_end"
        )
        normalized["last_updated_utc"] = _utc_now_iso()
        return _memory_store().save_memory(
            normalized,
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
        )
    except Exception:
        return False


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
