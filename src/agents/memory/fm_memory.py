"""PostgreSQL-backed conversation summaries for FurnaceMind AI Co-Operate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from furnace_data import relational

DEFAULT_MEMORY: dict[str, Any] = {
    "conversation_summary": "",
    "recent_turns": [],
    "last_updated_utc": None,
}

_MEMORY_SOURCE = "furnacemind_memory"
_SUMMARY_KIND = "conversation_summary"


def _utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO-8601 string.

    Args:
         - None

    Returns:
         - return: str - Current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def get_default_memory_path() -> Path:
    """
    Return the legacy FurnaceMind memory JSON path.

    Args:
         - None

    Returns:
         - return: Path - Legacy JSON file path used only for one-time migration.
    """
    return Path("src/storage/furnacemind/ai_cooperate_memory.json")


def _estimate_text_tokens(text: str | None) -> int:
    """
    Estimate token count for memory records.

    Args:
         - text: str | None - Text content to estimate.

    Returns:
         - return: int - Approximate token count.
    """
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _normalize_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalize a partial memory payload into the expected structure.

    Args:
         - memory: dict[str, Any] | None - Raw memory payload.

    Returns:
         - return: dict[str, Any] - Memory payload with all default keys present.
    """
    normalized = dict(DEFAULT_MEMORY)
    if isinstance(memory, dict):
        for key in DEFAULT_MEMORY:
            if key in memory:
                normalized[key] = memory[key]
    if not isinstance(normalized.get("recent_turns"), list):
        normalized["recent_turns"] = []
    if not isinstance(normalized.get("conversation_summary"), str):
        normalized["conversation_summary"] = ""
    return normalized


def _load_legacy_memory(path: Path | None = None) -> dict[str, Any]:
    """
    Load the legacy JSON memory file for migration fallback.

    Args:
         - path: Path | None - Optional legacy JSON path override.

    Returns:
         - return: dict[str, Any] - Normalized memory payload.
    """
    p = path or get_default_memory_path()
    try:
        if not p.exists():
            return _normalize_memory(None)
        data = json.loads(p.read_text(encoding="utf-8"))
        return _normalize_memory(data if isinstance(data, dict) else None)
    except Exception:
        return _normalize_memory(None)


def _has_persistent_content(memory: dict[str, Any]) -> bool:
    """
    Return whether a memory payload has SQL-backed persistent content.

    Args:
         - memory: dict[str, Any] - Normalized memory payload.

    Returns:
         - return: bool - True when a conversation summary exists.
    """
    return bool(str(memory.get("conversation_summary") or "").strip())


class FurnaceMindMemoryStore:
    """Database adapter for FurnaceMind conversation summaries."""

    def __init__(self) -> None:
        """
        Create memory repositories from the configured relational database.

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
        Load FurnaceMind memory from PostgreSQL.

        Args:
             - user_id: str - User that owns the memory.
             - conversation_id: str - Conversation whose summary should be loaded.

        Returns:
             - return: dict[str, Any] - Normalized memory payload.
        """
        memory = _normalize_memory(None)
        self._load_summary(memory, conversation_id=conversation_id)
        return memory

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
        Persist FurnaceMind conversation summary to PostgreSQL.

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
        self._save_summary(
            normalized,
            existing=existing,
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
        )

    def _load_summary(
        self,
        memory: dict[str, Any],
        *,
        conversation_id: str,
    ) -> None:
        """
        Load the latest FurnaceMind conversation summary into memory.

        Args:
             - memory: dict[str, Any] - Memory payload to mutate.
             - conversation_id: str - Conversation id to inspect.

        Returns:
             - return: None - This function does not return a value.
        """
        summaries = self._summaries.list_summaries(
            conversation_id=conversation_id,
            limit=20,
        )
        for summary in summaries:
            metadata = (
                summary.metadata_json if isinstance(summary.metadata_json, dict) else {}
            )
            if metadata.get("source") != _MEMORY_SOURCE:
                continue
            if metadata.get("kind") != _SUMMARY_KIND:
                continue
            memory["conversation_summary"] = summary.summary_text or ""
            memory["last_updated_utc"] = summary.created_at.isoformat()
            return

    def _save_summary(
        self,
        memory: dict[str, Any],
        *,
        existing: dict[str, Any],
        user_id: str,
        conversation_id: str,
        source_message_id_start: str | None,
        source_message_id_end: str | None,
    ) -> None:
        """
        Store a new conversation summary when it changed.

        Args:
             - memory: dict[str, Any] - Incoming normalized memory.
             - existing: dict[str, Any] - Existing normalized memory from DB.
             - user_id: str - User that owns the summary.
             - conversation_id: str - Conversation that owns the summary.
             - source_message_id_start: str | None - First source message id.
             - source_message_id_end: str | None - Last source message id.

        Returns:
             - return: None - This function does not return a value.
        """
        summary_text = str(memory.get("conversation_summary") or "").strip()
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
    Return a cached PostgreSQL memory store.

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
    legacy_path: Path | None = None,
) -> dict[str, Any]:
    """
    Load FurnaceMind conversation summary memory from PostgreSQL.

    Args:
         - user_id: str | None - User that owns the memory.
         - conversation_id: str | None - Conversation whose summary should be loaded.
         - legacy_path: Path | None - Optional legacy JSON path for migration fallback.

    Returns:
         - return: dict[str, Any] - Normalized memory payload.
    """
    if not user_id or not conversation_id:
        return _normalize_memory(None)

    try:
        memory = _memory_store().load_memory(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if _has_persistent_content(memory):
            return memory

        legacy_memory = _load_legacy_memory(legacy_path)
        if _has_persistent_content(legacy_memory):
            _memory_store().save_memory(
                legacy_memory,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            return legacy_memory
        return memory
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
    Persist FurnaceMind conversation summary memory to PostgreSQL.

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


def add_recent_turn(
    memory: dict[str, Any],
    *,
    user: str,
    assistant: str,
    max_turns: int = 8,
) -> dict[str, Any]:
    """
    Append a user/assistant turn to the in-memory rolling turn buffer.

    Recent chat turns are persisted durably by ``conversation_messages`` from
    ticket #111.  This helper keeps the existing memory shape for prompt-context
    callers without writing duplicate turn rows into the legacy JSON file.

    Args:
         - memory: dict[str, Any] - Current memory dict.
         - user: str - User message text.
         - assistant: str - Assistant response text.
         - max_turns: int - Maximum number of turns to retain.

    Returns:
         - return: dict[str, Any] - New memory dict with the turn appended.
    """
    memory = dict(memory or {})
    turns: list[dict[str, Any]] = list(memory.get("recent_turns") or [])
    turns.append(
        {
            "ts_utc": _utc_now_iso(),
            "user": (user or "").strip(),
            "assistant": (assistant or "").strip(),
        }
    )
    memory["recent_turns"] = turns[-max_turns:]
    return memory


def build_persistent_context(memory: dict[str, Any]) -> str:
    """
    Create a compact text block to inject into the system prompt.

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
