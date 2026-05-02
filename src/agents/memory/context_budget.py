"""Token estimation and context-budget helpers for FurnaceMind."""

from __future__ import annotations

import os
from typing import Any

try:
    import tiktoken
except Exception:  # pragma: no cover - optional runtime fallback
    tiktoken = None


def estimate_text_tokens(text: str | None) -> int:
    """Estimate token count for one text value."""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def estimate_chat_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count for chat messages."""
    total = 0
    for message in messages:
        if message.get("type") == "plotly":
            continue
        total += 4
        total += estimate_text_tokens(str(message.get("content") or ""))
    return total


def configured_context_budget() -> int:
    """Return configured FurnaceMind context budget tokens."""
    return int(os.getenv("FURNACEMIND_CONTEXT_BUDGET_TOKENS", "12000"))


def configured_auto_compress_threshold() -> int:
    """Return configured auto-compression threshold percentage."""
    return int(os.getenv("FURNACEMIND_AUTO_COMPRESS_THRESHOLD_PERCENT", "90"))


def build_context_budget(
    *,
    chat_messages: list[dict[str, Any]],
    base_tokens: int = 0,
    memory_tokens: int = 0,
    feedback_tokens: int = 0,
    docs_tokens: int = 0,
    tools_tokens: int = 0,
    budget_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a display-ready token budget dictionary."""
    budget = budget_tokens or configured_context_budget()
    chat_tokens = estimate_chat_tokens(chat_messages)
    total = (
        base_tokens
        + chat_tokens
        + memory_tokens
        + feedback_tokens
        + docs_tokens
        + tools_tokens
    )
    percent = min(100, int((total / max(1, budget)) * 100))
    return {
        "total_tokens": total,
        "budget_tokens": budget,
        "percent": percent,
        "base_tokens": base_tokens,
        "chat_tokens": chat_tokens,
        "memory_tokens": memory_tokens,
        "feedback_tokens": feedback_tokens,
        "docs_tokens": docs_tokens,
        "tools_tokens": tools_tokens,
        "status": context_budget_status(percent),
    }


def context_budget_status(percent: int) -> str:
    """Return a compact health label for context usage percentage."""
    if percent >= configured_auto_compress_threshold():
        return "Compressing"
    if percent >= 75:
        return "Watch"
    return "Healthy"
