"""Backend-only optional V-Sense review policy helpers."""

from __future__ import annotations

from typing import Any


PROMPT_VERSION = "vsense-review-v1"


def unavailable_review(reason: str = "LLM review is disabled by backend policy.") -> dict[str, Any]:
    """Return a structured unavailable review without affecting optimization."""

    return {
        "available": False,
        "prompt_version": PROMPT_VERSION,
        "markdown": None,
        "warnings": [reason],
        "latency_ms": None,
    }


def sanitize_markdown(markdown: str, *, max_chars: int = 4000) -> str:
    """Return bounded Markdown safe for advisory display."""

    clean = str(markdown or "").replace("\x00", "").strip()
    return clean[: max(1, int(max_chars))]
