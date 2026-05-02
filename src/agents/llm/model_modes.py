"""Reasoning effort helpers for FurnaceMind OpenRouter calls."""

from __future__ import annotations

import os

DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORT_OPTIONS = ("low", "medium", "high")
REASONING_EFFORT_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}
_LEGACY_MAP = {"fast": "low", "reasoning": "medium"}


def normalize_reasoning_effort(value: str | None) -> str:
    """Return a supported reasoning effort."""
    effort = (value or "").strip().lower()
    effort = _LEGACY_MAP.get(effort, effort)
    if effort in REASONING_EFFORT_OPTIONS:
        return effort
    return DEFAULT_REASONING_EFFORT


def reasoning_effort_label(effort: str) -> str:
    """Return the human-readable label for a reasoning effort."""
    return REASONING_EFFORT_LABELS[normalize_reasoning_effort(effort)]


def configured_model_name() -> str | None:
    """Return the single configured OpenRouter model."""
    return os.getenv("OPENROUTER_MODEL")


def configured_default_reasoning_effort() -> str:
    """Return the default reasoning effort from environment or medium."""
    return normalize_reasoning_effort(
        os.getenv("FURNACEMIND_REASONING_EFFORT")
        or os.getenv("OPENROUTER_REASONING_EFFORT")
    )
