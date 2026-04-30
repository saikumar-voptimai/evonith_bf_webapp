"""Persistent memory for the AI Co-Operate chatbot (FurnaceMind).

Stores a rolling conversation summary, do-not-repeat rules, and operator
preferences in a JSON file.  All writes are non-destructive: the file is
created on first access and errors during read/write are silently swallowed
to prevent UI disruption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MEMORY: Dict[str, Any] = {
    "conversation_summary": "",
    "do_not_repeat": [],
    "preferences": [],
    "recent_turns": [],
    "last_updated_utc": None,
}


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def get_default_memory_path() -> Path:
    """Return the default path for the FurnaceMind memory JSON file.

    The file lives inside the repository so it works on both Streamlit Cloud
    and local deployments.
    """
    # Keep it inside the repo so Streamlit Cloud/local both work.
    return Path("src/storage/furnacemind/ai_cooperate_memory.json")


def load_fm_memory(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the copilot memory dict from *path*, or return safe defaults.

    Merges the persisted data with :data:`DEFAULT_MEMORY` so that newly added
    keys are always present even in older files.  Returns defaults on any read
    or parse error.

    Args:
        path: Override the default memory file path.

    Returns:
        A fully-populated memory dict guaranteed to contain all
        :data:`DEFAULT_MEMORY` keys.
    """
    p = path or get_default_memory_path()
    try:
        if not p.exists():
            return dict(DEFAULT_MEMORY)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(DEFAULT_MEMORY)
        merged = dict(DEFAULT_MEMORY)
        merged.update(data)
        # Normalize list fields
        for k in ("do_not_repeat", "preferences", "recent_turns"):
            if not isinstance(merged.get(k), list):
                merged[k] = []
        if not isinstance(merged.get("conversation_summary"), str):
            merged["conversation_summary"] = ""
        return merged
    except Exception:
        return dict(DEFAULT_MEMORY)


def save_fm_memory(memory: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Persist the copilot *memory* dict to disk.

    Creates parent directories as needed.  Errors are silently swallowed to
    avoid disrupting the UI.

    Args:
        memory: Memory dict to serialise.
        path:   Override the default memory file path.
    """
    p = path or get_default_memory_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        memory = dict(memory or {})
        memory["last_updated_utc"] = _utc_now_iso()
        p.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def add_recent_turn(
    memory: Dict[str, Any],
    *,
    user: str,
    assistant: str,
    max_turns: int = 8,
) -> Dict[str, Any]:
    """Append a user/assistant turn to the rolling turn buffer.

    Keeps at most *max_turns* most recent turns, dropping older ones.

    Args:
        memory:     Current memory dict (not modified in-place).
        user:       User message text.
        assistant:  Assistant response text.
        max_turns:  Maximum number of turns to retain.

    Returns:
        A new memory dict with the turn appended.
    """
    memory = dict(memory or {})
    turns: List[Dict[str, Any]] = list(memory.get("recent_turns") or [])
    turns.append(
        {
            "ts_utc": _utc_now_iso(),
            "user": (user or "").strip(),
            "assistant": (assistant or "").strip(),
        }
    )
    memory["recent_turns"] = turns[-max_turns:]
    return memory


def add_do_not_repeat(
    memory: Dict[str, Any], rule: str, max_rules: int = 12
) -> Dict[str, Any]:
    """Append a do-not-repeat rule to the memory.

    Duplicate or empty rules are silently ignored.  At most *max_rules* rules
    are retained (oldest removed first).

    Args:
        memory:    Current memory dict (not modified in-place).
        rule:      Rule text to enforce (e.g. ``"Do not repeat unit costs"``).
        max_rules: Maximum number of rules to retain.

    Returns:
        A new memory dict with the rule appended (if not already present).
    """
    memory = dict(memory or {})
    rules = [
        r
        for r in (memory.get("do_not_repeat") or [])
        if isinstance(r, str) and r.strip()
    ]
    rule = (rule or "").strip()
    if rule and rule not in rules:
        rules.append(rule)
    memory["do_not_repeat"] = rules[-max_rules:]
    return memory


def build_persistent_context(memory: Dict[str, Any]) -> str:
    """Create a compact text block to inject into the system prompt."""
    if not memory:
        return ""

    parts: List[str] = []

    summary = (memory.get("conversation_summary") or "").strip()
    if summary:
        parts.append("PERSISTENT CONVERSATION SUMMARY (compressed):\n" + summary)

    rules = [
        r
        for r in (memory.get("do_not_repeat") or [])
        if isinstance(r, str) and r.strip()
    ]
    if rules:
        rules_block = "\n".join(f"- {r}" for r in rules)
        parts.append(
            "DO-NOT-REPEAT RULES (learned from prior corrections/errors):\n"
            + rules_block
        )

    prefs = [
        p for p in (memory.get("preferences") or []) if isinstance(p, str) and p.strip()
    ]
    if prefs:
        prefs_block = "\n".join(f"- {p}" for p in prefs)
        parts.append("OPERATOR PREFERENCES:\n" + prefs_block)

    return "\n\n".join(parts).strip()
