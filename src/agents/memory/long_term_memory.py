"""Long-term memory extraction helpers for FurnaceMind chat turns."""

from __future__ import annotations

from typing import Any


def build_long_term_memory_prompt(user_text: str, assistant_text: str) -> tuple[str, str]:
    """Build prompts for extracting durable chat memories."""
    system_prompt = (
        "You extract durable long-term memories for an industrial blast furnace "
        "assistant. Keep only facts, preferences, constraints, recurring operator "
        "needs, or important operational context that may help future answers. "
        "Do not memorize normal questions, temporary tasks, greetings, or content "
        "that only came from the assistant response."
    )
    user_prompt = f"""
User message:
{user_text}

Assistant response:
{assistant_text}

Return zero to five memories.
Rules:
- One memory per line.
- Each memory must be self-contained and useful later across future sessions.
- Store only durable facts explicitly stated by the user, such as user preferences,
  site/furnace identity, recurring comparison habits, constraints, or stable work context.
- Do not store temporary greetings, filler, or one-off wording.
- Do not store the assistant's explanation as a memory.
- Do not store "the user asked about X" unless the user says it is a recurring need.
- If there is nothing worth remembering, return exactly NONE.
"""
    return system_prompt, user_prompt.strip()


def extract_long_term_memories(
    *,
    user_text: str,
    assistant_text: str,
    llm_client: Any,
) -> list[str]:
    """Extract durable memories from one user/assistant turn."""
    system_prompt, user_prompt = build_long_term_memory_prompt(user_text, assistant_text)
    try:
        raw = llm_client.generate(system_prompt, user_prompt).strip()
    except Exception:
        raw = ""
    if not raw or raw.upper() == "NONE":
        return []
    memories: list[str] = []
    for line in raw.splitlines():
        clean = line.strip().lstrip("-*0123456789. )").strip()
        if _is_durable_memory(clean, user_text=user_text, assistant_text=assistant_text):
            memories.append(clean)
    return memories[:5]


def _is_durable_memory(memory: str, *, user_text: str, assistant_text: str) -> bool:
    """Return True when a candidate memory is durable enough to persist."""
    clean = " ".join(memory.strip().split())
    if not clean or clean.upper() == "NONE":
        return False
    if len(clean) > 280:
        return False

    lowered = clean.lower()
    blocked_phrases = (
        "user is requesting",
        "user asked",
        "user asks",
        "the user requested",
        "the user is asking",
        "furnacemind is",
        "assistant should",
        "answer should",
    )
    if any(phrase in lowered for phrase in blocked_phrases):
        return False

    user_lower = user_text.lower()
    durable_cues = (
        "remember",
        "usually",
        "always",
        "prefer",
        "preference",
        "my furnace",
        "our furnace",
        "i use",
        "we use",
        "i usually",
        "we usually",
        "i care",
        "we care",
        "important for me",
        "important for us",
    )
    if any(cue in user_lower for cue in durable_cues):
        return True

    memory_cues = (
        "usually",
        "prefers",
        "preference",
        "recurring",
        "uses",
        "cares about",
        "important",
        "constraint",
        "furnace",
        "shift",
    )
    return any(cue in lowered for cue in memory_cues) and len(clean.split()) <= 35
