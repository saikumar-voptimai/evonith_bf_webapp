"""Feedback lesson generation helpers for FurnaceMind."""

from __future__ import annotations

from typing import Any


def build_feedback_lesson_prompt(feedback: Any) -> tuple[str, str]:
    """Build system and user prompts for extracting one feedback lesson."""
    system_prompt = (
        "You convert operator feedback into one reusable instruction for a blast "
        "furnace assistant. Return one concise lesson. Do not mention SQL, UI, or metadata."
    )
    user_prompt = f"""
User question:
{getattr(feedback, "raw_user_message", "") or ""}

Assistant response:
{getattr(feedback, "prev_assistant_message", "") or ""}

Feedback polarity:
{getattr(feedback, "polarity", "")}

Feedback comment:
{getattr(feedback, "feedback_text", "") or ""}

Write one lesson the assistant should remember for future similar questions.
"""
    return system_prompt, user_prompt.strip()


def generate_feedback_lesson(feedback: Any, llm_client: Any) -> str:
    """Generate a reusable lesson from one feedback row."""
    system_prompt, user_prompt = build_feedback_lesson_prompt(feedback)
    try:
        lesson = llm_client.generate(system_prompt, user_prompt).strip()
    except Exception:
        lesson = ""
    if lesson:
        return lesson
    comment = getattr(feedback, "feedback_text", "") or ""
    polarity = getattr(feedback, "polarity", "") or "feedback"
    return f"When similar questions appear, account for this {polarity} feedback: {comment}".strip()
