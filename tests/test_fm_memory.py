from __future__ import annotations

from agents.memory import fm_memory


class FakeSummaryLLM:
    """Fake memory-summary LLM used to inspect summary prompts."""

    def __init__(self, response: str = "updated cumulative summary") -> None:
        """Create the fake LLM with a configured summary response."""
        self.response = response
        self.calls: list[dict[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Record the summary prompts and return the configured response."""
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.response


def _messages(count: int) -> list[dict]:
    """Build text chat messages with stable ids for summary-window tests."""
    return [
        {
            "role": "user" if index % 2 else "assistant",
            "content": f"message {index}",
            "display": f"message {index}",
            "type": "text",
            "message_id": f"msg-{index}",
            "conversation_id": "conv-1",
        }
        for index in range(1, count + 1)
    ]


def test_normal_summary_uses_messages_after_last_saved_summary() -> None:
    """Verify the next full summary window starts after saved coverage."""
    memory = {
        "conversation_summary": "summary through message 8",
        "source_message_id_end": "msg-8",
        "summarized_message_count": 8,
    }
    chat_history = _messages(16)
    llm = FakeSummaryLLM()

    assert fm_memory.should_generate_memory_summary(
        chat_history,
        window=8,
        memory=memory,
    )
    assert fm_memory.summary_source_message_ids(
        chat_history,
        window=8,
        memory=memory,
    ) == ("msg-9", "msg-16")

    result = fm_memory.generate_memory_summary(
        memory,
        chat_history=chat_history,
        llm=llm,
        summary_system_prompt="system",
        summary_token_limit=2000,
        window=8,
    )

    assert result["conversation_summary"] == "updated cumulative summary"
    assert result["source_message_id_start"] == "msg-9"
    assert result["source_message_id_end"] == "msg-16"
    assert result["summarized_message_count"] == 16
    assert "message 9" in llm.calls[0]["user_prompt"]
    assert "message 16" in llm.calls[0]["user_prompt"]
    assert "message 17" not in llm.calls[0]["user_prompt"]


def test_partial_tail_is_not_summarized_until_forced() -> None:
    """Verify a short unsummarized tail is skipped during normal turns."""
    memory = {
        "conversation_summary": "summary through message 16",
        "source_message_id_end": "msg-16",
        "summarized_message_count": 16,
    }
    chat_history = _messages(20)
    llm = FakeSummaryLLM()

    assert not fm_memory.should_generate_memory_summary(
        chat_history,
        window=8,
        memory=memory,
    )
    assert fm_memory.summary_source_message_ids(
        chat_history,
        window=8,
        memory=memory,
    ) == (None, None)

    unchanged = fm_memory.generate_memory_summary(
        memory,
        chat_history=chat_history,
        llm=llm,
        summary_system_prompt="system",
        summary_token_limit=2000,
        window=8,
    )

    assert unchanged["conversation_summary"] == "summary through message 16"
    assert llm.calls == []


def test_forced_summary_flushes_partial_tail() -> None:
    """Verify force mode summarizes the final short tail of a conversation."""
    memory = {
        "conversation_summary": "summary through message 16",
        "source_message_id_end": "msg-16",
        "summarized_message_count": 16,
    }
    chat_history = _messages(20)
    llm = FakeSummaryLLM()

    assert fm_memory.should_generate_memory_summary(
        chat_history,
        window=8,
        memory=memory,
        force=True,
    )
    assert fm_memory.summary_source_message_ids(
        chat_history,
        window=8,
        memory=memory,
        force=True,
    ) == ("msg-17", "msg-20")

    result = fm_memory.generate_memory_summary(
        memory,
        chat_history=chat_history,
        llm=llm,
        summary_system_prompt="system",
        summary_token_limit=2000,
        window=8,
        force=True,
    )

    assert result["conversation_summary"] == "updated cumulative summary"
    assert result["source_message_id_start"] == "msg-17"
    assert result["source_message_id_end"] == "msg-20"
    assert result["summarized_message_count"] == 20
    assert "message 16" in llm.calls[0]["user_prompt"]
    assert "message 17" in llm.calls[0]["user_prompt"]
    assert "message 20" in llm.calls[0]["user_prompt"]


def test_missed_boundary_can_catch_up_with_first_unsummarized_window() -> None:
    """Verify summary generation catches up from the first pending window."""
    chat_history = _messages(10)
    llm = FakeSummaryLLM()

    assert fm_memory.should_generate_memory_summary(chat_history, window=8)
    assert fm_memory.summary_source_message_ids(chat_history, window=8) == (
        "msg-1",
        "msg-8",
    )

    result = fm_memory.generate_memory_summary(
        {},
        chat_history=chat_history,
        llm=llm,
        summary_system_prompt="system",
        summary_token_limit=2000,
        window=8,
    )

    assert result["source_message_id_start"] == "msg-1"
    assert result["source_message_id_end"] == "msg-8"
    assert result["summarized_message_count"] == 8
    assert "message 1" in llm.calls[0]["user_prompt"]
    assert "message 8" in llm.calls[0]["user_prompt"]
    assert "message 9" not in llm.calls[0]["user_prompt"]
