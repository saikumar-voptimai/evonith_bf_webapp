from __future__ import annotations

import sys
import types


def _install_module(name: str, **attrs) -> None:
    """Install a lightweight import stub for unused page dependencies."""
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules.setdefault(name, module)


def _cache_resource(*args, **kwargs):
    """Return a no-op Streamlit cache decorator."""
    if args and callable(args[0]):
        return args[0]

    def decorator(func):
        """Return the wrapped function without caching."""
        return func

    return decorator


_install_module("streamlit", cache_resource=_cache_resource, session_state={})
_install_module(
    "agents.embeddings.cloud_embedding",
    CloudEmbeddingClient=type("CloudEmbeddingClient", (), {}),
)
_install_module("agents.furnace_tools", get_openai_tool_schemas=lambda: [])
_install_module("agents.furnacemind.agent", run_agent_loop=lambda **kwargs: "")
_install_module(
    "agents.furnacemind.context",
    SystemPromptContext=type("SystemPromptContext", (), {}),
)
_install_module("agents.furnacemind.skills", SkillEngine=type("SkillEngine", (), {}))
_install_module(
    "agents.llm.llm_client",
    OpenRouterClient=type("OpenRouterClient", (), {"__init__": lambda self, **kwargs: None}),
)
_install_module(
    "agents.memory.conversation_history",
    ConversationHistoryStore=type("ConversationHistoryStore", (), {}),
)
_install_module(
    "agents.memory.knowledge_vector_store",
    KnowledgeVectorStore=type("KnowledgeVectorStore", (), {}),
)
_install_module("agents.memory.vector_store", QdrantVectorStore=type("QdrantVectorStore", (), {}))
_install_module("ui.furnacemind.chat_interface")
_install_module("ui.furnacemind.feedback_flow")
_install_module(
    "utils.furnacemind.feedback_service",
    FurnaceMindFeedbackService=type("FurnaceMindFeedbackService", (), {}),
)
_install_module("utils.session", current_user_id=lambda: "user-1")
_install_module("utils.shift_windows", last_completed_shift=lambda: (None, ""))

from agents.furnacemind import ai_cooperate_page


class FakeSummaryLLM:
    """Fake summary LLM returning a deterministic cumulative summary."""

    def __init__(self, response: str = "complete cumulative summary") -> None:
        """Create the fake with a configured response."""
        self.response = response

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the configured summary text."""
        return self.response


class RaisingSummaryLLM:
    """Fake summary LLM that simulates provider failure."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Raise an error like a failed LLM request."""
        raise RuntimeError("summary failed")


class FakeSemanticMemoryService:
    """Fake semantic-memory service for page flush tests."""

    storage_available = True
    last_error = None

    def __init__(self) -> None:
        """Create the fake service and call log."""
        self.calls: list[dict] = []

    def add_summary(self, **kwargs) -> list[str]:
        """Record the semantic-memory request and return a saved fact id."""
        self.calls.append(kwargs)
        return ["fact-1"]


class FailingSemanticMemoryService:
    """Fake semantic-memory service that records a storage failure."""

    storage_available = True

    def __init__(self) -> None:
        """Create the fake with no initial error."""
        self.last_error = None

    def add_summary(self, **kwargs) -> list[str]:
        """Fail semantic extraction and expose the error via last_error."""
        self.last_error = "extract failed"
        return []


def _messages(count: int) -> list[dict]:
    """Build persisted text chat messages with stable ids."""
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


def test_new_chat_flush_saves_leftover_tail_and_indexes_semantic_memory(
    monkeypatch,
) -> None:
    """Verify New Chat force-flushes unsummarized messages before reset."""
    save_calls: list[dict] = []

    def fake_save(memory, **kwargs) -> bool:
        """Record a summary save request and report success."""
        save_calls.append({"memory": memory, **kwargs})
        return True

    monkeypatch.setattr(ai_cooperate_page.fm_memory, "save_fm_memory", fake_save)
    semantic_service = FakeSemanticMemoryService()

    result = ai_cooperate_page._flush_conversation_memory(
        chat_history=_messages(10),
        memory={
            "conversation_summary": "summary through 8",
            "source_message_id_end": "msg-8",
            "summarized_message_count": 8,
        },
        user_id="900ef580-57a1-517e-8271-c384e3785057",
        conversation_id="conv-1",
        semantic_memory_service=semantic_service,
        memory_summary_window=8,
        memory_summary_token_limit=2000,
        trigger="new_chat",
        force=True,
        memory_llm=FakeSummaryLLM(),
    )

    assert result.attempted
    assert result.summary_saved
    assert result.semantic_fact_ids == ("fact-1",)
    assert result.error is None
    assert save_calls[0]["source_message_id_start"] == "msg-9"
    assert save_calls[0]["source_message_id_end"] == "msg-10"
    assert save_calls[0]["memory"]["summarized_message_count"] == 10
    assert semantic_service.calls[0]["source_message_id_start"] == "msg-9"
    assert semantic_service.calls[0]["source_message_id_end"] == "msg-10"
    assert semantic_service.calls[0]["metadata"]["summary_trigger"] == "new_chat"


def test_new_chat_flush_does_not_save_when_summary_generation_fails(
    monkeypatch,
) -> None:
    """Verify failed final summaries do not create misleading coverage rows."""
    save_calls: list[dict] = []

    def fake_save(memory, **kwargs) -> bool:
        """Record unexpected summary saves during failure handling."""
        save_calls.append({"memory": memory, **kwargs})
        return True

    monkeypatch.setattr(ai_cooperate_page.fm_memory, "save_fm_memory", fake_save)

    result = ai_cooperate_page._flush_conversation_memory(
        chat_history=_messages(10),
        memory={
            "conversation_summary": "summary through 8",
            "source_message_id_end": "msg-8",
            "summarized_message_count": 8,
        },
        user_id="900ef580-57a1-517e-8271-c384e3785057",
        conversation_id="conv-1",
        semantic_memory_service=FakeSemanticMemoryService(),
        memory_summary_window=8,
        memory_summary_token_limit=2000,
        trigger="new_chat",
        force=True,
        memory_llm=RaisingSummaryLLM(),
    )

    assert result.attempted
    assert not result.summary_saved
    assert "summary did not complete" in str(result.error)
    assert save_calls == []


def test_new_chat_retry_indexes_already_saved_summary() -> None:
    """Verify a retry can extract facts from the latest saved summary."""
    semantic_service = FakeSemanticMemoryService()

    result = ai_cooperate_page._flush_conversation_memory(
        chat_history=_messages(10),
        memory={
            "conversation_summary": "summary through 10",
            "source_message_id_start": "msg-9",
            "source_message_id_end": "msg-10",
            "summarized_message_count": 10,
        },
        user_id="900ef580-57a1-517e-8271-c384e3785057",
        conversation_id="conv-1",
        semantic_memory_service=semantic_service,
        memory_summary_window=8,
        memory_summary_token_limit=2000,
        trigger="new_chat",
        force=True,
        retry_saved_summary=True,
        memory_llm=FakeSummaryLLM(),
    )

    assert result.attempted
    assert not result.summary_saved
    assert result.semantic_fact_ids == ("fact-1",)
    assert result.error is None
    assert semantic_service.calls[0]["summary"] == "summary through 10"
    assert semantic_service.calls[0]["metadata"]["summary_trigger"] == "new_chat_retry"


def test_new_chat_sets_error_when_semantic_retry_fails() -> None:
    """Verify a semantic retry failure keeps New Chat from silently resetting."""
    result = ai_cooperate_page._flush_conversation_memory(
        chat_history=_messages(10),
        memory={
            "conversation_summary": "summary through 10",
            "source_message_id_start": "msg-9",
            "source_message_id_end": "msg-10",
            "summarized_message_count": 10,
        },
        user_id="900ef580-57a1-517e-8271-c384e3785057",
        conversation_id="conv-1",
        semantic_memory_service=FailingSemanticMemoryService(),
        memory_summary_window=8,
        memory_summary_token_limit=2000,
        trigger="new_chat",
        force=True,
        retry_saved_summary=True,
        memory_llm=FakeSummaryLLM(),
    )

    assert result.attempted
    assert result.summary_saved
    assert "extract failed" in str(result.error)
