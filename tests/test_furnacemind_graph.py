"""Tests for the FurnaceMind LangGraph orchestration path."""

from __future__ import annotations

from typing import Any

from agents.furnacemind import graph as fm_graph


class _StatusBox:
    """Test double for the Streamlit status placeholder."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def status(self, label: str, expanded: bool = False) -> None:  # noqa: ARG002
        """Capture a status label emitted by the graph."""
        self.labels.append(label)


class _Function:
    """Fake OpenAI SDK function-call object."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """Fake OpenAI SDK tool-call object."""

    def __init__(self, name: str, arguments: str = "{}") -> None:
        self.id = f"call_{name}"
        self.function = _Function(name, arguments)


class _Message:
    """Fake OpenAI SDK chat message object."""

    def __init__(self, *, content: str = "", tool_calls: Any = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    """Fake OpenAI SDK choice wrapper."""

    def __init__(self, message: _Message) -> None:
        self.message = message


class _Completion:
    """Fake OpenAI SDK completion response."""

    def __init__(self, message: _Message) -> None:
        self.choices = [_Choice(message)]


class _FakeLLM:
    """Fake LLM that requests one tool before returning final text."""

    def __init__(
        self,
        *,
        tool_name: str = "fetch_ml_data",
        tool_arguments: str = '{"start_time": "2026-01-01"}',
        final_text: str = "<think>hidden</think>Final answer.",
    ) -> None:
        self.calls = 0
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        self.final_text = final_text

    def chat_completions(  # noqa: ARG002
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
    ) -> _Completion:
        """Return a fake tool call first, then final assistant text."""
        self.calls += 1
        if self.calls == 1:
            return _Completion(
                _Message(tool_calls=[_ToolCall(self.tool_name, self.tool_arguments)])
            )
        return _Completion(_Message(content=self.final_text))


def test_furnacemind_graph_executes_tool_then_returns_final_response(monkeypatch):
    """The graph should run requested tools and continue to final text."""
    calls = []

    def fake_execute_openai_tool_call(*, name: str, arguments: dict[str, Any]) -> str:
        calls.append((name, arguments))
        return "tool result"

    monkeypatch.setattr(
        fm_graph,
        "execute_openai_tool_call",
        fake_execute_openai_tool_call,
    )

    messages = [{"role": "user", "content": "check data"}]
    status_box = _StatusBox()

    response = fm_graph.run_furnacemind_graph_loop(
        llm=_FakeLLM(),
        messages=messages,
        tools=[],
        status_box=status_box,
    )

    assert response == "Final answer."
    assert calls == [("fetch_ml_data", {"start_time": "2026-01-01"})]
    assert status_box.labels == ["Reading ML dataset..."]
    assert messages[-2] == {
        "role": "tool",
        "tool_call_id": "call_fetch_ml_data",
        "name": "fetch_ml_data",
        "content": "tool result",
    }


def test_furnacemind_graph_preserves_mrag_visual_message(monkeypatch):
    """Knowledge-doc search should pass visual MRAG inputs to the next turn."""
    visual_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
        ],
    }

    monkeypatch.setattr(
        fm_graph,
        "execute_openai_tool_call",
        lambda **_: "knowledge result",
    )
    monkeypatch.setattr(
        fm_graph,
        "consume_pending_mrag_image_message",
        lambda: visual_message,
    )

    messages = [{"role": "user", "content": "inspect the chart"}]
    response = fm_graph.run_furnacemind_graph_loop(
        llm=_FakeLLM(
            tool_name="search_knowledge_docs",
            tool_arguments='{"query": "chart"}',
            final_text="Chart answer.",
        ),
        messages=messages,
        tools=[],
        status_box=_StatusBox(),
    )

    assert response == "Chart answer."
    assert visual_message in messages
    assert messages.index(visual_message) > 0


def test_furnacemind_graph_recovers_after_tool_error(monkeypatch):
    """Tool failures should be returned to the model instead of crashing."""

    def failing_tool(*, name: str, arguments: dict[str, Any]) -> str:  # noqa: ARG001
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        fm_graph,
        "execute_openai_tool_call",
        failing_tool,
    )

    llm = _FakeLLM(
        tool_name="fetch_online_data",
        tool_arguments='{"measurement": "bf"}',
        final_text="Telemetry is unavailable right now.",
    )
    messages = [{"role": "user", "content": "check live telemetry"}]

    response = fm_graph.run_furnacemind_graph_loop(
        llm=llm,
        messages=messages,
        tools=[],
        status_box=_StatusBox(),
    )

    assert response == "Telemetry is unavailable right now."
    assert llm.calls == 2
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    assert tool_messages == [
        {
            "role": "tool",
            "tool_call_id": "call_fetch_online_data",
            "name": "fetch_online_data",
            "content": "Tool `fetch_online_data` failed: RuntimeError: database unavailable",
        }
    ]


class _ToolChoiceRecordingLLM:
    """Fake LLM that records graph-level tool choice behavior."""

    def __init__(self, *, final_text: str = "Prompt-driven final answer.") -> None:
        self.tool_choices: list[str | dict[str, Any]] = []
        self.final_text = final_text

    def chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],  # noqa: ARG002
        tools: list[dict[str, Any]],  # noqa: ARG002
        tool_choice: str | dict[str, Any] = "auto",
    ) -> _Completion:
        """Record the tool choice and return a final assistant response."""
        self.tool_choices.append(tool_choice)
        return _Completion(_Message(content=self.final_text))


def test_graph_leaves_tool_routing_to_prompt_policy():
    """The graph should not force domain-specific tool choices itself."""
    llm = _ToolChoiceRecordingLLM()
    messages = [
        {
            "role": "user",
            "content": "Describe the chart in the uploaded BMO Analysis PPT.",
        }
    ]
    tools = [
        {"type": "function", "function": {"name": "search_knowledge_docs"}},
        {"type": "function", "function": {"name": "execute_python_plot"}},
    ]

    response = fm_graph.run_furnacemind_graph_loop(
        llm=llm,
        messages=messages,
        tools=tools,
        status_box=_StatusBox(),
    )

    assert response == "Prompt-driven final answer."
    assert llm.tool_choices == ["auto"]
