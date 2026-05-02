"""Tests for the minimal FurnaceMind LangGraph orchestration path."""

from __future__ import annotations

from typing import Any

from agents.furnacemind import graph as fm_graph


class _StatusBox:
    """
    Represents a test double for the Streamlit status placeholder.
    """

    def __init__(self) -> None:
        """
        Initialize the status-box test double.

        Returns:
            - None
        """
        self.labels: list[str] = []

    def status(self, label: str, expanded: bool = False) -> None:  # noqa: ARG002
        """
        Capture a status label passed by the graph.

        Args:
            - label: str - Status label emitted by the graph.
            - expanded: bool - Whether the status UI should be expanded.

        Returns:
            - None
        """
        self.labels.append(label)


class _Function:
    """
    Represents a fake SDK function-call object.
    """

    name = "fetch_ml_data"
    arguments = '{"start_time": "2026-01-01"}'


class _ToolCall:
    """
    Represents a fake SDK tool-call object.
    """

    id = "call_1"
    function = _Function()


class _Message:
    """
    Represents a fake SDK chat message object.
    """

    def __init__(self, *, content: str = "", tool_calls: Any = None) -> None:
        """
        Initialize a fake SDK chat message.

        Args:
            - content: str - Message content returned by the fake model.
            - tool_calls: Any - Tool calls returned by the fake model.

        Returns:
            - None
        """
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    """
    Represents a fake SDK choice wrapper.
    """

    def __init__(self, message: _Message) -> None:
        """
        Initialize a fake SDK choice.

        Args:
            - message: _Message - Fake message wrapped by the choice.

        Returns:
            - None
        """
        self.message = message


class _Completion:
    """
    Represents a fake SDK completion response.
    """

    def __init__(self, message: _Message) -> None:
        """
        Initialize a fake SDK completion.

        Args:
            - message: _Message - Fake message returned by the model.

        Returns:
            - None
        """
        self.choices = [_Choice(message)]


class _FakeLLM:
    """
    Represents a fake LLM that first requests a tool and then returns text.
    """

    def __init__(self) -> None:
        """
        Initialize the fake LLM call counter.

        Returns:
            - None
        """
        self.calls = 0

    def chat_completions(  # noqa: ARG002
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> _Completion:
        """
        Return a fake tool call on the first call and final text on the second.

        Args:
            - messages: list[dict[str, Any]] - OpenAI-compatible conversation messages.
            - tools: list[dict[str, Any]] - OpenAI-compatible tool schemas.
            - tool_choice: str - Tool-choice mode requested by the graph.

        Returns:
            - _Completion
        """
        self.calls += 1
        if self.calls == 1:
            return _Completion(_Message(tool_calls=[_ToolCall()]))
        return _Completion(_Message(content="<think>hidden</think>Final answer."))


def test_furnacemind_graph_executes_tool_then_returns_final_response(monkeypatch):
    """
    Verify the graph executes a requested tool and returns the final response.

    Args:
        - monkeypatch: Any - Pytest fixture used to replace the tool dispatcher.

    Returns:
        - None
    """
    calls = []

    def fake_execute_openai_tool_call(*, name: str, arguments: dict[str, Any]) -> str:
        """
        Capture tool dispatch arguments and return a fake tool result.

        Args:
            - name: str - Tool name requested by the graph.
            - arguments: dict[str, Any] - Tool arguments requested by the graph.

        Returns:
            - str
        """
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
        "tool_call_id": "call_1",
        "name": "fetch_ml_data",
        "content": "tool result",
    }
