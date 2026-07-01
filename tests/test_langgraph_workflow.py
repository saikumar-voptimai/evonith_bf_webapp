"""Regression tests for the FurnaceMind LangGraph agent workflow."""

from types import SimpleNamespace

from agents.langgraph_workflow import create_agent_workflow


def _completion(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeToolCallingLLM:
    """Small fake OpenRouter client that requests one tool, then answers."""

    def __init__(self):
        self.calls = 0

    def chat_completions(self, *, messages, tools, tool_choice):
        self.calls += 1
        if self.calls == 1:
            return _completion(
                SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name="fake_tool",
                                arguments='{"value": 2}',
                            ),
                        )
                    ],
                )
            )
        assert any(message.get("role") == "tool" for message in messages)
        return _completion(SimpleNamespace(content="final answer", tool_calls=[]))


def test_langgraph_workflow_executes_existing_tool_dispatcher(monkeypatch):
    calls = []

    def fake_execute_tool(*, name, arguments):
        calls.append((name, arguments))
        return f"tool result for {arguments['value']}"

    monkeypatch.setattr(
        "agents.langgraph_workflow._execute_tool_call",
        fake_execute_tool,
    )

    app = create_agent_workflow(
        llm=FakeToolCallingLLM(),
        tools=[{"type": "function", "function": {"name": "fake_tool"}}],
    )

    result = app.invoke(
        {"messages": [{"role": "user", "content": "run fake tool"}]},
        config={"recursion_limit": 5},
    )

    assert calls == [("fake_tool", {"value": 2})]
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result["messages"][-1]["content"] == "final answer"


def test_mrag_visual_message_is_appended_after_tool_messages(monkeypatch):
    class KnowledgeLLM(FakeToolCallingLLM):
        def chat_completions(self, *, messages, tools, tool_choice):
            self.calls += 1
            if self.calls == 1:
                return _completion(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_knowledge",
                                type="function",
                                function=SimpleNamespace(
                                    name="search_knowledge_docs",
                                    arguments='{"query": "chart"}',
                                ),
                            )
                        ],
                    )
                )
            return _completion(SimpleNamespace(content="visual answer", tool_calls=[]))

    monkeypatch.setattr(
        "agents.langgraph_workflow._execute_tool_call",
        lambda *, name, arguments: "knowledge text result",
    )
    monkeypatch.setattr(
        "agents.langgraph_workflow._consume_pending_visual_message",
        lambda: {"role": "user", "content": [{"type": "text", "text": "visual"}]},
    )

    app = create_agent_workflow(
        llm=KnowledgeLLM(),
        tools=[{"type": "function", "function": {"name": "search_knowledge_docs"}}],
    )

    result = app.invoke(
        {"messages": [{"role": "user", "content": "inspect chart"}]},
        config={"recursion_limit": 5},
    )

    roles = [message["role"] for message in result["messages"]]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    assert result["messages"][2]["tool_call_id"] == "call_knowledge"
    assert result["messages"][3]["content"][0]["text"] == "visual"
