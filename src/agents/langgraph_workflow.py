"""LangGraph workflow for the FurnaceMind tool-calling agent.

The existing FurnaceMind runtime already owns model configuration, tool schemas,
Streamlit session state, MRAG image queues, and skill prompt injection. This
module adds LangGraph orchestration around that runtime without replacing those
contracts: callers still pass an ``OpenRouterClient`` and OpenAI-compatible tool
schemas, while the graph manages the multi-step ``agent -> tools -> agent`` flow.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.llm.llm_client import OpenRouterClient


def _execute_tool_call(*, name: str, arguments: dict[str, Any]) -> str:
    """Execute a FurnaceMind tool through the existing dispatcher.

    The import is intentionally lazy so importing this workflow does not require
    every optional plotting/data dependency used by the full tool module.
    """
    from agents.furnace_tools import execute_openai_tool_call

    return execute_openai_tool_call(name=name, arguments=arguments)


def _consume_pending_visual_message() -> dict[str, Any] | None:
    """Consume queued MRAG visual evidence from the existing tool module.

    ``search_knowledge_docs`` can queue a follow-up multimodal user message
    after returning text evidence. The graph consumes that queued message here
    so the model can inspect image evidence on the next reasoning step.
    """
    from agents.furnace_tools import consume_pending_mrag_image_message

    return consume_pending_mrag_image_message()


def _ensure_langchain_debug_compat() -> None:
    """Provide the legacy ``langchain.debug`` flag expected by callbacks.

    The project uses ``langchain-core`` directly and may not have a full
    ``langchain`` package exposing the deprecated root-level ``debug`` attribute.
    LangGraph still asks LangChain Core for callback settings during invoke/stream,
    so this guard keeps the workflow compatible with the installed dependency set.
    """
    try:
        import langchain
    except Exception:
        return
    if not hasattr(langchain, "debug"):
        langchain.debug = False


class MessagesState(TypedDict):
    """State carried between LangGraph nodes for one FurnaceMind turn.

    ``messages`` intentionally uses OpenAI/OpenRouter-compatible dictionaries
    instead of LangChain message objects. Keeping that shape preserves the
    existing system prompt, multimodal MRAG image messages, tool-call records,
    and downstream Streamlit/tool expectations.
    """

    messages: list[dict[str, Any]]


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK tool call object into a message-safe dictionary.

    OpenRouter returns SDK objects, while the rest of FurnaceMind stores plain
    dictionaries in Streamlit and SQL history. Normalizing here keeps later graph
    nodes independent of provider-specific response classes.
    """
    function = getattr(tool_call, "function", None)
    return {
        "id": str(getattr(tool_call, "id", "")),
        "type": str(getattr(tool_call, "type", "function") or "function"),
        "function": {
            "name": str(getattr(function, "name", "") if function else ""),
            "arguments": str(
                getattr(function, "arguments", "{}") if function else "{}"
            ),
        },
    }


def _assistant_message_from_completion(completion: Any) -> dict[str, Any]:
    """Extract the assistant message returned by the existing OpenRouter client.

    The returned dictionary preserves assistant text and OpenAI-style tool calls
    exactly as the legacy loop expected, allowing existing UI rendering and tool
    execution code to continue consuming the same message shape.
    """
    message = completion.choices[0].message
    content = getattr(message, "content", None) or ""
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        assistant_message["tool_calls"] = [
            _tool_call_to_dict(tool_call) for tool_call in tool_calls
        ]
    return assistant_message


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Parse tool-call arguments into the dictionary expected by legacy tools.

    Tool arguments usually arrive as a JSON string. Bad or non-object JSON is
    treated as an empty argument set so one malformed model call cannot crash the
    graph before the tool node can return a controlled error message.
    """
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_name(tool_call: dict[str, Any]) -> str:
    """Return the function name from an OpenAI-style tool call dictionary.

    Keeping this lookup in one helper protects the tool node from missing
    ``function`` blocks and keeps provider-specific message parsing out of the
    orchestration logic.
    """
    function = tool_call.get("function") or {}
    return str(function.get("name") or "").strip()


def _tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Return parsed function arguments from an OpenAI-style tool call.

    The graph stores the original tool-call dictionary, then converts arguments
    only at execution time so message history remains faithful to the model
    response while tools still receive normal Python dictionaries.
    """
    function = tool_call.get("function") or {}
    return _parse_tool_arguments(function.get("arguments"))


def create_context_node() -> Any:
    """Build the graph node that carries prepared prompt and skill context.

    FurnaceMind resolves selected skills, semantically relevant skills, feedback,
    memory, and MRAG pre-retrieval before the agent loop starts. Those blocks are
    already present in the system/user messages passed into the graph. This node
    makes that prepared context an explicit workflow step and gives LangGraph a
    stable place to normalize turn state before model reasoning begins.
    """

    def context_node(state: MessagesState) -> MessagesState:
        return {"messages": list(state["messages"])}

    return context_node


def create_tool_node() -> Any:
    """Build the graph node that executes existing FurnaceMind tools.

    Tool execution remains delegated to ``execute_openai_tool_call`` so current
    Streamlit session side effects are preserved: fetched datasets, generated
    Plotly figures, knowledge retrieval traces, and MRAG visual evidence queues
    continue to work exactly as they do in the legacy agent loop.
    """

    def tool_node(state: MessagesState) -> MessagesState:
        messages = list(state["messages"])
        last_message = messages[-1] if messages else {}
        tool_calls = list(last_message.get("tool_calls") or [])
        tool_messages: list[dict[str, Any]] = []
        visual_messages: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            name = _tool_name(tool_call)
            tool_call_id = str(tool_call.get("id") or "")
            try:
                result = _execute_tool_call(
                    name=name,
                    arguments=_tool_arguments(tool_call),
                )
            except Exception as exc:
                result = f"Error executing tool {name}: {exc}"

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": str(result),
                }
            )

            if name == "search_knowledge_docs":
                pending_visual = _consume_pending_visual_message()
                if pending_visual is not None:
                    visual_messages.append(pending_visual)

        # Tool responses must immediately satisfy all assistant tool_call ids.
        # Visual MRAG user messages are appended only after those ToolMessages.
        return {"messages": messages + tool_messages + visual_messages}

    return tool_node


def create_model_node(
    *,
    llm: OpenRouterClient,
    tools: Sequence[dict[str, Any]],
) -> Any:
    """Build the reasoning node using the caller-provided model and tools.

    The page owns model selection and tool schema construction. This node only
    calls that configured client with the current graph messages, then appends
    the normalized assistant response back into workflow state.
    """

    def call_model(state: MessagesState) -> MessagesState:
        completion = llm.chat_completions(
            messages=state["messages"],
            tools=list(tools),
            tool_choice="auto",
        )
        return {
            "messages": state["messages"]
            + [_assistant_message_from_completion(completion)]
        }

    return call_model


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Route to tool execution when the latest assistant message requested tools.

    LangGraph uses this conditional edge after every model response. Assistant
    messages with tool calls loop through the tool node; plain assistant answers
    terminate the graph for the current FurnaceMind turn.
    """
    messages = state["messages"]
    if not messages:
        return "__end__"
    latest = messages[-1]
    if latest.get("role") == "assistant" and latest.get("tool_calls"):
        return "tools"
    return "__end__"


def create_agent_workflow(
    *,
    llm: OpenRouterClient,
    tools: Sequence[dict[str, Any]],
) -> Any:
    """Compile the FurnaceMind LangGraph workflow for one agent run.

    The compiled graph handles one chat turn at a time. Conversation persistence
    remains in the existing PostgreSQL/session-history layer, avoiding shared
    in-memory checkpoints between users while still giving LangGraph explicit
    state transitions inside the turn.
    """
    _ensure_langchain_debug_compat()
    workflow = StateGraph(MessagesState)
    workflow.add_node("context", create_context_node())
    workflow.add_node("agent", create_model_node(llm=llm, tools=tools))
    workflow.add_node("tools", create_tool_node())
    workflow.add_edge(START, "context")
    workflow.add_edge("context", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()
