"""LangGraph orchestration for the FurnaceMind tool-calling loop.

This is the minimal migration path: keep the existing OpenRouter client,
OpenAI-style tool schemas, and ``execute_openai_tool_call`` dispatcher, while
moving the loop control into a LangGraph state graph.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.furnace_tools import execute_openai_tool_call
from agents.llm.llm_client import OpenRouterClient

_MAX_ITERATIONS = 8

_TOOL_LABELS: dict[str, str] = {
    "fetch_ml_data": "Reading ML dataset...",
    "concat_datasets": "Stitching datasets...",
    "fetch_online_data": "Fetching live telemetry...",
    "fetch_offline_data": "Fetching offline report...",
    "merge_furnace_data": "Merging datasets...",
    "load_static_shift_data": "Loading shift data...",
    "search_shift_history": "Searching shift history...",
    "search_knowledge_docs": "Searching knowledge docs...",
    "execute_python_plot": "Generating plot...",
}


class FurnaceMindGraphState(TypedDict):
    """
    Represents the state carried through the minimal FurnaceMind LangGraph.
    """

    llm: OpenRouterClient
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    status_box: Any
    final_response: str
    last_tool_result: str | None
    iterations: int


def _strip_thinking(text: str) -> str:
    """
    Remove reasoning-model ``<think>...</think>`` blocks from model output.

    Args:
        - text: str - Raw model response text.

    Returns:
        - str
    """
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def _normalise_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """
    Convert OpenAI SDK tool call objects into OpenAI-compatible dictionaries.

    Args:
        - tool_calls: Any - Tool call objects returned by the OpenAI-compatible SDK.

    Returns:
        - list[dict[str, Any]]
    """
    if not tool_calls:
        return []
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in tool_calls
    ]


def _call_model(state: FurnaceMindGraphState) -> FurnaceMindGraphState:
    """
    Call the configured OpenRouter model with the current message list.

    Args:
        - state: FurnaceMindGraphState - Current graph state.

    Returns:
        - FurnaceMindGraphState
    """
    completion = state["llm"].chat_completions(
        messages=state["messages"],
        tools=state["tools"],
        tool_choice="auto",
    )
    msg = completion.choices[0].message

    content = _strip_thinking(getattr(msg, "content", None) or "")
    tool_calls = _normalise_tool_calls(getattr(msg, "tool_calls", None))

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    else:
        state["final_response"] = content

    state["messages"].append(assistant_message)
    return state


def _execute_tools(state: FurnaceMindGraphState) -> FurnaceMindGraphState:
    """
    Execute model-requested tools using the existing FurnaceMind dispatcher.

    Args:
        - state: FurnaceMindGraphState - Current graph state containing tool calls.

    Returns:
        - FurnaceMindGraphState
    """
    assistant_message = state["messages"][-1]

    for tool_call in assistant_message.get("tool_calls", []):
        function = tool_call.get("function", {})
        tool_name = function.get("name") or ""
        label = _TOOL_LABELS.get(tool_name, f"Running {tool_name}...")
        state["status_box"].status(label, expanded=False)

        try:
            args = json.loads(function.get("arguments") or "{}")
        except Exception:
            args = {}

        result = execute_openai_tool_call(name=tool_name, arguments=args)
        state["last_tool_result"] = result
        state["messages"].append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": tool_name,
                "content": result,
            }
        )

    state["iterations"] += 1
    return state


def _should_continue(state: FurnaceMindGraphState) -> str:
    """
    Decide whether the graph should execute tools or finalize.

    Args:
        - state: FurnaceMindGraphState - Current graph state after a model call.

    Returns:
        - str
    """
    if state["iterations"] >= _MAX_ITERATIONS:
        return "finalize"
    latest_message = state["messages"][-1] if state["messages"] else {}
    return "execute_tools" if latest_message.get("tool_calls") else "finalize"


def _after_tools(state: FurnaceMindGraphState) -> str:
    """
    Decide whether to call the model again after tool execution.

    Args:
        - state: FurnaceMindGraphState - Current graph state after tool execution.

    Returns:
        - str
    """
    return "finalize" if state["iterations"] >= _MAX_ITERATIONS else "call_model"


def _finalize(state: FurnaceMindGraphState) -> FurnaceMindGraphState:
    """
    Ensure the graph returns a user-visible response.

    Args:
        - state: FurnaceMindGraphState - Current graph state before completion.

    Returns:
        - FurnaceMindGraphState
    """
    if not state.get("final_response"):
        state["final_response"] = (
            state.get("last_tool_result") or "No response generated."
        )
    return state


def build_furnacemind_graph():
    """
    Build and compile the minimal FurnaceMind LangGraph.

    Returns:
        - Any
    """
    graph = StateGraph(FurnaceMindGraphState)
    graph.add_node("call_model", _call_model)
    graph.add_node("execute_tools", _execute_tools)
    graph.add_node("finalize", _finalize)

    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        _should_continue,
        {
            "execute_tools": "execute_tools",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute_tools",
        _after_tools,
        {
            "call_model": "call_model",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_furnacemind_graph_loop(
    *,
    llm: OpenRouterClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    status_box: Any,
) -> str:
    """
    Run the minimal LangGraph and return the final assistant response.

    Args:
        - llm: OpenRouterClient - Configured OpenRouter client.
        - messages: list[dict[str, Any]] - OpenAI-compatible conversation messages.
        - tools: list[dict[str, Any]] - OpenAI-compatible tool schemas.
        - status_box: Any - Streamlit status placeholder used for tool progress.

    Returns:
        - str
    """
    graph = build_furnacemind_graph()
    result = graph.invoke(
        {
            "llm": llm,
            "messages": messages,
            "tools": tools,
            "status_box": status_box,
            "final_response": "",
            "last_tool_result": None,
            "iterations": 0,
        }
    )
    return result["final_response"]
