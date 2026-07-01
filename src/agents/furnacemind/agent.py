"""Agent loop - OpenRouter tool-calling with LangGraph orchestration.

``run_agent_loop`` preserves the existing FurnaceMind contract: callers provide
an ``OpenRouterClient``, OpenAI-compatible tool schemas, and Streamlit status
placeholders. LangGraph now owns the multi-step ``agent -> tools -> agent``
workflow state while the existing tool dispatcher continues to execute the real
application tools.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from langgraph.errors import GraphRecursionError

from agents.langgraph_workflow import create_agent_workflow
from agents.llm.llm_client import OpenRouterClient

# Status labels shown in the UI while a tool is running.
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

_MAX_ITERATIONS = 8


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models.

    Some OpenRouter models can include private reasoning traces in the response
    body. The UI should show only the final answer, so cleanup stays at the
    agent boundary before Streamlit renders the message.
    """
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def _thread_id() -> str:
    """Return a per-run thread id for LangGraph runtime configuration.

    The app already persists conversation state in SQL/session history and passes
    the current history into each call. A fresh id avoids accidental checkpoint
    sharing if a checkpointer is added later.
    """
    return f"furnacemind-turn-{uuid4()}"


def _latest_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool result messages from the current graph state.

    LangGraph emits the full message list after each node update. Filtering the
    current state lets the Streamlit wrapper keep the latest tool result as a
    useful fallback if the model stops before producing a final assistant answer.
    """
    return [message for message in messages if message.get("role") == "tool"]


def run_agent_loop(
    *,
    llm: OpenRouterClient,
    messages: list[dict],
    tools: list[dict],
    status_box,
    response_box,
) -> str:
    """Run one FurnaceMind agent turn through the LangGraph workflow.

    Args:
        llm: Configured OpenRouter client supplied by the page.
        messages: OpenAI/OpenRouter-format conversation state for this turn.
        tools: Tool schemas returned by ``get_openai_tool_schemas``.
        status_box: Streamlit placeholder used for tool-running status labels.
        response_box: Streamlit placeholder used to render the final answer.

    Returns:
        Final assistant response with reasoning-model thinking blocks removed.
    """
    app = create_agent_workflow(llm=llm, tools=tools)
    final_response = ""
    last_tool_result = ""
    final_messages = list(messages)
    config = {
        "configurable": {"thread_id": _thread_id()},
        "recursion_limit": (_MAX_ITERATIONS * 2) + 1,
    }

    try:
        for event in app.stream(
            {"messages": list(messages)},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_state in event.items():
                graph_messages = node_state.get("messages", [])
                if not graph_messages:
                    continue
                final_messages = graph_messages
                latest_msg = graph_messages[-1]

                if node_name == "agent":
                    tool_calls = latest_msg.get("tool_calls") or []
                    if tool_calls:
                        for tool_call in tool_calls:
                            function = tool_call.get("function") or {}
                            name = str(function.get("name") or "")
                            label = _TOOL_LABELS.get(name, f"Running {name}...")
                            status_box.status(label, expanded=False)
                    else:
                        final_response = _strip_thinking(
                            str(latest_msg.get("content") or "")
                        )

                elif node_name == "tools":
                    tool_messages = _latest_tool_messages(graph_messages)
                    if tool_messages:
                        last_tool_result = str(tool_messages[-1].get("content") or "")

    except GraphRecursionError:
        final_response = (
            last_tool_result
            or "I reached the tool-call limit before producing a final answer."
        )

    status_box.empty()

    if not final_response:
        final_response = last_tool_result or "No response generated."

    messages[:] = final_messages
    response_box.markdown(final_response)
    return final_response
