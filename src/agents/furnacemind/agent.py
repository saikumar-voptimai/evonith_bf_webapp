"""FurnaceMind agent-loop entry point backed by LangGraph.

The public ``run_agent_loop`` function is intentionally kept stable for the
Streamlit page. Internally, the model/tool loop is now executed by the
LangGraph workflow in :mod:`agents.furnacemind.graph`.
"""

from __future__ import annotations

from typing import Any

from agents.furnacemind.graph import run_furnacemind_graph_loop
from agents.llm.llm_client import OpenRouterClient


def run_agent_loop(
    *,
    llm: OpenRouterClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    status_box: Any,
    response_box: Any,
) -> str:
    """Run the LangGraph workflow and render the final assistant response.

    Args:
        llm: Configured OpenRouter client used for chat completions.
        messages: OpenAI-compatible conversation messages. The graph mutates
            this list by appending assistant/tool turns so existing callers can
            inspect the executed workflow.
        tools: OpenAI-compatible tool schemas exposed to the model.
        status_box: Streamlit placeholder used for tool progress updates.
        response_box: Streamlit placeholder used to render the final response.

    Returns:
        Final assistant response with reasoning ``<think>`` blocks removed.
    """
    final_response = run_furnacemind_graph_loop(
        llm=llm,
        messages=messages,
        tools=tools,
        status_box=status_box,
    )
    status_box.empty()
    response_box.markdown(final_response)
    return final_response
