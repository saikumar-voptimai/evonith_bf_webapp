"""Agent loop entry point for FurnaceMind.

The public ``run_agent_loop`` API is kept for existing call sites, but the loop
is now orchestrated by ``agents.furnacemind.graph`` using LangGraph.
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
    """
    Drive the LangGraph tool-calling workflow and render the final response.

    Args:
        - llm: OpenRouterClient - Configured OpenRouter client.
        - messages: list[dict[str, Any]] - OpenAI-compatible conversation messages.
        - tools: list[dict[str, Any]] - OpenAI-compatible tool schemas.
        - status_box: Any - Streamlit status placeholder used for tool progress.
        - response_box: Any - Streamlit placeholder used to render the final response.

    Returns:
        - str
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
