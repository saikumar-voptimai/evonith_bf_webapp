"""Agent loop — OpenRouter tool-calling with reasoning-model cleanup.

``run_agent_loop`` drives the tool-calling conversation until the model
produces a final text response (no tool_calls) or the iteration cap is hit.

Reasoning models (e.g. MiniMax M2.5, DeepSeek-R1) emit ``<think>…</think>``
blocks inside the content field.  ``_strip_thinking`` removes them so
operators never see the internal reasoning trace.
"""

from __future__ import annotations

import json
import re

from agents.furnace_tools import execute_openai_tool_call
from llm.llm_client import OpenRouterClient

# ── Status labels shown in the UI while a tool is running ───────────────────
_TOOL_LABELS: dict[str, str] = {
    "fetch_ml_data":        "Reading ML dataset…",
    "concat_datasets":      "Stitching datasets…",
    "fetch_online_data":    "Fetching live telemetry…",
    "fetch_offline_data":   "Fetching offline report…",
    "merge_furnace_data":   "Merging datasets…",
    "load_static_shift_data": "Loading shift data…",
    "search_shift_history": "Searching shift history…",
    "search_knowledge_docs":"Searching knowledge docs…",
    "execute_python_plot":  "Generating plot…",
}

_MAX_ITERATIONS = 8


def _strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def run_agent_loop(
    *,
    llm: OpenRouterClient,
    messages: list[dict],
    tools: list[dict],
    status_box,
    response_box,
) -> str:
    """Drive the tool-calling loop and return the final text response.

    Parameters
    ----------
    llm:          Configured OpenRouterClient.
    messages:     Full conversation history in OpenAI format (mutated in-place).
    tools:        Tool schemas from ``get_openai_tool_schemas()``.
    status_box:   ``st.empty()`` placeholder for status badges.
    response_box: ``st.empty()`` placeholder for the streaming response text.

    Returns
    -------
    str  Final assistant response (thinking blocks stripped).
    """
    final_response = ""
    last_tool_result: str | None = None

    for _ in range(_MAX_ITERATIONS):
        completion = llm.chat_completions(messages=messages, tools=tools, tool_choice="auto")
        msg = completion.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None)
        content    = _strip_thinking(getattr(msg, "content", None) or "")

        if tool_calls:
            # Append the assistant turn (with tool_calls) to the conversation.
            messages.append({
                "role":    "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                label = _TOOL_LABELS.get(tc.function.name, f"Running {tc.function.name}…")
                status_box.status(label, expanded=False)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                result = execute_openai_tool_call(name=tc.function.name, arguments=args)
                last_tool_result = result
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         tc.function.name,
                    "content":      result,
                })
            continue  # next iteration

        # No tool_calls → this is the final text response.
        final_response = content
        break

    status_box.empty()

    if not final_response:
        final_response = last_tool_result or "No response generated."

    response_box.markdown(final_response)
    return final_response
