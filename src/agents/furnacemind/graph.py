"""LangGraph workflow for the FurnaceMind tool-calling agent.

This module replaces the old hand-written model/tool loop with a small
LangGraph state machine while keeping the rest of FurnaceMind unchanged. The
Streamlit page still prepares the system prompt, selected skill context,
semantically retrieved skill context, memory, feedback lessons, and knowledge
context before this workflow starts.

The graph is responsible for only the agent loop:

1. Send the current message state and tool schemas to the LLM.
2. If the LLM asks for tools, execute those tools through the existing
   FurnaceMind tool dispatcher.
3. Append tool outputs, optional MRAG visual messages, and assistant turns back
   into the shared state.
4. Continue until the model returns a final answer or the iteration limit is
   reached.

Tool-routing policy stays in ``agents.furnacemind.prompts`` so this graph remains
an orchestration layer instead of accumulating domain-specific routing rules.

Tool failures are returned to the model as normal tool messages. That lets the
assistant explain the problem or choose a fallback instead of letting a Streamlit
page exception break the user session.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

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
    "web_search": "Searching web...",
    "web_scrape_ingest": "Indexing web source...",
    "execute_python_plot": "Generating plot...",
}


class FurnaceMindGraphState(TypedDict):
    """State object passed between all FurnaceMind LangGraph nodes.

    Fields:
        llm: Configured OpenRouter client used for every model call.
        messages: OpenAI-compatible conversation list. Nodes mutate this list by
            appending assistant messages, tool outputs, and optional MRAG visual
            messages.
        tools: OpenAI-compatible tool schemas exposed to the model.
        status_box: Streamlit placeholder used to show the currently running
            tool label.
        final_response: Final assistant response once the model stops requesting
            tools.
        last_tool_result: Last tool output, used as a defensive fallback if the
            workflow reaches its iteration limit before a final response exists.
        iterations: Number of tool-execution rounds already completed.
    """

    llm: OpenRouterClient
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    status_box: Any
    final_response: str
    last_tool_result: str | None
    iterations: int


def execute_openai_tool_call(*, name: str, arguments: dict[str, Any]) -> str:
    """Run one FurnaceMind tool through the existing dispatcher.

    Args:
        name: OpenAI function/tool name selected by the model.
        arguments: Parsed JSON arguments for that tool.

    Returns:
        String result produced by the existing FurnaceMind tool layer.

    Notes:
        The import is intentionally lazy. Importing ``agents.furnace_tools`` can
        pull in optional plotting and data dependencies, so delaying it keeps this
        graph module importable in lightweight tests and app startup paths.
    """
    from agents.furnace_tools import execute_openai_tool_call as _execute

    return _execute(name=name, arguments=arguments)


def consume_pending_mrag_image_message() -> dict[str, Any] | None:
    """Read the visual MRAG message queued by knowledge-document search.

    Returns:
        A user-role OpenAI multimodal message containing image inputs, or
        ``None`` when the latest knowledge search did not prepare visual context.

    Notes:
        ``search_knowledge_docs`` can retrieve both text chunks and image chunks.
        The text comes back through the normal tool result. The image payload is
        stored separately by the tool layer, then appended here so the next model
        turn can reason over the retrieved visuals with a vision-capable model.
    """
    from agents.furnace_tools import consume_pending_mrag_image_message as _consume

    return _consume()


def _ensure_langchain_debug_compat() -> None:
    """Provide the ``langchain.debug`` attribute expected by this dependency set.

    LangGraph internally asks LangChain whether debug mode is enabled. The
    versions currently locked for this application can expose that value through
    a package shape that does not define ``langchain.debug``. Creating the
    attribute before invocation keeps the graph stable without changing runtime
    behavior. This shim can be removed after the LangChain and LangGraph package
    versions are aligned.
    """
    try:
        import langchain
    except Exception:
        return
    if not hasattr(langchain, "debug"):
        langchain.debug = False


def _strip_thinking(text: str) -> str:
    """Remove hidden reasoning blocks before storing or rendering output.

    Args:
        text: Raw assistant content returned by the chat model.

    Returns:
        Assistant content with any ``<think>...</think>`` block removed.
    """
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def _tool_call_attr(tool_call: Any, name: str, default: Any = None) -> Any:
    """Read a tool-call field from SDK objects and dictionary test doubles.

    Args:
        tool_call: OpenAI SDK object, nested function object, or dictionary.
        name: Attribute/key to read.
        default: Value returned when the attribute/key is missing.

    Returns:
        The requested value from either object attributes or dictionary keys.
    """
    if isinstance(tool_call, dict):
        return tool_call.get(name, default)
    return getattr(tool_call, name, default)


def _normalise_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Convert model tool-call objects into OpenAI-compatible dictionaries.

    Args:
        tool_calls: Tool calls returned by the LLM client. They may be SDK
            objects in production or dictionaries in tests.

    Returns:
        A list of dictionaries with ``id``, ``type``, and ``function`` keys. The
        normalized shape is what gets appended to ``messages`` before tool
        outputs are added.
    """
    if not tool_calls:
        return []

    normalised: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = _tool_call_attr(tool_call, "function", {})
        function_name = _tool_call_attr(function, "name", "")
        function_args = _tool_call_attr(function, "arguments", "{}")
        normalised.append(
            {
                "id": _tool_call_attr(tool_call, "id"),
                "type": _tool_call_attr(tool_call, "type", "function"),
                "function": {
                    "name": function_name,
                    "arguments": function_args,
                },
            }
        )
    return normalised


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Parse LLM-emitted tool arguments into a dictionary safe for dispatch.

    Args:
        raw_arguments: JSON string, dictionary, empty value, or malformed value
            emitted by the model for a tool call.

    Returns:
        Parsed dictionary arguments. Invalid JSON and non-dictionary JSON values
        are treated as an empty argument dictionary so one malformed tool call
        does not crash the workflow.
    """
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_error_result(tool_name: str, exc: Exception) -> str:
    """Format a tool exception as model-readable tool output.

    Args:
        tool_name: Tool name that failed. Empty names are replaced with a stable
            placeholder for debugging.
        exc: Exception raised by the tool dispatcher.

    Returns:
        A short string that can be appended as a normal ``role='tool'`` message.
        The graph then loops back to the model, allowing a graceful final answer
        instead of surfacing the exception in the Streamlit UI.
    """
    clean_name = tool_name or "unknown_tool"
    return f"Tool `{clean_name}` failed: {type(exc).__name__}: {exc}"


def _call_model(state: FurnaceMindGraphState) -> FurnaceMindGraphState:
    """LangGraph node that asks the model for the next action.

    Args:
        state: Current workflow state containing messages, available tools, and
            any prior tool outputs.

    Returns:
        The same state object after appending an assistant message. If the model
        returned tool calls, they are normalized and stored on that assistant
        message. If the model returned plain content, ``final_response`` is set
        and the graph can finalize.
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
    """LangGraph node that executes tool calls requested by the model.

    Args:
        state: Current workflow state whose latest assistant message contains
            OpenAI-compatible ``tool_calls``.

    Returns:
        The same state object after appending one ``role='tool'`` message per
        tool call, optional MRAG visual context, and an incremented iteration
        count.

    Error handling:
        Tool exceptions are converted into tool-result messages. This keeps the
        graph moving and gives the model a chance to explain the failure or use
        another path instead of raising the exception to Streamlit.
    """
    assistant_message = state["messages"][-1]
    pending_visual_message: dict[str, Any] | None = None

    for tool_call in assistant_message.get("tool_calls", []):
        function = tool_call.get("function", {})
        tool_name = str(function.get("name") or "")
        label = _TOOL_LABELS.get(tool_name, f"Running {tool_name}...")
        state["status_box"].status(label, expanded=False)

        try:
            result = execute_openai_tool_call(
                name=tool_name,
                arguments=_parse_tool_arguments(function.get("arguments")),
            )
        except Exception as exc:
            result = _tool_error_result(tool_name, exc)
        state["last_tool_result"] = result
        state["messages"].append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": tool_name,
                "content": result,
            }
        )
        if tool_name == "search_knowledge_docs":
            pending_visual_message = consume_pending_mrag_image_message()

    if pending_visual_message is not None:
        state["messages"].append(pending_visual_message)

    state["iterations"] += 1
    return state


def _should_continue(state: FurnaceMindGraphState) -> str:
    """Choose the next graph edge after a model response.

    Args:
        state: Current workflow state after ``call_model`` has appended the
            latest assistant message.

    Returns:
        ``'execute_tools'`` when the assistant requested tool calls and the
        iteration limit has not been reached; otherwise ``'finalize'``.
    """
    if state["iterations"] >= _MAX_ITERATIONS:
        return "finalize"
    latest_message = state["messages"][-1] if state["messages"] else {}
    return "execute_tools" if latest_message.get("tool_calls") else "finalize"


def _after_tools(state: FurnaceMindGraphState) -> str:
    """Choose the next graph edge after tool execution.

    Args:
        state: Current workflow state after ``execute_tools`` has appended tool
            results and incremented the iteration counter.

    Returns:
        ``'call_model'`` while another model turn is allowed; otherwise
        ``'finalize'`` when the iteration cap has been reached.
    """
    return "finalize" if state["iterations"] >= _MAX_ITERATIONS else "call_model"


def _finalize(state: FurnaceMindGraphState) -> FurnaceMindGraphState:
    """LangGraph node that guarantees a user-visible final response.

    Args:
        state: Current workflow state after either a final model response or an
            iteration-limit stop.

    Returns:
        The same state object with ``final_response`` populated. If the model did
        not provide final text, the last tool result is used as a defensive
        fallback; if no tool ran, a generic fallback message is used.
    """
    if not state.get("final_response"):
        state["final_response"] = (
            state.get("last_tool_result") or "No response generated."
        )
    return state


@lru_cache(maxsize=1)
def build_furnacemind_graph():
    """Compile the FurnaceMind LangGraph workflow once per process.

    Returns:
        A compiled graph with three nodes:
        ``call_model`` asks the LLM for the next action, ``execute_tools`` runs
        requested tools, and ``finalize`` guarantees a response. Conditional
        edges loop between model and tools until the model stops requesting
        tools or the iteration cap is reached.
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
    """Run one complete FurnaceMind model/tool workflow.

    Args:
        llm: Configured OpenRouter chat client used by the graph.
        messages: OpenAI-compatible conversation state prepared by the page. The
            list is mutated in-place with assistant turns, tool results, and any
            MRAG visual messages produced during the run.
        tools: OpenAI-compatible schemas for tools the model may call.
        status_box: Streamlit placeholder used to display progress labels while
            tools execute.

    Returns:
        Final assistant response text. Hidden reasoning blocks have already been
        stripped before this value is set.
    """
    _ensure_langchain_debug_compat()
    result = build_furnacemind_graph().invoke(
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
