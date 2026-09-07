"""LangGraph workflow for the FurnaceMind tool-calling agent.

This module replaces the old hand-written model/tool loop with a small
LangGraph state machine while keeping the rest of FurnaceMind unchanged. The
Interactive and scheduled callers prepare their own system prompt and runtime
context before this workflow starts. The graph accepts injected progress and
tool-dispatch boundaries so the same model loop can run in Streamlit or in a
restricted background worker.

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
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.llm.llm_client import OpenRouterClient

_MAX_ITERATIONS = 8

ToolDispatcher = Callable[..., str]
ActivityCallback = Callable[[], None]

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
    """State object passed between all FurnaceMind LangGraph nodes.

    Fields:
        llm: Configured OpenRouter client used for every model call.
        messages: OpenAI-compatible conversation list. Nodes mutate this list by
            appending assistant messages, tool outputs, and optional MRAG visual
            messages.
        tools: OpenAI-compatible tool schemas exposed to the model.
        status_box: Streamlit-shaped progress sink; unattended runs use a no-op
            implementation.
        final_response: Final assistant response once the model stops requesting
            tools.
        last_tool_result: Last tool output, used as a defensive fallback if the
            workflow reaches its iteration limit before a final response exists.
        iterations: Number of tool-execution rounds already completed.
        tool_dispatcher: Runtime-specific function/tool execution boundary.
        allowed_tool_names: Optional server-side tool policy allowlist.
        fail_on_tool_error: Whether malformed or failed tool calls abort the run.
        activity_callback: Optional lease and cancellation checkpoint.
        tool_events: Sanitized names, outcomes, and result sizes for auditing.
    """

    llm: OpenRouterClient
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    status_box: Any
    final_response: str
    last_tool_result: str | None
    iterations: int
    tool_dispatcher: ToolDispatcher
    allowed_tool_names: frozenset[str] | None
    fail_on_tool_error: bool
    activity_callback: ActivityCallback | None
    tool_events: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class FurnaceMindGraphResult:
    """Structured result from one FurnaceMind model/tool workflow.

    ``tool_events`` deliberately contains only tool names, success flags, and
    result sizes. Tool arguments and results can contain plant data or secrets
    and therefore do not belong in scheduled-run metadata.
    """

    final_response: str
    tool_events: tuple[dict[str, object], ...]
    iterations: int


class FurnaceMindGraphToolError(RuntimeError):
    """Raised when strict unattended execution cannot safely run a tool call."""


class NullStatusBox:
    """No-op progress sink used by non-Streamlit FurnaceMind callers."""

    def status(self, _label: str, *, expanded: bool = False) -> None:
        """Discard a progress update from the graph."""

        del expanded


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


def _parse_tool_arguments_strict(raw_arguments: Any) -> dict[str, Any]:
    """Parse tool arguments or reject malformed model output.

    Interactive chat keeps the forgiving parser above so the model can recover
    from a malformed call. Scheduled execution uses this strict variant because
    silently replacing invalid arguments with an empty object can create a
    successful-looking report for the wrong data window.
    """

    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        raise FurnaceMindGraphToolError("A scheduled tool call omitted its arguments.")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise FurnaceMindGraphToolError(
            "A scheduled tool call contained malformed JSON arguments."
        ) from exc
    if not isinstance(parsed, dict):
        raise FurnaceMindGraphToolError(
            "A scheduled tool call must use a JSON object for its arguments."
        )
    return parsed


def _report_activity(state: FurnaceMindGraphState) -> None:
    """Notify an optional runner callback before and after external work."""

    callback = state.get("activity_callback")
    if callback is not None:
        callback()


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
    _report_activity(state)
    try:
        completion = state["llm"].chat_completions(
            messages=state["messages"],
            tools=state["tools"],
            tool_choice="auto",
        )
    finally:
        _report_activity(state)
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

        succeeded = True
        try:
            allowed_tool_names = state.get("allowed_tool_names")
            if allowed_tool_names is not None and tool_name not in allowed_tool_names:
                raise FurnaceMindGraphToolError(
                    f"Tool {tool_name or 'unknown_tool'} is not allowed by this job policy."
                )
            raw_arguments = function.get("arguments")
            arguments = (
                _parse_tool_arguments_strict(raw_arguments)
                if state.get("fail_on_tool_error")
                else _parse_tool_arguments(raw_arguments)
            )
            _report_activity(state)
            result = state["tool_dispatcher"](
                name=tool_name,
                arguments=arguments,
            )
        except Exception as exc:
            succeeded = False
            if state.get("fail_on_tool_error"):
                if isinstance(exc, FurnaceMindGraphToolError):
                    raise
                raise FurnaceMindGraphToolError(
                    f"Scheduled tool {tool_name or 'unknown_tool'} failed."
                ) from exc
            result = _tool_error_result(tool_name, exc)
        finally:
            _report_activity(state)
        state["tool_events"].append(
            {
                "name": tool_name or "unknown_tool",
                "succeeded": succeeded,
                "result_characters": len(result),
            }
        )
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


def run_furnacemind_graph(
    *,
    llm: OpenRouterClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    status_box: Any | None = None,
    tool_dispatcher: ToolDispatcher | None = None,
    allowed_tool_names: frozenset[str] | set[str] | None = None,
    fail_on_tool_error: bool = False,
    activity_callback: ActivityCallback | None = None,
) -> FurnaceMindGraphResult:
    """Run one complete workflow and return safe structured execution data.

    Args:
        llm: Configured OpenRouter chat client used by the graph.
        messages: OpenAI-compatible conversation state prepared by the page. The
            list is mutated in-place with assistant turns, tool results, and any
            MRAG visual messages produced during the run.
        tools: OpenAI-compatible schemas for tools the model may call.
        status_box: Optional Streamlit-shaped progress sink. Headless callers
            receive a no-op sink when it is omitted.
        tool_dispatcher: Callable that executes one named local tool.
        allowed_tool_names: Optional independent runtime allowlist. This is
            enforced even if a model emits a tool absent from its schemas.
        fail_on_tool_error: Raise tool and argument failures instead of returning
            them to the model as conversational tool messages.
        activity_callback: Optional lease/cancellation check invoked around model
            and tool calls.

    Returns:
        Final text plus a sanitized tool event summary and iteration count.
    """
    _ensure_langchain_debug_compat()
    result = build_furnacemind_graph().invoke(
        {
            "llm": llm,
            "messages": messages,
            "tools": tools,
            "status_box": status_box or NullStatusBox(),
            "final_response": "",
            "last_tool_result": None,
            "iterations": 0,
            "tool_dispatcher": tool_dispatcher or execute_openai_tool_call,
            "allowed_tool_names": (
                frozenset(allowed_tool_names)
                if allowed_tool_names is not None
                else None
            ),
            "fail_on_tool_error": fail_on_tool_error,
            "activity_callback": activity_callback,
            "tool_events": [],
        }
    )
    return FurnaceMindGraphResult(
        final_response=result["final_response"],
        tool_events=tuple(dict(event) for event in result["tool_events"]),
        iterations=result["iterations"],
    )


def run_furnacemind_graph_loop(
    *,
    llm: OpenRouterClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    status_box: Any,
) -> str:
    """Run the backward-compatible interactive graph entry point.

    The Streamlit wrapper historically consumed only a string. New unattended
    callers should use :func:`run_furnacemind_graph` to receive structured,
    privacy-bounded execution metadata and strict tool enforcement.
    """

    return run_furnacemind_graph(
        llm=llm,
        messages=messages,
        tools=tools,
        status_box=status_box,
    ).final_response
