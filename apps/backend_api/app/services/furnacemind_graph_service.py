"""Run FurnaceMind's bounded model-and-tool loop as a backend LangGraph.

The runtime in this file coordinates one assistant turn through three nodes:

1. ``call_model`` sends the accumulated messages and available tool schemas to
   a caller-supplied model adapter.
2. ``execute_tools`` runs each tool requested by the model, appends structured
   tool results, and returns control to the model.
3. ``finalize`` guarantees a user-visible response when the model finishes or a
   configured safety limit stops the loop.

The graph owns orchestration concerns: provider-neutral input and output types,
JSON-serializable state, iteration and tool-call limits, recursion protection,
response-length limiting, hidden ``<think>`` removal, safe handling of raw tool
exceptions, warnings, counters, and a node-execution trace.

This file does not connect the graph to FastAPI routes, Streamlit components,
database persistence, a specific model provider, or concrete furnace tools.
Those integrations must implement the model and tool protocols defined here
and pass the adapters to ``FurnaceMindGraphRuntime``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, TypedDict, cast

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.furnacemind_safety_service import warning

log = logging.getLogger(__name__)

GraphStatus = Literal["running", "completed", "limit_reached"]
ToolStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class FurnaceMindGraphToolCall:
    """Describe one tool call requested by the model.

    Attributes:
        name: Provider-independent name of the tool to execute.
        arguments: Named arguments supplied by the model. The runtime converts
            them to detached JSON-compatible data before execution.
        call_id: Provider-generated identifier used to associate the tool result
            with the assistant request. The runtime creates a stable identifier
            when the model adapter omits one.
    """

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True)
class FurnaceMindGraphModelResponse:
    """Represent one response returned by the graph's model adapter.

    Attributes:
        content: Assistant text returned for this model turn. Hidden reasoning
            is removed and the configured response-length limit is applied
            before the text is stored in graph state.
        tool_calls: Zero or more tools requested by the model. An empty tuple
            means the content is treated as the final assistant response.
        metadata: Optional provider information safe to preserve with the
            assistant message, such as a model identifier or usage summary.
    """

    content: str = ""
    tool_calls: tuple[FurnaceMindGraphToolCall, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FurnaceMindGraphToolResult:
    """Represent the normalized outcome of one tool execution.

    Attributes:
        output: Successful tool output. Strings are passed to the model as-is;
            other values are converted to JSON-compatible data.
        status: ``"completed"`` for success or ``"failed"`` for failure.
        error_code: Stable machine-readable failure code, if execution failed.
        error_message: Safe user/model-readable failure description. Tool
            adapters must not include credentials or other sensitive values.
    """

    output: Any = None
    status: ToolStatus = "completed"
    error_code: str | None = None
    error_message: str | None = None


class FurnaceMindGraphModel(Protocol):
    """Define the provider-independent contract for model integrations.

    Implementations translate graph messages and tool schemas into a provider
    request, then translate the provider response into
    :class:`FurnaceMindGraphModelResponse`. Provider SDK objects must not be
    placed directly into graph state.
    """

    def __call__(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> FurnaceMindGraphModelResponse:
        """Generate the next assistant response for the current graph state.

        Args:
            messages: Detached conversation messages accumulated by the graph.
            tools: Provider-neutral tool schemas available for this model call.

        Returns:
            Assistant content, requested tool calls, and optional safe metadata.
        """
        ...


class FurnaceMindGraphToolExecutor(Protocol):
    """Define the provider-independent contract for tool integrations.

    Implementations dispatch a normalized tool call to an approved backend tool
    and return :class:`FurnaceMindGraphToolResult`. Authorization, allow-listing,
    timeouts, and domain-specific validation remain the adapter's responsibility.
    """

    def __call__(
        self, tool_call: FurnaceMindGraphToolCall
    ) -> FurnaceMindGraphToolResult:
        """Execute one normalized model-requested tool call.

        Args:
            tool_call: Tool name, arguments, and graph-assigned call identifier.

        Returns:
            A successful tool output or a safe structured failure.
        """
        ...


class FurnaceMindGraphState(TypedDict):
    """Define the JSON-serializable state shared by every graph node.

    Fields:
        messages: Conversation messages accumulated across model and tool turns.
        available_tools: Provider-neutral schemas exposed to the model.
        pending_tool_calls: Normalized calls waiting for tool execution.
        tool_results: Structured outcomes of every executed tool call.
        final_response: User-visible assistant text selected during finalization.
        status: Current or terminal runtime status.
        stop_reason: Reason finalization was selected, such as
            ``"model_response"``, ``"iteration_limit"``, or
            ``"tool_call_limit"``.
        iterations: Number of completed tool-execution rounds. One round may
            contain multiple tool calls from a single model response.
        model_calls: Total number of model adapter invocations.
        tool_call_count: Total number of individual tools executed.
        warnings: Safe structured warnings collected during the run.
        trace: Ordered names of graph nodes executed during the run.
        metadata: Detached caller metadata preserved across graph nodes.
    """

    messages: list[dict[str, Any]]
    available_tools: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    final_response: str
    status: GraphStatus
    stop_reason: str | None
    iterations: int
    model_calls: int
    tool_call_count: int
    warnings: list[dict[str, Any]]
    trace: list[str]
    metadata: dict[str, Any]


class FurnaceMindGraphRuntime:
    """Compile and run the bounded FurnaceMind orchestration workflow.

    A runtime instance compiles one LangGraph and reuses it for invocations. Each
    invocation starts with detached caller input, alternates between model and
    tool nodes while tool calls remain, and ends at the finalization node. Model
    and tool behavior is injected through protocols, keeping graph routing
    independent from providers, APIs, persistence, the UI, and domain tools.

    Two proactive limits bound the loop. ``max_iterations`` controls completed
    tool-execution rounds, while ``max_tool_calls`` controls individual tool
    calls across all rounds. LangGraph's recursion limit provides an additional
    defensive guard if normal routing fails to terminate.

    Attributes:
        settings: Backend safety and response-size configuration.
        model: Adapter used by the ``call_model`` node.
        tool_executor: Optional adapter used by the ``execute_tools`` node.
        max_iterations: Maximum completed tool-execution rounds per invocation.
        max_tool_calls: Maximum individual tool executions per invocation.
        graph: Compiled LangGraph state machine.
    """

    def __init__(
        self,
        *,
        model: FurnaceMindGraphModel,
        tool_executor: FurnaceMindGraphToolExecutor | None = None,
        settings: BackendSettings | None = None,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        """Configure and compile a bounded FurnaceMind graph.

        Args:
            model: Provider adapter called whenever the graph needs a model turn.
            tool_executor: Optional adapter used to execute requested tools.
            settings: Backend settings; loaded from the environment when omitted.
            max_iterations: Optional override for completed tool-execution
                rounds. A round can execute multiple calls from one model turn.
            max_tool_calls: Optional override for the total number of individual
                tool calls allowed across the invocation.

        Raises:
            ValueError: If either configured execution limit is not positive.
        """
        self.settings = settings or load_backend_settings()
        self.model = model
        self.tool_executor = tool_executor
        self.max_iterations = int(
            max_iterations
            if max_iterations is not None
            else self.settings.furnacemind_graph_max_iterations
        )
        self.max_tool_calls = int(
            max_tool_calls
            if max_tool_calls is not None
            else self.settings.furnacemind_max_tool_calls_per_run
        )
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.graph = self._build_graph()

    def invoke(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> FurnaceMindGraphState:
        """Run one bounded model/tool workflow.

        Input collections are copied through a JSON-safe conversion before graph
        execution, so neither the caller's messages nor tool schemas are mutated.

        Args:
            messages: Conversation messages to provide to the model adapter.
            tools: Tool schemas the model may select during this run.
            metadata: Optional request metadata to preserve in the final state.

        Returns:
            A detached, JSON-serializable state containing the final response,
            messages, tool results, execution counters, warnings, and trace.

        Raises:
            ApiError: If model execution fails, a model response or tool call has
                an invalid contract shape, or graph recursion exceeds its
                defensive safety limit. Ordinary tool failures are returned in
                state instead of being raised.
        """

        initial_state: FurnaceMindGraphState = {
            "messages": cast(list[dict[str, Any]], _json_safe(list(messages))),
            "available_tools": cast(list[dict[str, Any]], _json_safe(list(tools))),
            "pending_tool_calls": [],
            "tool_results": [],
            "final_response": "",
            "status": "running",
            "stop_reason": None,
            "iterations": 0,
            "model_calls": 0,
            "tool_call_count": 0,
            "warnings": [],
            "trace": [],
            "metadata": cast(dict[str, Any], _json_safe(dict(metadata or {}))),
        }
        try:
            result = self.graph.invoke(
                initial_state,
                {"recursion_limit": (self.max_iterations * 2) + 6},
            )
        except GraphRecursionError as exc:  # defensive; proactive guards should win
            raise ApiError(
                "FURNACEMIND_GRAPH_RECURSION_LIMIT",
                "FurnaceMind graph exceeded its safe execution limit.",
                status_code=500,
            ) from exc
        return cast(FurnaceMindGraphState, _json_safe(dict(result)))

    def _build_graph(self) -> Any:
        """Compile the model, tool-execution, and finalization state machine.

        The graph always starts at ``call_model``. A model response containing
        accepted tool calls routes to ``execute_tools`` and then back to the
        model. A final model response or a safety-limit stop routes to
        ``finalize``, which connects to LangGraph's ``END`` node.

        Returns:
            The compiled LangGraph executable used by ``invoke``.
        """
        builder = StateGraph(FurnaceMindGraphState)
        builder.add_node("call_model", self._call_model)
        builder.add_node("execute_tools", self._execute_tools)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            self._route_after_model,
            {"execute_tools": "execute_tools", "finalize": "finalize"},
        )
        builder.add_edge("execute_tools", "call_model")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _call_model(self, state: FurnaceMindGraphState) -> dict[str, Any]:
        """Request the next model action and update graph routing state.

        The node passes accumulated messages and available schemas to the model
        adapter. It removes hidden reasoning, caps visible response length,
        normalizes tool calls, records safe response metadata, and applies the
        iteration and tool-call limits before allowing execution to continue.

        Args:
            state: Current graph state containing conversation messages, tool
                schemas, counters, and warnings from earlier nodes.

        Returns:
            A partial state update containing the appended assistant message,
            normalized pending calls, final response when no calls remain,
            updated counters and warnings, stop reason, and trace entry.

        Raises:
            ApiError: If model execution fails or returns an invalid response.
        """
        try:
            response = self.model(
                messages=state["messages"],
                tools=state["available_tools"],
            )
        except ApiError:
            raise
        except Exception as exc:
            log.warning(
                "furnacemind.graph.model_failed error_type=%s",
                type(exc).__name__,
            )
            raise ApiError(
                "FURNACEMIND_GRAPH_MODEL_FAILED",
                "FurnaceMind graph model execution failed.",
                status_code=502,
            ) from exc
        if not isinstance(response, FurnaceMindGraphModelResponse):
            raise ApiError(
                "FURNACEMIND_GRAPH_MODEL_RESPONSE_INVALID",
                "FurnaceMind graph model returned an invalid response.",
                status_code=502,
            )

        model_call_number = state["model_calls"] + 1
        content = _strip_hidden_reasoning(response.content)[
            : self.settings.furnacemind_max_response_chars
        ]
        pending = [
            _normalise_tool_call(item, model_call_number, index)
            for index, item in enumerate(response.tool_calls, start=1)
        ]
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if pending:
            assistant_message["tool_calls"] = [
                {
                    "id": item["id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                }
                for item in pending
            ]
        response_metadata = cast(dict[str, Any], _json_safe(dict(response.metadata)))
        if response_metadata:
            assistant_message["metadata"] = response_metadata

        stop_reason: str | None = None
        final_response = ""
        new_warnings = list(state["warnings"])
        if pending and state["iterations"] >= self.max_iterations:
            pending = []
            stop_reason = "iteration_limit"
            new_warnings.append(
                warning(
                    "FURNACEMIND_GRAPH_ITERATION_LIMIT",
                    "FurnaceMind stopped after reaching the configured graph iteration limit.",
                    {"max_iterations": self.max_iterations},
                )
            )
        elif pending and state["tool_call_count"] + len(pending) > self.max_tool_calls:
            pending = []
            stop_reason = "tool_call_limit"
            new_warnings.append(
                warning(
                    "FURNACEMIND_GRAPH_TOOL_CALL_LIMIT",
                    "FurnaceMind stopped before exceeding the configured tool-call limit.",
                    {"max_tool_calls": self.max_tool_calls},
                )
            )
        elif not pending:
            stop_reason = "model_response"
            final_response = content

        return {
            "messages": [*state["messages"], assistant_message],
            "pending_tool_calls": pending,
            "final_response": final_response,
            "stop_reason": stop_reason,
            "model_calls": model_call_number,
            "warnings": new_warnings,
            "trace": [*state["trace"], "call_model"],
        }

    def _execute_tools(self, state: FurnaceMindGraphState) -> dict[str, Any]:
        """Execute all pending tool calls and append model-readable results.

        The node executes every call from the latest accepted model response in
        order. Tool failures become model-readable result messages and warnings,
        allowing the next model turn to explain or recover from the failure.
        Raw unexpected exception messages are not included in graph state.

        Args:
            state: Graph state containing the normalized pending tool calls and
                all messages and tool results accumulated so far.

        Returns:
            A partial state update with one tool message and structured result
            per call, cleared pending calls, incremented execution counters,
            warnings, and an execution trace entry.
        """
        messages = list(state["messages"])
        results = list(state["tool_results"])
        new_warnings = list(state["warnings"])

        for item in state["pending_tool_calls"]:
            call = FurnaceMindGraphToolCall(
                name=str(item["name"]),
                arguments=cast(dict[str, Any], _json_safe(item["arguments"])),
                call_id=str(item["id"]),
            )
            result = self._execute_one_tool(call)
            result_record = {
                "tool_call_id": call.call_id,
                "name": call.name,
                "status": result.status,
                "output": _json_safe(result.output),
                "error_code": result.error_code,
                "error_message": result.error_message,
            }
            results.append(result_record)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "content": _tool_message_content(result),
                }
            )
            if result.status == "failed":
                new_warnings.append(
                    warning(
                        result.error_code or "FURNACEMIND_GRAPH_TOOL_FAILED",
                        result.error_message or "FurnaceMind tool execution failed.",
                        {"tool_name": call.name},
                    )
                )

        return {
            "messages": messages,
            "pending_tool_calls": [],
            "tool_results": results,
            "iterations": state["iterations"] + 1,
            "tool_call_count": state["tool_call_count"]
            + len(state["pending_tool_calls"]),
            "trace": [*state["trace"], "execute_tools"],
            "warnings": new_warnings,
        }

    def _execute_one_tool(
        self,
        call: FurnaceMindGraphToolCall,
    ) -> FurnaceMindGraphToolResult:
        """Execute one tool call and normalize failures into graph-safe results.

        A missing executor, a structured ``ApiError``, an unexpected exception,
        or an invalid adapter return value is converted to a failed
        ``FurnaceMindGraphToolResult``. Unexpected exception text is logged only
        by exception type and is not returned to the model.

        Args:
            call: Normalized tool call produced by the model node.

        Returns:
            The adapter result, or a structured failure when the executor is
            unavailable, raises an exception, or returns an invalid value.
        """
        if self.tool_executor is None:
            return FurnaceMindGraphToolResult(
                status="failed",
                error_code="FURNACEMIND_GRAPH_TOOL_EXECUTOR_UNAVAILABLE",
                error_message="FurnaceMind tool executor is unavailable.",
            )
        try:
            result = self.tool_executor(call)
        except ApiError as exc:
            return FurnaceMindGraphToolResult(
                status="failed",
                error_code=exc.code,
                error_message=exc.message,
            )
        except Exception as exc:
            log.warning(
                "furnacemind.graph.tool_failed tool_name=%s error_type=%s",
                call.name,
                type(exc).__name__,
            )
            return FurnaceMindGraphToolResult(
                status="failed",
                error_code="FURNACEMIND_GRAPH_TOOL_FAILED",
                error_message="FurnaceMind tool execution failed.",
            )
        if not isinstance(result, FurnaceMindGraphToolResult):
            return FurnaceMindGraphToolResult(
                status="failed",
                error_code="FURNACEMIND_GRAPH_TOOL_RESPONSE_INVALID",
                error_message="FurnaceMind tool returned an invalid response.",
            )
        return result

    @staticmethod
    def _route_after_model(state: FurnaceMindGraphState) -> str:
        """Select the node that follows a model response.

        Args:
            state: Graph state after ``call_model`` has applied safety limits.

        Returns:
            ``"execute_tools"`` when accepted pending calls remain; otherwise
            ``"finalize"`` for a final model response or limit stop.
        """
        return "execute_tools" if state["pending_tool_calls"] else "finalize"

    @staticmethod
    def _finalize(state: FurnaceMindGraphState) -> dict[str, Any]:
        """Guarantee a visible response and assign the terminal graph status.

        If the model supplied no usable final text, this node creates a stable
        fallback explaining the applicable limit or the absence of a response.

        Args:
            state: Graph state after a final model response or safety-limit stop.

        Returns:
            A partial terminal-state update containing the final response,
            ``"completed"`` or ``"limit_reached"`` status, stop reason, and
            final trace entry.
        """
        stop_reason = state["stop_reason"] or "no_response"
        response = state["final_response"].strip()
        if not response:
            if stop_reason == "iteration_limit":
                response = (
                    "FurnaceMind stopped after reaching the configured graph "
                    "iteration limit."
                )
            elif stop_reason == "tool_call_limit":
                response = (
                    "FurnaceMind stopped before exceeding the configured "
                    "tool-call limit."
                )
            else:
                response = "FurnaceMind did not generate a final response."
        status: GraphStatus = (
            "limit_reached"
            if stop_reason in {"iteration_limit", "tool_call_limit"}
            else "completed"
        )
        return {
            "final_response": response,
            "status": status,
            "stop_reason": stop_reason,
            "trace": [*state["trace"], "finalize"],
        }


def _normalise_tool_call(
    tool_call: FurnaceMindGraphToolCall,
    model_call_number: int,
    index: int,
) -> dict[str, Any]:
    """Convert one provider-neutral tool call into serializable graph state.

    Args:
        tool_call: Tool call returned by the model adapter.
        model_call_number: One-based number of the current model invocation.
        index: One-based position of this call in the model response.

    Returns:
        A dictionary containing a stable ID, normalized name, and safe arguments.

    Raises:
        ApiError: If the model adapter returned an unsupported tool-call value.
    """
    if not isinstance(tool_call, FurnaceMindGraphToolCall):
        raise ApiError(
            "FURNACEMIND_GRAPH_TOOL_CALL_INVALID",
            "FurnaceMind graph model returned an invalid tool call.",
            status_code=502,
        )
    call_id = str(tool_call.call_id or f"call_{model_call_number}_{index}")
    return {
        "id": call_id,
        "name": str(tool_call.name or "").strip(),
        "arguments": cast(dict[str, Any], _json_safe(dict(tool_call.arguments))),
    }


def _tool_message_content(result: FurnaceMindGraphToolResult) -> str:
    """Serialize a tool result for the next model conversation turn.

    Args:
        result: Structured result returned by the tool adapter.

    Returns:
        Plain text for successful string outputs. Structured successful outputs
        and failures are returned as deterministic JSON strings.
    """
    if result.status == "failed":
        return json.dumps(
            {
                "status": "failed",
                "error_code": result.error_code or "FURNACEMIND_GRAPH_TOOL_FAILED",
                "error_message": result.error_message
                or "FurnaceMind tool execution failed.",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if isinstance(result.output, str):
        return result.output
    return json.dumps(
        _json_safe(result.output),
        ensure_ascii=False,
        sort_keys=True,
    )


def _strip_hidden_reasoning(value: Any) -> str:
    """Remove hidden reasoning blocks from model-visible response content.

    Args:
        value: Raw model content. ``None`` and other values are converted to text.

    Returns:
        Trimmed text with every case-insensitive ``<think>...</think>`` block,
        including multiline blocks, removed.
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        str(value or ""),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _json_safe(value: Any) -> Any:
    """Deep-copy a value into data that can be serialized as JSON.

    The JSON encode/decode round trip detaches caller- and adapter-owned mutable
    objects from graph state. Values unsupported by the JSON encoder are stored
    using ``str(value)``.

    Args:
        value: Adapter or caller data that may contain non-JSON-native values.

    Returns:
        A detached value composed of JSON-compatible dictionaries, lists,
        strings, numbers, booleans, and null values.

    Notes:
        This helper provides serialization and detachment, not secret redaction.
        Callers and adapters must remove sensitive values before passing data to
        the graph.
    """

    return json.loads(json.dumps(value, default=str, ensure_ascii=False))
