"""Behavioral tests for the FurnaceMind backend LangGraph runtime.

This module uses deterministic model and tool adapters to verify the complete
model-to-tool loop without contacting external providers. The tests cover
successful execution, serializable and detached state, hidden-reasoning
removal, iteration and tool-call limits, safe tool-failure recovery, response
length limits, environment setting bounds, and isolation from Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.services.furnacemind_graph_service import (
    FurnaceMindGraphModelResponse,
    FurnaceMindGraphRuntime,
    FurnaceMindGraphToolCall,
    FurnaceMindGraphToolResult,
)


def _settings(tmp_path, **overrides) -> BackendSettings:
    """Build isolated backend settings with optional graph-limit overrides."""
    return BackendSettings(
        backend_env="test",
        auth_secret_key="test-secret",
        furnacemind_database_url=f"sqlite:///{(tmp_path / 'fm.db').as_posix()}",
        **overrides,
    )


class _SequenceModel:
    """Return predefined model responses while recording adapter inputs."""

    def __init__(self, responses: Sequence[FurnaceMindGraphModelResponse]) -> None:
        """Store the ordered responses that subsequent model calls consume."""
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> FurnaceMindGraphModelResponse:
        """Record one model invocation and return its next queued response."""
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        return self.responses.pop(0)


class _RepeatingToolModel:
    """Request one tool on every model turn to exercise loop protection."""

    def __init__(self) -> None:
        """Initialize the model-call counter used in generated tool calls."""
        self.calls = 0

    def __call__(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],  # noqa: ARG002
        tools: Sequence[Mapping[str, Any]],  # noqa: ARG002
    ) -> FurnaceMindGraphModelResponse:
        """Return a unique tool call for every model invocation."""
        self.calls += 1
        return FurnaceMindGraphModelResponse(
            tool_calls=(
                FurnaceMindGraphToolCall(
                    name="data_summary",
                    arguments={"round": self.calls},
                    call_id=f"call_{self.calls}",
                ),
            )
        )


def test_graph_executes_tool_then_returns_serializable_final_state(tmp_path):
    """The graph should execute a tool and return detached serializable state."""
    model = _SequenceModel(
        [
            FurnaceMindGraphModelResponse(
                tool_calls=(
                    FurnaceMindGraphToolCall(
                        name="data_summary",
                        arguments={"rows": [{"pressure": 1}]},
                        call_id="call_summary",
                    ),
                )
            ),
            FurnaceMindGraphModelResponse(
                content="<think>private reasoning</think>Pressure is stable.",
                metadata={"provider": "stub"},
            ),
        ]
    )
    executed: list[FurnaceMindGraphToolCall] = []

    def execute(call: FurnaceMindGraphToolCall) -> FurnaceMindGraphToolResult:
        """Record the requested tool call and return a structured result."""
        executed.append(call)
        return FurnaceMindGraphToolResult(output={"row_count": 1})

    original_messages = [{"role": "user", "content": "Check pressure"}]
    runtime = FurnaceMindGraphRuntime(
        model=model,
        tool_executor=execute,
        settings=_settings(tmp_path),
    )

    result = runtime.invoke(
        messages=original_messages,
        tools=[{"type": "function", "function": {"name": "data_summary"}}],
        metadata={"request_id": "req-1"},
    )

    assert result["final_response"] == "Pressure is stable."
    assert result["status"] == "completed"
    assert result["stop_reason"] == "model_response"
    assert result["iterations"] == 1
    assert result["model_calls"] == 2
    assert result["tool_call_count"] == 1
    assert result["trace"] == [
        "call_model",
        "execute_tools",
        "call_model",
        "finalize",
    ]
    assert executed == [
        FurnaceMindGraphToolCall(
            name="data_summary",
            arguments={"rows": [{"pressure": 1}]},
            call_id="call_summary",
        )
    ]
    assert result["tool_results"][0]["output"] == {"row_count": 1}
    assert "private reasoning" not in json.dumps(result)
    assert json.loads(json.dumps(result))["metadata"] == {"request_id": "req-1"}
    assert original_messages == [{"role": "user", "content": "Check pressure"}]


def test_graph_stops_repeated_tool_loop_at_configured_iteration_limit(tmp_path):
    """Repeated tool requests should stop at the configured iteration limit."""
    model = _RepeatingToolModel()
    executed: list[FurnaceMindGraphToolCall] = []

    def execute(call: FurnaceMindGraphToolCall) -> FurnaceMindGraphToolResult:
        """Record each tool call made before the iteration limit is reached."""
        executed.append(call)
        return FurnaceMindGraphToolResult(output="tool result")

    runtime = FurnaceMindGraphRuntime(
        model=model,
        tool_executor=execute,
        settings=_settings(
            tmp_path,
            furnacemind_graph_max_iterations=2,
            furnacemind_max_tool_calls_per_run=10,
        ),
    )

    result = runtime.invoke(messages=[{"role": "user", "content": "loop"}])

    assert result["status"] == "limit_reached"
    assert result["stop_reason"] == "iteration_limit"
    assert result["iterations"] == 2
    assert result["tool_call_count"] == 2
    assert len(executed) == 2
    assert model.calls == 3
    assert result["warnings"][-1]["code"] == "FURNACEMIND_GRAPH_ITERATION_LIMIT"
    assert "iteration limit" in result["final_response"]


def test_graph_converts_tool_exception_to_safe_result_and_recovers(tmp_path, caplog):
    """Tool exceptions should be redacted and returned to the model safely."""
    model = _SequenceModel(
        [
            FurnaceMindGraphModelResponse(
                tool_calls=(
                    FurnaceMindGraphToolCall(
                        name="data_summary",
                        arguments={},
                        call_id="call_failure",
                    ),
                )
            ),
            FurnaceMindGraphModelResponse(content="The tool is unavailable."),
        ]
    )

    def execute(
        call: FurnaceMindGraphToolCall,
    ) -> FurnaceMindGraphToolResult:  # noqa: ARG001
        """Raise an error containing a secret to verify graph redaction."""
        raise RuntimeError("database password=secret")

    runtime = FurnaceMindGraphRuntime(
        model=model,
        tool_executor=execute,
        settings=_settings(tmp_path),
    )

    result = runtime.invoke(messages=[{"role": "user", "content": "Use data"}])

    assert result["status"] == "completed"
    assert result["final_response"] == "The tool is unavailable."
    assert result["tool_results"] == [
        {
            "tool_call_id": "call_failure",
            "name": "data_summary",
            "status": "failed",
            "output": None,
            "error_code": "FURNACEMIND_GRAPH_TOOL_FAILED",
            "error_message": "FurnaceMind tool execution failed.",
        }
    ]
    serialized = json.dumps(result)
    assert "password" not in serialized
    assert "secret" not in serialized
    assert "password" not in caplog.text
    assert "secret" not in caplog.text


def test_graph_stops_before_exceeding_tool_call_limit(tmp_path):
    """A model response exceeding the call budget should execute no tools."""
    model = _SequenceModel(
        [
            FurnaceMindGraphModelResponse(
                tool_calls=(
                    FurnaceMindGraphToolCall(name="data_summary"),
                    FurnaceMindGraphToolCall(name="anomaly_summary"),
                )
            )
        ]
    )
    executed: list[FurnaceMindGraphToolCall] = []

    def execute(call: FurnaceMindGraphToolCall) -> FurnaceMindGraphToolResult:
        """Record any tool call that incorrectly passes the graph guard."""
        executed.append(call)
        return FurnaceMindGraphToolResult(output={})

    runtime = FurnaceMindGraphRuntime(
        model=model,
        tool_executor=execute,
        settings=_settings(tmp_path, furnacemind_max_tool_calls_per_run=1),
    )

    result = runtime.invoke(messages=[{"role": "user", "content": "Use tools"}])

    assert result["status"] == "limit_reached"
    assert result["stop_reason"] == "tool_call_limit"
    assert result["tool_call_count"] == 0
    assert executed == []
    assert result["warnings"][-1]["code"] == "FURNACEMIND_GRAPH_TOOL_CALL_LIMIT"


def test_graph_module_has_no_streamlit_dependency_and_env_limit_is_bounded(
    tmp_path,
    monkeypatch,
):
    """The backend graph should avoid Streamlit and clamp unsafe settings."""
    from apps.backend_api.app.services import furnacemind_graph_service as graph_module

    source = Path(graph_module.__file__).read_text(encoding="utf-8").lower()
    assert "import streamlit" not in source
    assert "from streamlit" not in source

    monkeypatch.setenv("EVONITH_FURNACEMIND_GRAPH_MAX_ITERATIONS", "0")
    settings = _settings(tmp_path)
    assert settings.furnacemind_graph_max_iterations == 1

    bounded_runtime = FurnaceMindGraphRuntime(
        model=_SequenceModel(
            [FurnaceMindGraphModelResponse(content="response too long")]
        ),
        settings=_settings(tmp_path, furnacemind_max_response_chars=8),
    )
    bounded = bounded_runtime.invoke(messages=[{"role": "user", "content": "answer"}])
    assert bounded["final_response"] == "response"
