"""Focused tests for the constrained Phase 5 FurnaceMind job executor."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from agents.furnacemind.runtime_state import get_runtime_state
from data.scheduled_tasks import ScheduledJobView
from utils.scheduled_tasks.furnacemind_executor import (
    PRODUCTION_EXECUTOR_CAPABILITY_READY,
    FurnaceMindScheduledTaskExecutor,
    ScheduledTaskDataError,
    ScheduledTaskModelOutputError,
    ScheduledTaskProductionValidationError,
    validate_production_definition,
)
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)

_SCHEDULED_FOR = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)


def _definition(**overrides: object) -> dict[str, object]:
    """Build a valid hourly report definition with optional field overrides."""

    values: dict[str, object] = {
        "name": "Hourly BF2 review",
        "instructions": "Summarize the BF2 readings for the operator.",
        "furnace": "BF2",
        "data_period": "last_24_hours",
        "output_format": "operator_summary",
        "schedule_kind": "hourly",
        "delivery_channel": "in_app",
        "job_type": "furnace_summary",
        "analysis_level": "low",
        "data_source": "online_process_data",
        "hourly_minute": 0,
    }
    values.update(overrides)
    return build_task_definition(
        ScheduledTaskInput(**values),  # type: ignore[arg-type]
        generated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def _eta_definition(**overrides: object) -> dict[str, object]:
    """Build a valid deterministic ETA CO task definition."""

    values: dict[str, object] = {
        "name": "ETA CO check",
        "instructions": "Report ETA CO values and threshold status.",
        "furnace": "BF2",
        "data_period": "",
        "output_format": "exception_alert",
        "schedule_kind": "hourly",
        "delivery_channel": "in_app",
        "job_type": "eta_co_report",
        "analysis_level": "none",
        "data_source": "online_process_data",
        "hourly_minute": 0,
        "report_duration_minutes": 60,
        "aggregation_interval": "5min",
        "warning_threshold": 42.0,
        "critical_threshold": 40.0,
        "include_graph": False,
        "include_ai_summary": False,
    }
    values.update(overrides)
    return build_task_definition(
        ScheduledTaskInput(**values),  # type: ignore[arg-type]
        generated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def _job(definition: dict[str, object]) -> ScheduledJobView:
    """Wrap a definition in the persistence-independent job view."""

    return ScheduledJobView(
        job_id="00000000-0000-0000-0000-000000000100",
        job_name=str(definition["job_name"]),
        schema_version="scheduled-job-definition/v1",
        definition=definition,
        status="active",
        is_active=True,
        created_by_user_id="00000000-0000-0000-0000-000000000001",
        created_by_username="operator.test",
        target_device_id="bf2-jetson-01",
        timer_unit_name="furnacemind-job@test.timer",
        provisioning_error=None,
        activated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def _tool_schemas() -> list[dict[str, Any]]:
    """Return fetch and prohibited tool schemas for allowlist tests."""

    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_online_data",
                "description": "online",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_ml_data",
                "description": "historical",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python_plot",
                "description": "prohibited",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_docs",
                "description": "prohibited",
                "parameters": {"type": "object"},
            },
        },
    ]


class _FakeLLM:
    """Expose only safe model metadata used by the executor."""

    primary_model = "vendor/primary-model"
    last_actual_model = "vendor/actual-model"
    reasoning_effort = "medium"

    def usage_metadata(self) -> dict[str, object]:
        """Return metadata containing one deliberately sensitive ignored field."""

        return {
            "reasoning_level": "Low",
            "reasoning_effort": self.reasoning_effort,
            "primary_model": self.primary_model,
            "actual_model": self.last_actual_model,
            "model_status": "completed",
            "model_error": "token=must-not-persist",
            "raw_prompt": "must-not-persist",
        }


def _online_frame() -> pd.DataFrame:
    """Return data spanning both sides of the planned 24-hour window."""

    return pd.DataFrame(
        {
            "body_etaco": [38.0, 39.0, 41.0, 43.0, 99.0],
            "fuel_rate": [500.0, 490.0, 480.0, 470.0, 1.0],
        },
        index=pd.DatetimeIndex(
            [
                "2026-09-04T09:59:00Z",
                "2026-09-04T10:00:00Z",
                "2026-09-05T09:00:00Z",
                "2026-09-05T09:59:00Z",
                "2026-09-05T10:00:00Z",
            ]
        ),
    )


def test_capability_marker_and_production_validation_fail_closed() -> None:
    """The adapter should advertise readiness but reject unimplemented outputs."""

    definition = _definition()
    definition["delivery"] = {
        "channel": "email",
        "notify_on_failure": True,
        "recipients": ["operator@example.com"],
        "subject": "Report",
        "attachments": [],
    }

    errors = validate_production_definition(definition)

    assert PRODUCTION_EXECUTOR_CAPABILITY_READY is True
    assert "delivery.channel: only in_app delivery is implemented" in errors

    graph_errors = validate_production_definition(
        _eta_definition(
            analysis_level="low",
            include_ai_summary=True,
            include_graph=True,
        )
    )
    assert (
        "inputs.include_graph: scheduled graph artifacts are not implemented"
        in graph_errors
    )

    historical_eta_errors = validate_production_definition(
        _eta_definition(data_source="historical_furnace_data")
    )
    assert any(
        "ETA CO reports require online_process_data" in error
        for error in historical_eta_errors
    )


def test_validation_rejects_an_unbound_data_source() -> None:
    """A schema-valid new source must not silently gain production access."""

    definition = _definition()
    definition["inputs"]["data_source"] = "future_write_capable_source"  # type: ignore[index]

    with pytest.raises(ScheduledTaskProductionValidationError, match="read-only"):
        FurnaceMindScheduledTaskExecutor(
            tool_schema_provider=_tool_schemas,
        ).execute(job=_job(definition), scheduled_for=_SCHEDULED_FOR)


def test_llm_route_maps_reasoning_and_overrides_model_window_arguments() -> None:
    """Low analysis should use Low reasoning and immutable online fetch times."""

    reasoning_levels: list[str] = []
    dispatches: list[tuple[str, dict[str, Any]]] = []
    graph_calls: list[dict[str, Any]] = []

    def _llm_factory(reasoning_level: str) -> _FakeLLM:
        """Record the explicit analysis-to-reasoning mapping."""

        reasoning_levels.append(reasoning_level)
        return _FakeLLM()

    def _dispatch(*, name: str, arguments: dict[str, Any]) -> str:
        """Record trusted arguments and publish a run-local DataFrame."""

        dispatches.append((name, arguments))
        get_runtime_state()["fm_df"] = _online_frame()
        return "ONLINE DATA: fetched sensitive values"

    def _run_graph(**kwargs: Any) -> Any:
        """Simulate a model trying to widen the planned fetch window."""

        graph_calls.append(kwargs)
        result = kwargs["tool_dispatcher"](
            name="fetch_online_data",
            arguments={
                "lookback": "90d",
                "start_time_utc": "1900-01-01T00:00:00Z",
                "measurement_groups": ["miscellaneous"],
            },
        )
        return SimpleNamespace(
            final_response="BF2 remained stable in the planned window.",
            tool_events=(
                {
                    "name": "fetch_online_data",
                    "succeeded": True,
                    "result_characters": len(result),
                },
            ),
            iterations=1,
        )

    output = FurnaceMindScheduledTaskExecutor(
        llm_factory=_llm_factory,
        graph_runner=_run_graph,
        tool_schema_provider=_tool_schemas,
        tool_dispatcher=_dispatch,
    ).execute(job=_job(_definition()), scheduled_for=_SCHEDULED_FOR)

    assert reasoning_levels == ["Low"]
    assert dispatches == [
        (
            "fetch_online_data",
            {
                "start_time_utc": "2026-09-04T10:00:00Z",
                "end_time_utc": "2026-09-05T10:00:00Z",
            },
        )
    ]
    offered_tools = graph_calls[0]["tools"]
    assert [item["function"]["name"] for item in offered_tools] == ["fetch_online_data"]
    assert offered_tools[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert graph_calls[0]["allowed_tool_names"] == {"fetch_online_data"}
    assert graph_calls[0]["fail_on_tool_error"] is True
    assert output.content == "BF2 remained stable in the planned window."
    assert output.metadata["model"] == {
        "reasoning_level": "Low",
        "reasoning_effort": "medium",
        "primary_model": "vendor/primary-model",
        "actual_model": "vendor/actual-model",
        "model_status": "completed",
    }
    assert "must-not-persist" not in repr(output.metadata)
    assert "1900-01-01" not in repr(output.metadata)
    assert "sensitive values" not in repr(output.metadata)


@pytest.mark.parametrize(
    ("analysis_level", "expected_reasoning"),
    (("medium", "Medium"), ("high", "High")),
)
def test_historical_llm_route_maps_reasoning_and_uses_exact_times(
    analysis_level: str,
    expected_reasoning: str,
) -> None:
    """Historical runs should explicitly map reasoning and bind UTC endpoints."""

    reasoning_levels: list[str] = []
    dispatches: list[tuple[str, dict[str, Any]]] = []

    def _dispatch(*, name: str, arguments: dict[str, Any]) -> str:
        """Publish IST-naive historical rows within the planned UTC window."""

        dispatches.append((name, arguments))
        get_runtime_state()["fm_df"] = pd.DataFrame(
            {"FURNACETOPGASANALYSISCO2ETACO": [41.0, 42.0]},
            index=pd.DatetimeIndex(["2026-09-04T15:30:00", "2026-09-05T15:00:00"]),
        )
        return "ML STATIC DATA: fetched"

    def _run_graph(**kwargs: Any) -> Any:
        """Execute the single bound historical tool and return model text."""

        result = kwargs["tool_dispatcher"](
            name="fetch_ml_data",
            arguments={"start_time": "2000-01-01"},
        )
        return SimpleNamespace(
            final_response="Historical summary.",
            tool_events=(
                {
                    "name": "fetch_ml_data",
                    "succeeded": True,
                    "result_characters": len(result),
                },
            ),
            iterations=1,
        )

    executor = FurnaceMindScheduledTaskExecutor(
        llm_factory=lambda level: reasoning_levels.append(level) or _FakeLLM(),
        graph_runner=_run_graph,
        tool_schema_provider=_tool_schemas,
        tool_dispatcher=_dispatch,
    )
    executor.execute(
        job=_job(
            _definition(
                analysis_level=analysis_level,
                data_source="historical_furnace_data",
            )
        ),
        scheduled_for=_SCHEDULED_FOR,
    )

    assert reasoning_levels == [expected_reasoning]
    assert dispatches == [
        (
            "fetch_ml_data",
            {
                "start_time": "2026-09-04T10:00:00Z",
                "end_time": "2026-09-05T10:00:00Z",
            },
        )
    ]


@pytest.mark.parametrize("analysis_level", ("none", "low"))
def test_disabled_ai_summary_is_truly_no_llm_and_returns_eta_statistics(
    analysis_level: str,
) -> None:
    """ETA jobs with AI disabled should fetch once and summarize directly."""

    calls: list[tuple[str, dict[str, Any]]] = []

    def _unexpected_llm(_reasoning_level: str) -> _FakeLLM:
        """Fail if the deterministic path attempts to construct an LLM."""

        raise AssertionError("analysis_level=none must not construct an LLM")

    def _unexpected_graph(**_kwargs: Any) -> Any:
        """Fail if the deterministic path attempts to invoke LangGraph."""

        raise AssertionError("analysis_level=none must not invoke the graph")

    def _dispatch(*, name: str, arguments: dict[str, Any]) -> str:
        """Publish rows before, within, and at the exclusive window end."""

        calls.append((name, arguments))
        assert get_runtime_state()["fm_disable_tool_error_file"] is True
        get_runtime_state()["fm_df"] = pd.DataFrame(
            {
                "body_etaco": [99.0, 39.0, 41.0, 43.0, 99.0],
                "fuel_rate": [1.0, 490.0, 480.0, 470.0, 1.0],
            },
            index=pd.DatetimeIndex(
                [
                    "2026-09-05T08:59:00Z",
                    "2026-09-05T09:00:00Z",
                    "2026-09-05T09:30:00Z",
                    "2026-09-05T09:59:00Z",
                    "2026-09-05T10:00:00Z",
                ]
            ),
        )
        return "ONLINE DATA: fetched"

    output = FurnaceMindScheduledTaskExecutor(
        llm_factory=_unexpected_llm,
        graph_runner=_unexpected_graph,
        tool_schema_provider=_tool_schemas,
        tool_dispatcher=_dispatch,
    ).execute(
        job=_job(
            _eta_definition(
                analysis_level=analysis_level,
                include_ai_summary=False,
            )
        ),
        scheduled_for=_SCHEDULED_FOR,
    )

    assert calls == [
        (
            "fetch_online_data",
            {
                "start_time_utc": "2026-09-05T09:00:00Z",
                "end_time_utc": "2026-09-05T10:00:00Z",
                "window": "5 minutes",
            },
        )
    ]
    assert output.metadata["analysis_level"] == analysis_level
    assert output.metadata["model"] == {"used": False}
    assert output.content_json is not None
    assert output.content_json["row_count"] == 3
    assert output.content_json["eta_co"] == {
        "signal": "body_etaco",
        "count": 3,
        "minimum": 39.0,
        "maximum": 43.0,
        "mean": 41.0,
        "latest": 43.0,
        "warning_threshold": 42.0,
        "critical_threshold": 40.0,
        "readings_below_warning": 2,
        "readings_below_critical": 1,
        "latest_status": "normal",
    }
    assert "99.0" not in (output.content or "")


@pytest.mark.parametrize(
    "tool_result",
    (
        "",
        "Fetch Error: historian unavailable",
        "ONLINE DATA: No data found.",
        "ML STATIC DATA\nGAP NOTE: missing recent rows",
    ),
)
def test_tool_error_strings_are_terminal_failures(tool_result: str) -> None:
    """Error-like fetch strings must not become successful scheduled reports."""

    def _dispatch(**_kwargs: Any) -> str:
        """Return the parameterized failed tool result."""

        get_runtime_state()["fm_df"] = _online_frame()
        return tool_result

    with pytest.raises(ScheduledTaskDataError, match="did not succeed"):
        FurnaceMindScheduledTaskExecutor(
            tool_schema_provider=_tool_schemas,
            tool_dispatcher=_dispatch,
        ).execute(
            job=_job(_definition(analysis_level="none")), scheduled_for=_SCHEDULED_FOR
        )


def test_empty_dataframe_is_a_terminal_fetch_failure() -> None:
    """A success-looking tool string cannot hide an empty dataset."""

    def _dispatch(**_kwargs: Any) -> str:
        """Publish an empty frame while returning a success-looking message."""

        get_runtime_state()["fm_df"] = pd.DataFrame()
        return "ONLINE DATA: fetched"

    with pytest.raises(ScheduledTaskDataError, match="no rows"):
        FurnaceMindScheduledTaskExecutor(
            tool_schema_provider=_tool_schemas,
            tool_dispatcher=_dispatch,
        ).execute(
            job=_job(_definition(analysis_level="none")), scheduled_for=_SCHEDULED_FOR
        )


@pytest.mark.parametrize("final_response", ("", "   "))
def test_llm_route_requires_a_successful_fetch_and_nonempty_text(
    final_response: str,
) -> None:
    """LLM output must be grounded by the actual bound dispatcher and have text."""

    def _dispatch(**_kwargs: Any) -> str:
        """Publish a valid dataset for the graph stub."""

        get_runtime_state()["fm_df"] = _online_frame()
        return "ONLINE DATA: fetched"

    def _run_graph(**kwargs: Any) -> Any:
        """Run the required fetch, then return the parameterized empty text."""

        result = kwargs["tool_dispatcher"](
            name="fetch_online_data",
            arguments={},
        )
        return SimpleNamespace(
            final_response=final_response,
            tool_events=(
                {
                    "name": "fetch_online_data",
                    "succeeded": True,
                    "result_characters": len(result),
                },
            ),
            iterations=1,
        )

    with pytest.raises(ScheduledTaskModelOutputError, match="no report text"):
        FurnaceMindScheduledTaskExecutor(
            llm_factory=lambda _level: _FakeLLM(),
            graph_runner=_run_graph,
            tool_schema_provider=_tool_schemas,
            tool_dispatcher=_dispatch,
        ).execute(job=_job(_definition()), scheduled_for=_SCHEDULED_FOR)


def test_llm_route_rejects_a_report_without_an_actual_tool_dispatch() -> None:
    """Synthetic event metadata cannot substitute for the required data fetch."""

    def _run_graph(**_kwargs: Any) -> Any:
        """Return a fake success event without invoking the dispatcher."""

        return SimpleNamespace(
            final_response="Ungrounded report.",
            tool_events=(
                {
                    "name": "fetch_online_data",
                    "succeeded": True,
                    "result_characters": 100,
                },
            ),
            iterations=1,
        )

    with pytest.raises(ScheduledTaskModelOutputError, match="required data fetch"):
        FurnaceMindScheduledTaskExecutor(
            llm_factory=lambda _level: _FakeLLM(),
            graph_runner=_run_graph,
            tool_schema_provider=_tool_schemas,
        ).execute(job=_job(_definition()), scheduled_for=_SCHEDULED_FOR)


def test_activity_callback_can_be_replaced_for_runner_lease_checks() -> None:
    """The runner should be able to inject a fresh lease callback per attempt."""

    activity: list[str] = []

    def _dispatch(**_kwargs: Any) -> str:
        """Publish a valid deterministic dataset."""

        get_runtime_state()["fm_df"] = _online_frame()
        return "ONLINE DATA: fetched"

    executor = FurnaceMindScheduledTaskExecutor(
        tool_schema_provider=_tool_schemas,
        tool_dispatcher=_dispatch,
    )
    executor.set_activity_callback(lambda: activity.append("checked"))

    executor.execute(
        job=_job(_definition(analysis_level="none")),
        scheduled_for=_SCHEDULED_FOR,
    )

    assert activity == ["checked", "checked"]
