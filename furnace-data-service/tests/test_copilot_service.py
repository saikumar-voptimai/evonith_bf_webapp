"""Tests for Copilot orchestration service."""

from __future__ import annotations

from app.core.config import BackendSettings
from app.core.errors import ApiError
from app.services.copilot_service import CopilotService


def _payload(**overrides):
    payload = {
        "question": "What is happening?",
        "input_data": [{"temp": 100.0}, {"temp": 140.0}],
        "include_recent_data": True,
        "include_anomaly": True,
        "allow_llm": False,
        "options": {},
    }
    payload.update(overrides)
    return payload


def test_copilot_service_config_recent_data_anomaly_and_fallback():
    service = CopilotService(settings=BackendSettings(backend_env="test"))

    config = service.get_config()
    recent = service.get_recent_data({"source": "input_data", "filters": {"rows": [{"x": 1}]}})
    anomaly = service.analyze_anomaly({"input_data": [{"x": 1}, {"x": 5}]})
    analysis = service.analyze_question(_payload())

    assert config["llm_enabled"] is False
    assert recent["returned_rows"] == 1
    assert anomaly["signals"]
    assert analysis["llm_used"] is False
    assert analysis["evidence"]


def test_copilot_service_uses_mock_llm_when_explicitly_enabled():
    service = CopilotService(
        settings=BackendSettings(
            backend_env="test",
            copilot_llm_enabled=True,
            copilot_enable_provider_calls=True,
            copilot_provider="mock",
        )
    )

    result = service.analyze_question(_payload(allow_llm=True))

    assert result["llm_used"] is True
    assert result["provider_name"] == "mock"


def test_copilot_service_job_artifact_and_missing_job(tmp_path, monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    service = CopilotService(
        settings=BackendSettings(
            backend_env="test",
            copilot_job_threshold_rows=1,
        )
    )

    result = service.analyze_question(_payload(input_data=[{"x": 1}, {"x": 2}]))
    job = service.start_analysis_job(_payload())
    status = service.get_job_status(job.job_id)

    assert result["artifacts"][0]["artifact_id"]
    assert status["status"] == "completed"
    try:
        service.get_job_status("missing")
    except ApiError as exc:
        assert exc.code == "COPILOT_JOB_NOT_FOUND"
    else:
        raise AssertionError("Expected missing job error")
