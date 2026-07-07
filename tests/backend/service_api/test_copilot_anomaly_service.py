"""Tests for Copilot anomaly service."""

from __future__ import annotations

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.copilot_anomaly_service import CopilotAnomalyService


def test_copilot_anomaly_returns_stable_signals():
    service = CopilotAnomalyService(settings=BackendSettings(backend_env="test"))

    result = service.analyze(
        {
            "input_data": [
                {"temp": 100.0, "pressure": 1.0},
                {"temp": 130.0, "pressure": 1.1},
                {"temp": 190.0, "pressure": 1.2},
            ]
        }
    )

    assert result["signals"]
    assert result["severity"] in {"low", "medium", "high", "critical"}
    assert result["tables"]["signals"]["returned_rows"] >= 1


def test_copilot_anomaly_empty_and_invalid_inputs_are_structured():
    service = CopilotAnomalyService(settings=BackendSettings(backend_env="test"))

    for payload, code in [
        ({"input_data": []}, "COPILOT_ANOMALY_DATA_EMPTY"),
        ({"input_data": ["bad"]}, "COPILOT_ANOMALY_INPUT_INVALID"),
    ]:
        try:
            service.analyze(payload)
        except ApiError as exc:
            assert exc.code == code
        else:
            raise AssertionError(f"Expected {code}")
