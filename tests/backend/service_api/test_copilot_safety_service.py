"""Tests for Copilot safety and redaction."""

from __future__ import annotations

from app.core.config import BackendSettings
from app.core.errors import ApiError
from app.services.copilot_safety_service import CopilotSafetyService


def test_copilot_safety_redacts_sensitive_fields_without_mutating_original():
    service = CopilotSafetyService(settings=BackendSettings(backend_env="test"))
    payload = {
        "password": "secret",
        "headers": {"Authorization": "Bearer token"},
        "nested": {"api_key": "key"},
        "safe": "value",
    }

    redacted = service.redact(payload)

    assert redacted["password"] == "[REDACTED]"
    assert redacted["headers"]["Authorization"] == "[REDACTED]"
    assert redacted["nested"]["api_key"] == "[REDACTED]"
    assert redacted["safe"] == "value"
    assert payload["password"] == "secret"


def test_copilot_safety_caps_rows_and_prompt_chars():
    service = CopilotSafetyService(
        settings=BackendSettings(
            backend_env="test",
            copilot_max_context_rows=2,
            copilot_max_prompt_chars=5,
        )
    )

    rows, row_warnings, truncated = service.cap_rows([{"x": 1}, {"x": 2}, {"x": 3}])
    prompt, prompt_warnings = service.enforce_prompt_limit("abcdefgh")

    assert rows == [{"x": 1}, {"x": 2}]
    assert truncated is True
    assert row_warnings[0]["code"] == "COPILOT_CONTEXT_TOO_LARGE"
    assert prompt == "abcde"
    assert prompt_warnings[0]["code"] == "COPILOT_PROMPT_TOO_LARGE"


def test_copilot_safety_blocks_raw_data_when_disabled():
    service = CopilotSafetyService(
        settings=BackendSettings(
            backend_env="test",
            copilot_allow_raw_data_to_llm=False,
        )
    )

    try:
        service.ensure_raw_data_allowed(requested=True)
    except ApiError as exc:
        assert exc.code == "COPILOT_RAW_DATA_NOT_ALLOWED"
    else:
        raise AssertionError("Expected raw data request to be blocked")
