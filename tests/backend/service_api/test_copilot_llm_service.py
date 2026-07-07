"""Tests for Copilot LLM abstraction."""

from __future__ import annotations

import sys

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.copilot_llm_service import CopilotLLMService


def test_copilot_llm_disabled_returns_structured_error():
    service = CopilotLLMService(
        BackendSettings(backend_env="test", copilot_llm_enabled=False)
    )

    try:
        service.generate("prompt")
    except ApiError as exc:
        assert exc.code == "COPILOT_LLM_DISABLED"
    else:
        raise AssertionError("Expected disabled LLM error")


def test_copilot_llm_mock_provider_is_deterministic():
    service = CopilotLLMService(
        BackendSettings(
            backend_env="test",
            copilot_llm_enabled=True,
            copilot_enable_provider_calls=True,
            copilot_provider="mock",
            copilot_model="mock-model",
        )
    )

    result = service.generate("prompt")

    assert result.provider_name == "mock"
    assert result.model_name == "mock-model"
    assert "Mock Copilot analysis" in result.text


def test_copilot_llm_timeout_and_invalid_response_are_structured():
    service = CopilotLLMService(
        BackendSettings(
            backend_env="test",
            copilot_llm_enabled=True,
            copilot_enable_provider_calls=True,
            copilot_provider="mock",
        )
    )

    for option, code in [
        ("simulate_timeout", "COPILOT_LLM_TIMEOUT"),
        ("simulate_invalid_response", "COPILOT_LLM_RESPONSE_INVALID"),
    ]:
        try:
            service.generate("prompt", options={option: True})
        except ApiError as exc:
            assert exc.code == code
        else:
            raise AssertionError(f"Expected {code}")


def test_copilot_llm_does_not_import_openai_for_disabled_or_mock_modes():
    sys.modules.pop("openai", None)
    service = CopilotLLMService(
        BackendSettings(
            backend_env="test",
            copilot_llm_enabled=True,
            copilot_enable_provider_calls=True,
            copilot_provider="mock",
        )
    )

    service.generate("prompt")

    assert "openai" not in sys.modules
