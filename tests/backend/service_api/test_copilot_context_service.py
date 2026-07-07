"""Tests for Copilot context and prompt services."""

from __future__ import annotations

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.services.copilot_context_service import CopilotContextService
from apps.backend_api.app.services.copilot_prompt_service import CopilotPromptService


def test_copilot_context_excludes_raw_rows_by_default_and_redacts():
    settings = BackendSettings(
        backend_env="test",
        copilot_allow_raw_data_to_llm=False,
    )
    service = CopilotContextService(settings=settings)

    context = service.build_context(
        question="What changed?",
        data={"summary": {"row_count": 1}, "rows": [{"api_key": "secret"}], "columns": []},
        anomaly={"summary": {"signal_count": 1}, "signals": [{"name": "temp"}]},
        analysis_mode="summary",
    )

    assert context["sample_rows"] == []
    assert context["warnings"][0]["code"] == "COPILOT_RAW_DATA_NOT_ALLOWED"


def test_copilot_prompt_is_redacted_and_capped():
    settings = BackendSettings(
        backend_env="test",
        copilot_max_prompt_chars=80,
    )
    service = CopilotPromptService(settings=settings)

    prompt, warnings = service.build_prompt({"question": "x", "password": "secret", "blob": "z" * 200})

    assert len(prompt) == 80
    assert "secret" not in prompt
    assert warnings[0]["code"] == "COPILOT_PROMPT_TOO_LARGE"
