"""Optional provider abstraction for Copilot LLM calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError


@dataclass(frozen=True)
class CopilotLLMResult:
    text: str
    provider_name: str | None
    model_name: str | None


class CopilotLLMService:
    """Small provider-agnostic LLM service with disabled/mock modes."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def is_available(self) -> bool:
        if not self.settings.copilot_llm_enabled or not self.settings.copilot_enable_provider_calls:
            return False
        provider = self.settings.copilot_provider
        if provider == "mock":
            return True
        return bool(provider and os.getenv(self.settings.copilot_api_key_env, ""))

    def generate(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> CopilotLLMResult:
        options = options or {}
        if options.get("simulate_timeout"):
            raise ApiError("COPILOT_LLM_TIMEOUT", "Copilot LLM provider timed out.", status_code=504)
        if options.get("simulate_invalid_response"):
            raise ApiError("COPILOT_LLM_RESPONSE_INVALID", "Copilot LLM provider returned an invalid response.", status_code=502)
        if not self.settings.copilot_llm_enabled or not self.settings.copilot_enable_provider_calls:
            raise ApiError("COPILOT_LLM_DISABLED", "Copilot LLM provider calls are disabled.", status_code=409)

        provider = self.settings.copilot_provider
        model = self.settings.copilot_model or None
        if provider == "mock":
            return CopilotLLMResult(
                text=f"Mock Copilot analysis complete. Context characters: {len(prompt)}.",
                provider_name="mock",
                model_name=model or "mock-copilot",
            )
        if not provider or not os.getenv(self.settings.copilot_api_key_env, ""):
            raise ApiError("COPILOT_LLM_PROVIDER_NOT_CONFIGURED", "Copilot LLM provider is not configured.", status_code=409)
        if provider != "openai":
            raise ApiError("COPILOT_LLM_PROVIDER_UNAVAILABLE", "Configured Copilot LLM provider is unavailable.", status_code=503)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv(self.settings.copilot_api_key_env))
            completion = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=self.settings.copilot_timeout_seconds,
            )
            text = completion.choices[0].message.content or ""
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError("COPILOT_LLM_INTERNAL_ERROR", "Copilot LLM provider failed.", status_code=502) from exc
        if not text:
            raise ApiError("COPILOT_LLM_RESPONSE_INVALID", "Copilot LLM provider returned no text.", status_code=502)
        return CopilotLLMResult(
            text=text[: self.settings.copilot_max_output_chars],
            provider_name=provider,
            model_name=model or "gpt-4o-mini",
        )
