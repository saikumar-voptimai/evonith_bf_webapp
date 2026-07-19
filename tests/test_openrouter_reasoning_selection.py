"""Tests for OpenRouter reasoning-level model routing."""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

sys.modules.pop("agents.llm.llm_client", None)
if "agents.llm" in sys.modules:
    vars(sys.modules["agents.llm"]).pop("llm_client", None)
llm_client = importlib.import_module("agents.llm.llm_client")
settings_module = importlib.import_module("utils.settings")
Settings = settings_module.Settings
OpenRouterLLMConfig = settings_module.OpenRouterLLMConfig
OpenRouterReasoningProfile = settings_module.OpenRouterReasoningProfile
normalize_openrouter_reasoning_level = (
    settings_module.normalize_openrouter_reasoning_level
)


class _Message:
    """Fake OpenAI SDK message object."""

    content = "ok"


class _Choice:
    """Fake OpenAI SDK choice wrapper."""

    def __init__(self) -> None:
        self.message = _Message()


class _Completion:
    """Fake OpenAI SDK completion with the model OpenRouter actually used."""

    def __init__(self, *, model: str) -> None:
        self.model = model
        self.choices = [_Choice()]


class _FakeCompletions:
    """Capture completion calls and return one configured response."""

    def __init__(self, completion: _Completion) -> None:
        self.completion = completion
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Completion:
        """Record request arguments and return the fake completion."""
        self.calls.append(kwargs)
        return self.completion


class _FailingCompletions:
    """Capture completion calls and raise one configured error."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Completion:
        """Record request arguments, then raise the configured exception."""
        self.calls.append(kwargs)
        raise self.exc


def _patch_openrouter_settings(monkeypatch, fake_completions: _FakeCompletions) -> None:
    """Install deterministic OpenRouter config and SDK fake for one test."""
    monkeypatch.setattr(
        llm_client,
        "OpenAI",
        lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=fake_completions)),
    )
    monkeypatch.setattr(
        llm_client.settings.llm,
        "openrouter",
        OpenRouterLLMConfig(
            api_key="test-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
            model_name="base/model",
            memory_compression_model_name="memory/model",
            reasoning_profiles={
                "Low": OpenRouterReasoningProfile(
                    level="Low",
                    model_name="low/model",
                    reasoning_effort="low",
                ),
                "Medium": OpenRouterReasoningProfile(
                    level="Medium",
                    model_name="medium/model",
                    reasoning_effort="medium",
                ),
                "High": OpenRouterReasoningProfile(
                    level="High",
                    model_name="high/model",
                    reasoning_effort="high",
                ),
            },
            default_reasoning_level="Medium",
            max_tokens=321,
        ),
    )


def test_reasoning_level_normalization_defaults_to_medium() -> None:
    """Missing or invalid UI values should resolve to the Medium profile."""
    assert normalize_openrouter_reasoning_level(None) == "Medium"
    assert normalize_openrouter_reasoning_level("unexpected") == "Medium"
    assert normalize_openrouter_reasoning_level("low") == "Low"
    assert normalize_openrouter_reasoning_level("fast") == "Low"
    assert normalize_openrouter_reasoning_level("HIGH") == "High"
    assert normalize_openrouter_reasoning_level("slow") == "High"


def test_default_reasoning_profiles_use_requested_models(monkeypatch) -> None:
    """Built-in profiles should use one configured model per reasoning level."""
    for env_name in (
        "OPENROUTER_LOW_MODEL",
        "OPENROUTER_LOW_REASONING_EFFORT",
        "OPENROUTER_FAST_MODEL",
        "OPENROUTER_FAST_REASONING_EFFORT",
        "OPENROUTER_MEDIUM_MODEL",
        "OPENROUTER_MEDIUM_REASONING_EFFORT",
        "OPENROUTER_HIGH_MODEL",
        "OPENROUTER_HIGH_REASONING_EFFORT",
        "OPENROUTER_SLOW_MODEL",
        "OPENROUTER_SLOW_REASONING_EFFORT",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "legacy/default-model")

    profiles = Settings._load_llm_settings().openrouter.reasoning_profiles

    assert profiles["Low"].model_name == "google/gemma-4-26b-a4b-it"
    assert profiles["Low"].reasoning_effort is None
    assert profiles["Medium"].model_name == "openai/gpt-5.4-nano"
    assert profiles["Medium"].reasoning_effort is None
    assert profiles["High"].model_name == "google/gemini-3.1-flash-lite-preview"
    assert profiles["High"].reasoning_effort is None


def test_default_profiles_do_not_send_reasoning_parameter(monkeypatch) -> None:
    """Default model profiles should not force reasoning effort onto every model."""
    for env_name in (
        "OPENROUTER_LOW_MODEL",
        "OPENROUTER_LOW_REASONING_EFFORT",
        "OPENROUTER_FAST_MODEL",
        "OPENROUTER_FAST_REASONING_EFFORT",
        "OPENROUTER_MEDIUM_MODEL",
        "OPENROUTER_MEDIUM_REASONING_EFFORT",
        "OPENROUTER_HIGH_MODEL",
        "OPENROUTER_HIGH_REASONING_EFFORT",
        "OPENROUTER_SLOW_MODEL",
        "OPENROUTER_SLOW_REASONING_EFFORT",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(
        llm_client,
        "OpenAI",
        lambda **_: SimpleNamespace(
            chat=SimpleNamespace(
                completions=_FakeCompletions(_Completion(model="google/gemma-4-31b-it"))
            )
        ),
    )
    monkeypatch.setattr(
        llm_client.settings.llm,
        "openrouter",
        Settings._load_llm_settings().openrouter,
    )

    client = llm_client.OpenRouterClient(reasoning_level="Low")
    client.chat_completions(messages=[{"role": "user", "content": "hello"}])

    request = client.client.chat.completions.calls[0]
    assert request["model"] == "google/gemma-4-26b-a4b-it"
    assert "extra_body" not in request
    assert client.usage_metadata()["reasoning_effort"] is None


def test_openrouter_client_sends_reasoning_profile_without_fallbacks(
    monkeypatch,
) -> None:
    """Explicit profile effort should be sent without fallback routing."""
    fake_completions = _FakeCompletions(_Completion(model="low/model"))
    _patch_openrouter_settings(monkeypatch, fake_completions)

    client = llm_client.OpenRouterClient(reasoning_level="Low")
    client.chat_completions(messages=[{"role": "user", "content": "hello"}])

    request = fake_completions.calls[0]
    assert request["model"] == "low/model"
    assert request["max_completion_tokens"] == 321
    assert request["extra_body"] == {
        "reasoning": {"effort": "low", "exclude": True},
    }
    assert client.usage_metadata() == {
        "reasoning_level": "Low",
        "reasoning_effort": "low",
        "primary_model": "low/model",
        "actual_model": "low/model",
        "model_status": "completed",
        "model_error": None,
    }


def test_openrouter_client_does_not_retry_busy_model_with_max_tokens(monkeypatch) -> None:
    """Busy model errors should fail once instead of retrying the same request."""
    fake_completions = _FailingCompletions(RuntimeError("provider busy"))
    _patch_openrouter_settings(monkeypatch, fake_completions)

    client = llm_client.OpenRouterClient(reasoning_level="Medium")

    try:
        client.chat_completions(messages=[{"role": "user", "content": "hello"}])
    except RuntimeError as exc:
        assert str(exc) == "provider busy"
    else:  # pragma: no cover - defensive assertion for fake client behavior
        raise AssertionError("busy model error should propagate")

    assert len(fake_completions.calls) == 1
    assert "max_completion_tokens" in fake_completions.calls[0]
    assert "max_tokens" not in fake_completions.calls[0]


def test_openrouter_client_records_busy_model_message(monkeypatch) -> None:
    """Unavailable selected models should produce a clear retry suggestion."""
    fake_completions = _FakeCompletions(_Completion(model="medium/model"))
    _patch_openrouter_settings(monkeypatch, fake_completions)

    client = llm_client.OpenRouterClient(reasoning_level="Medium")
    client.record_failure(RuntimeError("provider busy"))

    assert client.unavailable_message() == (
        "Medium effort level models are busy. Please try another mode."
    )
    assert client.usage_metadata() == {
        "reasoning_level": "Medium",
        "reasoning_effort": "medium",
        "primary_model": "medium/model",
        "actual_model": None,
        "model_status": "failed",
        "model_error": "RuntimeError: provider busy",
    }


def test_openrouter_client_defaults_invalid_level_to_medium(monkeypatch) -> None:
    """Invalid reasoning levels should use the configured Medium profile."""
    fake_completions = _FakeCompletions(_Completion(model="medium/model"))
    _patch_openrouter_settings(monkeypatch, fake_completions)

    client = llm_client.OpenRouterClient(reasoning_level="bad-value")
    client.chat_completions(messages=[{"role": "user", "content": "hello"}])

    request = fake_completions.calls[0]
    assert client.reasoning_level == "Medium"
    assert request["model"] == "medium/model"
    assert request["extra_body"]["reasoning"] == {
        "effort": "medium",
        "exclude": True,
    }


def test_explicit_model_override_bypasses_reasoning_profiles(monkeypatch) -> None:
    """Specialized background jobs can still force one model exactly."""
    fake_completions = _FakeCompletions(_Completion(model="memory/model"))
    _patch_openrouter_settings(monkeypatch, fake_completions)

    client = llm_client.OpenRouterClient(model_name="memory/model")
    client.generate(system_prompt="system", user_prompt="user")

    request = fake_completions.calls[0]
    assert request["model"] == "memory/model"
    assert "extra_body" not in request
    assert client.usage_metadata()["reasoning_level"] is None
