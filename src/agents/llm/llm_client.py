"""LLM inference clients for BF2 FurnaceMind and AI Copilot features.

Provides two concrete client classes:

* :class:`OpenRouterClient` – used by the FurnaceMind agent; routes all
  traffic through OpenRouter so model switches are config-only changes.
* :class:`OpenAIClient` – used by AI Copilot and V-OptimAIse; calls the
  OpenAI Chat Completions (or Responses) API directly.

Use :func:`get_llm_client` to obtain the default client.
"""

# FurnaceMind/llm/llm.py
# Purpose: LLM inference clients (text generation only)

from __future__ import annotations

import io as _io
import logging
import os as _os
from typing import Any, List, Literal, Optional

from openai import OpenAI

from utils.settings import normalize_openrouter_reasoning_level, settings

Provider = Literal["openrouter", "openai"]
ApiMode = Literal["responses", "chat_completions"]


# OPENROUTER CLIENT
class OpenRouterClient:
    """OpenRouter chat client with optional reasoning-profile routing.

    Normal FurnaceMind chat turns resolve a user-facing reasoning level
    (``Low``, ``Medium``, or ``High``) into one configured model and optional
    OpenRouter reasoning effort. Background jobs can still pass ``model_name``
    to force a specific model and bypass the user-facing routing profiles.
    """

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        *,
        model_name: str | None = None,
        reasoning_level: str | None = None,
    ) -> None:
        """Initialize an OpenRouter chat client from application settings.

        Args:
            model_name: Optional model override for specialized LLM tasks such
                as memory compression. Overrides do not receive UI reasoning
                effort routing.
            reasoning_level: Optional user-selected reasoning level. Missing or
                invalid values resolve to the configured Medium profile.

        Raises:
            ValueError: If ``OPENROUTER_API_KEY`` is not configured.
        """
        cfg = settings.llm.openrouter
        if not cfg.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        if model_name:
            self.reasoning_level: str | None = None
            self.reasoning_effort: str | None = None
            self.model = model_name.strip()
        else:
            self.reasoning_level = normalize_openrouter_reasoning_level(
                reasoning_level or cfg.default_reasoning_level
            )
            profile = cfg.reasoning_profiles.get(self.reasoning_level)
            if profile is None:
                self.reasoning_level = "Medium"
                profile = cfg.reasoning_profiles.get("Medium")
            self.model = (profile.model_name if profile else cfg.model_name).strip()
            self.reasoning_effort = (
                str(profile.reasoning_effort).strip()
                if profile and profile.reasoning_effort
                else None
            )

        self.primary_model = self.model
        self.max_tokens = cfg.max_tokens
        self.last_actual_model: str | None = None
        self.last_error: str | None = None

        self.extra_headers = {
            "HTTP-Referer": settings.app.environment,
            "X-Title": "FurnaceMind",
        }

    def _extra_body(self) -> dict[str, Any]:
        """Build OpenRouter-only request options for reasoning effort.

        The OpenAI-compatible SDK accepts these fields through ``extra_body``.
        Reasoning effort is included only when the selected profile configures
        it.
        """
        extra_body: dict[str, Any] = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {
                "effort": self.reasoning_effort,
                "exclude": True,
            }
        return extra_body

    def _base_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        stop: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        """Return shared Chat Completions request arguments.

        Both plain text generation and tool-calling use the same model,
        messages, stop tokens, OpenRouter headers, and optional routing options.
        Keeping this in one place prevents profile handling from drifting across
        the two call paths.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stop": stop,
            "extra_headers": self.extra_headers,
        }
        extra_body = self._extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    @staticmethod
    def _should_retry_with_max_tokens(exc: Exception) -> bool:
        """Return True only for max-completion-token compatibility failures.

        OpenRouter can surface provider-specific parameter errors for older
        OpenAI-compatible models. Those should retry the same model with
        ``max_tokens``; availability, rate-limit, and provider-busy errors
        should fail immediately so the UI can ask the operator to try another
        reasoning mode.
        """
        return "max_completion_tokens" in f"{type(exc).__name__}: {exc}".lower()

    def _record_completion(self, completion: Any) -> Any:
        """Record which model actually answered an OpenRouter request.

        This method stores the returned model id and emits a structured log entry
        for audit/debugging, including the selected reasoning level and primary
        model configured for the turn.
        """
        actual_model = str(getattr(completion, "model", None) or self.primary_model)
        self.last_actual_model = actual_model
        self.last_error = None
        self._logger.info(
            "openrouter_completion",
            extra={
                "reasoning_level": self.reasoning_level,
                "reasoning_effort": self.reasoning_effort,
                "primary_model": self.primary_model,
                "actual_model": actual_model,
            },
        )
        return completion

    def record_failure(self, exc: Exception) -> None:
        """Record a failed OpenRouter request for chat metadata and logs.

        FurnaceMind no longer asks OpenRouter to try fallback models for a
        selected reasoning level. When the configured model is unavailable, the
        UI can call this method before showing a user-facing retry suggestion.
        """
        self.last_actual_model = None
        self.last_error = f"{type(exc).__name__}: {exc}"
        self._logger.warning(
            "openrouter_completion_failed",
            extra={
                "reasoning_level": self.reasoning_level,
                "reasoning_effort": self.reasoning_effort,
                "primary_model": self.primary_model,
                "model_error": self.last_error,
            },
        )

    def unavailable_message(self) -> str:
        """Return the operator-facing message for an unavailable model profile."""
        if self.reasoning_level:
            return (
                f"{self.reasoning_level} effort level models are busy. "
                "Please try another mode."
            )
        return "The selected model is busy. Please try another mode."

    def usage_metadata(self) -> dict[str, Any]:
        """Return SQL-friendly metadata for the completed chat turn.

        Callers persist this beside the assistant message so a conversation row
        records the selected reasoning level, primary model, actual model used
        by OpenRouter, and whether the model call completed or failed.
        """
        return {
            "reasoning_level": self.reasoning_level,
            "reasoning_effort": self.reasoning_effort,
            "primary_model": self.primary_model,
            "actual_model": None
            if self.last_error
            else self.last_actual_model or self.primary_model,
            "model_status": "failed" if self.last_error else "completed",
            "model_error": self.last_error,
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate a response from the LLM given system and user prompts.

        Tries ``max_completion_tokens`` first and retries the same model with
        ``max_tokens`` for older OpenAI-compatible models. Reasoning is sent
        only when explicitly configured for the selected profile.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = self._base_kwargs(messages=messages, stop=stop)
        try:
            completion = self.client.chat.completions.create(
                **kwargs,
                max_completion_tokens=self.max_tokens,
            )
        except Exception as exc:
            if not self._should_retry_with_max_tokens(exc):
                raise
            # Compatibility retry for older models that only accept max_tokens.
            completion = self.client.chat.completions.create(
                **kwargs,
                max_tokens=self.max_tokens,
            )
        self._record_completion(completion)
        return completion.choices[0].message.content or ""

    def chat_completions(
        self,
        *,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[Any] = "auto",
        stop: Optional[List[str]] = None,
    ) -> Any:
        """Call OpenRouter Chat Completions with optional tool-calling.

        Returns the raw OpenAI-compatible completion object. The client records
        the actual model returned by OpenRouter so callers can persist which
        configured model answered the turn.
        """

        kwargs = self._base_kwargs(messages=messages, stop=stop)
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        # Prefer max_completion_tokens (newer OpenAI-compatible APIs). Retry the
        # same model with max_tokens for older providers.
        try:
            completion = self.client.chat.completions.create(
                **kwargs,
                max_completion_tokens=self.max_tokens,
            )
        except Exception as exc:
            if not self._should_retry_with_max_tokens(exc):
                raise
            completion = self.client.chat.completions.create(
                **kwargs,
                max_tokens=self.max_tokens,
            )
        return self._record_completion(completion)


# OPENAI CLIENT
class OpenAIClient:
    """Wrapper around the OpenAI Chat Completions and Responses APIs.

    Supports both ``chat_completions`` and ``responses`` API modes
    (controlled by the ``OPENAI_API_MODE`` environment variable).  Contains
    robust text extraction logic for structured GPT-5-mini outputs.

    Attributes:
        model:      OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        max_tokens: Maximum tokens for each completion response.
        api_mode:   Active API mode: ``"responses"`` or ``"chat_completions"``.
    """

    def __init__(self) -> None:
        """Initialise the OpenAI client from :data:`~utils.settings.settings`.

        Raises:
            ValueError: If ``OPENAI_API_KEY`` is not set.
        """
        cfg = settings.llm.openai
        if not cfg.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = (
            OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            if cfg.base_url
            else OpenAI(api_key=cfg.api_key)
        )

        self.model = cfg.model_name
        self.max_tokens = cfg.max_tokens
        self.api_mode: ApiMode = cfg.api_mode

        self.extra_headers = {
            "X-Title": "FurnaceMind",
            "HTTP-Referer": settings.app.environment,
        }

    @staticmethod
    def _extract_text_from_responses(resp: Any) -> str:
        """
        Robust extraction of text from Responses API result.
        Necessary for GPT-5-mini and other structured outputs.
        """
        # 1) SDK convenience field
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # 2) Walk structured output
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t:
                        parts.append(t)

        return "".join(parts).strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate a response using the OpenAI Chat Completions API.

        Tries ``max_completion_tokens`` first and falls back to ``max_tokens``
        for compatibility with older models.

        Args:
            system_prompt: Instruction/persona prompt for the assistant.
            user_prompt:   User message to respond to.
            stop:          Optional list of stop sequences.

        Returns:
            The assistant’s response text, or an empty string on failure.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=self.max_tokens,
                stop=stop,
                extra_headers=self.extra_headers,
            )
        except Exception:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                stop=stop,
                extra_headers=self.extra_headers,
            )

        return completion.choices[0].message.content or ""

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> object:
        """
        Chat completion with optional tool definitions.
        Returns the raw ChatCompletion response message object
        (which may contain tool_calls).
        """
        kwargs = dict(
            model=self.model,
            messages=messages,
            extra_headers=self.extra_headers,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            kwargs["max_completion_tokens"] = self.max_tokens
            completion = self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = self.max_tokens
            completion = self.client.chat.completions.create(**kwargs)

        return completion.choices[0].message


# CLIENT SELECTION
def get_llm_client(prefer: Optional[Provider] = None):
    """Return the configured LLM client.

    FurnaceMind is intentionally routed through OpenRouter to make model switching
    a config-only change and avoid provider/endpoint mismatches.
    """

    _ = prefer  # kept for backward compatibility; intentionally unused
    return OpenRouterClient()


# ==========================================================
# 🔹 OPENAI RESPONSES API — code_interpreter helper
# ==========================================================

OPENAI_MODEL: str = _os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    files: list[tuple[str, bytes]] | None = None,
) -> str:
    """Send prompts to the OpenAI Responses API with optional file attachments.

    Uses the ``code_interpreter`` tool for data-heavy analysis tasks.  This is
    the single entry point for AI Copilot page interactions — no other module
    should instantiate a raw OpenAI client for this purpose.

    Parameters
    ----------
    system_prompt:  LLM persona / task framing.
    user_prompt:    The main user message (may contain pre-computed data).
    files:          Optional list of ``(filename, bytes)`` tuples to upload
                    and attach to the code_interpreter tool.

    Returns
    -------
    str  The model's text response, or an error message string.
    """
    api_key = _os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠️ OPENAI_API_KEY not set."

    client = OpenAI(api_key=api_key)
    file_ids: list[str] = []

    if files:
        for fname, fbytes in files:
            try:
                up = client.files.create(
                    file=(fname, _io.BytesIO(fbytes)), purpose="assistants"
                )
                file_ids.append(up.id)
            except Exception:
                pass  # skip failed uploads; proceed without that file

    req: dict = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
    }

    if file_ids:
        req["attachments"] = [
            {"file_id": fid, "tools": [{"type": "code_interpreter"}]}
            for fid in file_ids
        ]
        req["tool_resources"] = {"code_interpreter": {"file_ids": file_ids}}

    try:
        response = client.responses.create(**req)
        text = getattr(response, "output_text", None)
        if text:
            return text
        return "⚠️ LLM returned no output_text."
    except Exception as err:
        return f"LLM call failed: {err}"
