# FurnaceMind/llm/llm.py
# Purpose: LLM inference clients (text generation only)

from __future__ import annotations
from typing import List, Optional, Literal, Any
from openai import OpenAI
from FurnaceMind.utils.settings import settings


Provider = Literal["openrouter", "openai"]
ApiMode = Literal["responses", "chat_completions"]



# OPENROUTER CLIENT
class OpenRouterClient:
    """
    Wrapper around OpenRouter's OpenAI-compatible API.
    Text generation only.
    """

    def __init__(self):
        cfg = settings.llm.openrouter
        if not cfg.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

        self.model = cfg.model_name
        self.max_tokens = cfg.max_tokens

        self.extra_headers = {
            "HTTP-Referer": settings.app.environment,
            "X-Title": "FurnaceMind",
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
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
            return completion.choices[0].message.content or ""
        except Exception:
            # fallback for older models that only accept max_tokens
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

    def chat_completions(
        self,
        *,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[Any] = "auto",
        stop: Optional[List[str]] = None,
    ) -> Any:
        """Low-level Chat Completions call with optional tool-calling.
        Returns the raw response message object, which may contain tool_calls.
        Parameters:
        - messages: List of message dicts (role/content) for the conversation.
        - tools: Optional list of tool definitions (if using tool-calling).
        - tool_choice: Optional tool choice strategy (e.g. "auto", "none", or
        specific tool name).
        - stop: Optional list of stop tokens for generation.
        Returns:
        - The raw message object from the Chat Completion response, which may include tool_calls if tools were provided and chosen.
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stop": stop,
            "extra_headers": self.extra_headers,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        # Prefer max_completion_tokens (newer OpenAI-compatible APIs). Fallback to max_tokens.
        try:
            return self.client.chat.completions.create(
                **kwargs,
                max_completion_tokens=self.max_tokens,
            )
        except Exception:
            return self.client.chat.completions.create(
                **kwargs,
                max_tokens=self.max_tokens,
            )




# OPENAI CLIENT
class OpenAIClient:
    """
    Wrapper around OpenAI API.
    Text generation only.

    - Uses Chat Completions OR Responses API
    - Robust text extraction for GPT-5-mini
    """

    def __init__(self):
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
