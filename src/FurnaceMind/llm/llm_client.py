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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

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
        except Exception as err:
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




# CLIENT SELECTION
def get_llm_client(prefer: Optional[Provider] = None):
    provider = prefer or settings.llm.provider

    if provider == "openrouter":
        return OpenRouterClient() if settings.llm.openrouter.api_key else OpenAIClient()

    if provider == "openai":
        return OpenAIClient() if settings.llm.openai.api_key else OpenRouterClient()

    raise ValueError(f"Unsupported provider: {provider}")