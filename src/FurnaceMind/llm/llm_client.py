# FurnaceMind/llm/llm_client.py
# Purpose: LLM inference clients (text generation only)
# Fixed: Proper exception handling, retry with backoff, removed dead code

from __future__ import annotations

import time
import logging
from typing import List, Optional, Literal, Any

from openai import OpenAI, BadRequestError, AuthenticationError, RateLimitError, APITimeoutError
from FurnaceMind.utils.settings import settings

logger = logging.getLogger(__name__)

Provider = Literal["openrouter", "openai"]
ApiMode = Literal["responses", "chat_completions"]


# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


def _retry_with_backoff(fn, *, max_retries: int = MAX_RETRIES):
    """
    Retry a callable with exponential backoff for transient errors only.
    Raises immediately on auth errors or bad requests.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except AuthenticationError:
            raise  # never retry auth failures
        except BadRequestError:
            raise  # never retry malformed requests
        except RateLimitError as e:
            last_err = e
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait:.1f}s")
            time.sleep(wait)
        except (APITimeoutError, ConnectionError, TimeoutError) as e:
            last_err = e
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"Transient error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait:.1f}s")
            time.sleep(wait)
    raise last_err  # type: ignore[misc]



# OpenRouter Client
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
            timeout=60.0,
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

        def _call():
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    stop=stop,
                    extra_headers=self.extra_headers,
                )
            except BadRequestError:
                # Fallback for models that only accept max_tokens
                logger.info("max_completion_tokens not supported, falling back to max_tokens")
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    stop=stop,
                    extra_headers=self.extra_headers,
                )
            return completion.choices[0].message.content or ""

        return _retry_with_backoff(_call)



# OpenAI Client
class OpenAIClient:
    """
    Wrapper around OpenAI API.
    Text generation only — Chat Completions or Responses API.
    """

    def __init__(self):
        cfg = settings.llm.openai
        if not cfg.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = (
            OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0)
            if cfg.base_url
            else OpenAI(api_key=cfg.api_key, timeout=60.0)
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
        """
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def _call():
            # Responses API path
            if self.api_mode == "responses":
                resp = self.client.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    max_output_tokens=self.max_tokens,
                )
                return self._extract_text_from_responses(resp)

            # Chat Completions path (default)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    stop=stop,
                    extra_headers=self.extra_headers,
                )
            except BadRequestError:
                logger.info("max_completion_tokens not supported, falling back to max_tokens")
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    stop=stop,
                    extra_headers=self.extra_headers,
                )
            return completion.choices[0].message.content or ""

        return _retry_with_backoff(_call)



# Client Selection
def get_llm_client(prefer: Optional[Provider] = None):
    provider = prefer or settings.llm.provider

    if provider == "openrouter":
        return OpenRouterClient() if settings.llm.openrouter.api_key else OpenAIClient()

    if provider == "openai":
        return OpenAIClient() if settings.llm.openai.api_key else OpenRouterClient()

    raise ValueError(f"Unsupported provider: {provider}")