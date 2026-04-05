"""OpenAI LLM wrapper for AI Copilot.

Single entry point: ``call_llm(system, user, files)``.

Uses the Responses API with code_interpreter enabled.
Falls back gracefully if the API call fails.
"""

from __future__ import annotations

import io
import os

from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_CODE_INTERPRETER = True


def call_llm(
    system_prompt: str,
    user_prompt: str,
    files: list[tuple[str, bytes]] | None = None,
) -> str:
    """Send prompts to OpenAI Responses API with optional file attachments.

    Parameters
    ----------
    system_prompt:  LLM persona / task framing.
    user_prompt:    The main user message (may contain pre-computed data).
    files:          Optional list of (filename, bytes) tuples to upload and
                    attach to the code_interpreter tool.

    Returns
    -------
    str  The model's text response, or an error message.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠️ OPENAI_API_KEY not set."

    client = OpenAI(api_key=api_key)
    file_ids: list[str] = []

    if files:
        for fname, fbytes in files:
            try:
                up = client.files.create(
                    file=(fname, io.BytesIO(fbytes)), purpose="assistants"
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
    }

    if USE_CODE_INTERPRETER:
        req["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]

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
