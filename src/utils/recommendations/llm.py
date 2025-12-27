import os
import io
import json
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5-mini")
USE_CODE_INTERPRETER = True 

def call_llm(system_prompt: str, user_prompt: str, files: list[tuple[str, bytes]] | None = None) -> str:
    """
    Sends prompts to OpenAI Responses API.
    - Optionally enables the code_interpreter tool.
    - If files are provided, uploads and attaches them for the tool to access.
    - Falls back to a plain Chat Completions request if Responses API call fails.
    """
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set."

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Upload files and collect their IDs
    file_ids: list[str] = []
    if files:
        for fname, fbytes in files:
            try:
                up = client.files.create(file=(fname, io.BytesIO(fbytes)), purpose="assistants")
                file_ids.append(up.id)
            except Exception:
                # ignore upload errors, proceed without that file
                pass

    # Build request
    req = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    tools = []
    if USE_CODE_INTERPRETER:
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
    if tools:
        req["tools"] = tools

    # Attach files primarily via top-level attachments, fallback via tool_resources
    if file_ids:
        req["attachments"] = [
            {"file_id": fid, "tools": [{"type": "code_interpreter"}]} for fid in file_ids
        ]
        req["tool_resources"] = {"code_interpreter": {"file_ids": file_ids}}

    # Try Responses API first
    last_err = None
    try:
        response = client.responses.create(**req)
        # New SDK provides output_text for convenience
        text = getattr(response, "output_text", None)
        if text:
            return text
        # Fallback to raw JSON dump if no convenience field is present
        return json.dumps(response.to_dict(), indent=2)
    except Exception as err1:
        last_err = err1

    # Fallback to Chat Completions (older SDKs)
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        chat = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.2)
        if chat and chat.choices:
            return chat.choices[0].message.content or ""
    except Exception as err2:
        last_err = err2

    return f"LLM call failed: {last_err}"
