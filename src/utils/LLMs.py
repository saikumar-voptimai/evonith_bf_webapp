from openai import OpenAI
from dotenv import load_dotenv
import json
import io
import os


# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Prefer an env-provided name; fall back to a safe small model
OPENAI_MODEL   = os.getenv("OPENAI_MODEL")
USE_CODE_INTERPRETER = True 
load_dotenv()


class LLMs:
    def __init__(self):
        pass

    def gpt_llm(self, system_prompt: str, user_prompt: str, files: list[tuple[str, bytes]] | None = None) -> str:
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

        return f"LLM call failed: {last_err}"


    def openRouter_llm(self, system_prompt: str, user_prompt: str, files: list[tuple[str, bytes]] | None = None) -> str:
        """
        Call an OpenRouter model using the OpenAI client interface.
        Requires:
        - OPENROUTER_API_KEY
        - OPENROUTER_MODEL 
        """
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_model = os.getenv("OPENROUTER_MODEL")

        if not openrouter_key:
            return "LLM call failed: OPENROUTER_API_KEY not set."

        try:
            # --- OpenRouter client ---
            or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            extra_headers = {
                "HTTP-Referer": os.getenv("SITE_URL", ""),
                "X-Title": os.getenv("SITE_TITLE", ""),
            }

            completion = or_client.chat.completions.create(
                model=openrouter_model,
                messages=messages,
                extra_headers=extra_headers,
            )

            return completion.choices[0].message.content or ""

        except Exception as err:
            print(f"[WARN] OpenRouter failed: {err}")
            return f"LLM call failed: {err}"
        
        
default_llms = LLMs()

def gpt_llm(system_prompt: str, user_prompt: str, files: list[tuple[str, bytes]] | None = None) -> str:
    return default_llms.gpt_llm(system_prompt, user_prompt, files)

def openRouter_llm(system_prompt: str, user_prompt: str, files: list[tuple[str, bytes]] | None = None) -> str:
    return default_llms.openRouter_llm(system_prompt, user_prompt, files)
