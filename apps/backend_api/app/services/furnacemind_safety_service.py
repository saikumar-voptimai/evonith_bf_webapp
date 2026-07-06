"""Safety, redaction, and limit helpers for FurnaceMind."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from furnace_data.runtime_paths import get_runtime_dir

_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|pwd|token|api[_-]?key|secret|authorization|bearer|connection[_-]?string|database_url|qdrant)",
    re.IGNORECASE,
)
_SECRET_TEXT_RE = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._\-]+|([A-Za-z0-9_]*(?:API_KEY|TOKEN|SECRET)[A-Za-z0-9_]*\s*=\s*)[^\s]+",
    re.IGNORECASE,
)
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s`'\"]+")
_POSIX_RUNTIME_RE = re.compile(r"/(?:var/lib|tmp|home|Users|dev)/[^\s`'\"]+", re.IGNORECASE)
_REDACTED = "[REDACTED]"


def warning(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details or {}}


class FurnaceMindSafetyService:
    """Redact sensitive values and enforce FurnaceMind safety limits."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def redact(self, value: Any) -> Any:
        return self._redact_value(copy.deepcopy(value), parent_key="")

    def _redact_value(self, value: Any, *, parent_key: str) -> Any:
        if _SENSITIVE_KEY_RE.search(parent_key):
            return _REDACTED
        if isinstance(value, dict):
            return {str(key): self._redact_value(item, parent_key=str(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact_value(item, parent_key=parent_key) for item in value]
        if isinstance(value, tuple):
            return [self._redact_value(item, parent_key=parent_key) for item in value]
        if isinstance(value, Path):
            return _REDACTED
        if isinstance(value, str):
            return self._redact_text(value)
        return value

    def _redact_text(self, text: str) -> str:
        output = _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1) or match.group(2)}{_REDACTED}", text)
        runtime = str(get_runtime_dir())
        if runtime and runtime in output:
            output = output.replace(runtime, _REDACTED)
        output = _WINDOWS_PATH_RE.sub(_REDACTED, output)
        output = _POSIX_RUNTIME_RE.sub(_REDACTED, output)
        return output

    def enforce_message_limit(self, content: str) -> str:
        text = str(content or "")
        if len(text) > self.settings.furnacemind_max_message_chars:
            raise ApiError(
                "FURNACEMIND_INPUT_TOO_LARGE",
                "FurnaceMind message exceeds the configured size limit.",
                status_code=413,
                details={"max_message_chars": self.settings.furnacemind_max_message_chars},
            )
        return text

    def cap_prompt(self, prompt: str) -> tuple[str, list[dict[str, Any]]]:
        max_chars = self.settings.furnacemind_max_prompt_chars
        if len(prompt) <= max_chars:
            return prompt, []
        return (
            prompt[:max_chars],
            [
                warning(
                    "FURNACEMIND_PROMPT_TOO_LARGE",
                    f"Prompt was truncated to {max_chars} characters.",
                    {"original_chars": len(prompt), "returned_chars": max_chars},
                )
            ],
        )

    def cap_history(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        max_items = self.settings.furnacemind_max_history_messages
        if len(messages) <= max_items:
            return messages, []
        return (
            messages[-max_items:],
            [
                warning(
                    "FURNACEMIND_CONTEXT_TOO_LARGE",
                    f"Conversation history capped to {max_items} messages.",
                    {"message_count": len(messages), "returned_messages": max_items},
                )
            ],
        )

    def cap_context_docs(self, docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        max_docs = self.settings.furnacemind_max_context_docs
        if len(docs) <= max_docs:
            return docs, []
        return (
            docs[:max_docs],
            [
                warning(
                    "FURNACEMIND_CONTEXT_TOO_LARGE",
                    f"Context documents capped to {max_docs}.",
                    {"document_count": len(docs), "returned_documents": max_docs},
                )
            ],
        )

    def cap_context_chars(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        max_chars = self.settings.furnacemind_max_context_chars
        if len(text) <= max_chars:
            return text, []
        return (
            text[:max_chars],
            [
                warning(
                    "FURNACEMIND_CONTEXT_TOO_LARGE",
                    f"Retrieved context capped to {max_chars} characters.",
                    {"original_chars": len(text), "returned_chars": max_chars},
                )
            ],
        )

    def ensure_raw_docs_allowed(self, *, requested: bool) -> None:
        if requested and not self.settings.furnacemind_allow_raw_docs_to_llm:
            raise ApiError(
                "FURNACEMIND_RAW_DOCUMENTS_NOT_ALLOWED",
                "Raw full documents cannot be sent to LLM providers by configuration.",
                status_code=403,
            )

    def block_unsafe_options(self, options: dict[str, Any] | None) -> None:
        options = options or {}
        if options.get("enable_code_execution") or options.get("code_execution"):
            raise ApiError(
                "FURNACEMIND_CODE_EXECUTION_DISABLED",
                "FurnaceMind code execution is disabled.",
                status_code=403,
            )
        if options.get("enable_shell_execution") or options.get("shell_execution"):
            raise ApiError(
                "FURNACEMIND_SHELL_EXECUTION_DISABLED",
                "FurnaceMind shell execution is disabled.",
                status_code=403,
            )
        requested_tools = options.get("tool_calls") or []
        for item in requested_tools if isinstance(requested_tools, list) else []:
            name = str((item or {}).get("name") or "").lower()
            if any(token in name for token in ("python", "shell", "exec", "eval", "code")):
                raise ApiError(
                    "FURNACEMIND_UNSAFE_INPUT",
                    "Requested FurnaceMind tool is unsafe.",
                    status_code=403,
                )

    def safe_log_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        redacted = self.redact(payload)
        return {
            "keys": sorted(redacted.keys()),
            "chars": len(json.dumps(redacted, default=str)),
        }
