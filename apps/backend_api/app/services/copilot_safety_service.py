"""Safety and redaction helpers for Copilot analysis."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError

_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|pwd|token|api[_-]?key|secret|authorization|bearer|connection[_-]?string|database_url)",
    re.IGNORECASE,
)
_SECRET_TEXT_RE = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._\-]+|([A-Za-z0-9_]*API_KEY[A-Za-z0-9_]*\s*=\s*)[^\s]+",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"


def warning(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details or {}}


class CopilotSafetyService:
    """Redact sensitive values and enforce Copilot context limits."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def redact(self, value: Any) -> Any:
        """Return a redacted deep copy without mutating the original object."""
        return self._redact_value(copy.deepcopy(value), parent_key="")

    def _redact_value(self, value: Any, *, parent_key: str) -> Any:
        if _SENSITIVE_KEY_RE.search(parent_key):
            return _REDACTED
        if isinstance(value, dict):
            return {
                str(key): self._redact_value(item, parent_key=str(key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_value(item, parent_key=parent_key) for item in value]
        if isinstance(value, tuple):
            return [self._redact_value(item, parent_key=parent_key) for item in value]
        if isinstance(value, str):
            return _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1) or match.group(2)}{_REDACTED}", value)
        return value

    def cap_rows(self, rows: list[dict[str, Any]], limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        max_rows = min(
            self.settings.copilot_max_context_rows,
            max(0, int(limit if limit is not None else self.settings.copilot_max_context_rows)),
        )
        capped = list(rows[:max_rows])
        truncated = len(rows) > len(capped)
        warnings = []
        if truncated:
            warnings.append(
                warning(
                    "COPILOT_CONTEXT_TOO_LARGE",
                    f"Context rows capped to {max_rows}.",
                    {"row_count": len(rows), "returned_rows": len(capped)},
                )
            )
        return capped, warnings, truncated

    def enforce_prompt_limit(self, prompt: str) -> tuple[str, list[dict[str, Any]]]:
        max_chars = self.settings.copilot_max_prompt_chars
        if len(prompt) <= max_chars:
            return prompt, []
        if max_chars <= 0:
            raise ApiError("COPILOT_PROMPT_TOO_LARGE", "Prompt size limit is invalid.", status_code=413)
        return (
            prompt[:max_chars],
            [
                warning(
                    "COPILOT_PROMPT_TOO_LARGE",
                    f"Prompt was truncated to {max_chars} characters.",
                    {"original_chars": len(prompt), "returned_chars": max_chars},
                )
            ],
        )

    def ensure_raw_data_allowed(self, *, requested: bool) -> None:
        if requested and not self.settings.copilot_allow_raw_data_to_llm:
            raise ApiError(
                "COPILOT_RAW_DATA_NOT_ALLOWED",
                "Raw plant data cannot be sent to LLM providers by configuration.",
                status_code=403,
            )

    def safe_log_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        redacted = self.redact(payload)
        return {
            "keys": sorted(redacted.keys()),
            "chars": len(json.dumps(redacted, default=str)),
        }
