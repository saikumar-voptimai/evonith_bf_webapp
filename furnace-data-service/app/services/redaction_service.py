"""Central secret and path redaction helpers for operational surfaces."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from furnace_data.runtime_paths import get_runtime_dir

REDACTED = "[REDACTED]"

_SENSITIVE_KEY_RE = re.compile(
    r"(password|passwd|token|authorization|bearer|api[_-]?key|secret|"
    r"credential|connection[_-]?string|database[_-]?url|qdrant[_-]?api[_-]?key|"
    r"openai[_-]?api[_-]?key|access[_-]?token|refresh[_-]?token|password[_-]?hash|"
    r"(^|[_-])key($|[_-]))",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(OPENAI_API_KEY|QDRANT_API_KEY|EVONITH_AUTH_SECRET_KEY|api_key|token|password|secret)"
    r"\s*=\s*[^,\s]+"
)
_DB_URL_RE = re.compile(
    r"(?i)\b(postgresql|postgres|mysql|mssql|sqlite)://[^\s,;]+"
)


def _runtime_strings() -> list[str]:
    try:
        runtime = str(get_runtime_dir())
    except Exception:
        return []
    values = {runtime, runtime.replace("\\", "/")}
    return [value for value in values if value]


def is_sensitive_key(key: object) -> bool:
    """Return true when a mapping key should have its value redacted."""
    return bool(_SENSITIVE_KEY_RE.search(str(key or "")))


def contains_secret_like_value(text: str) -> bool:
    """Return true when text looks like a token, secret assignment, or DB URL."""
    value = str(text or "")
    return bool(
        _BEARER_RE.search(value)
        or _JWT_RE.search(value)
        or _KEY_VALUE_RE.search(value)
        or _DB_URL_RE.search(value)
    )


def redact_text(text: str) -> str:
    """Redact secret-like values and internal runtime paths from text."""
    output = str(text or "")
    output = _BEARER_RE.sub("Bearer " + REDACTED, output)
    output = _JWT_RE.sub(REDACTED, output)
    output = _KEY_VALUE_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", output)
    output = _DB_URL_RE.sub(lambda m: f"{m.group(1)}://{REDACTED}", output)
    for runtime in _runtime_strings():
        output = output.replace(runtime, "[RUNTIME_DIR]")
    return output


def redact_value(
    value: Any,
    *,
    max_depth: int = 5,
    max_items: int = 100,
    _depth: int = 0,
) -> Any:
    """Return a redacted copy of a supported value without mutating input."""
    if _depth >= max_depth:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return REDACTED if contains_secret_like_value(value) else redact_text(value)
    if isinstance(value, Mapping):
        return redact_dict(value, max_depth=max_depth, max_items=max_items, _depth=_depth)
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:max_items]
        redacted = [
            redact_value(item, max_depth=max_depth, max_items=max_items, _depth=_depth + 1)
            for item in items
        ]
        if len(value) > max_items:
            redacted.append("[TRUNCATED]")
        return redacted
    return redact_text(str(value))


def redact_dict(
    data: Mapping[str, Any],
    *,
    max_depth: int = 5,
    max_items: int = 100,
    _depth: int = 0,
) -> dict[str, Any]:
    """Return a redacted copy of a mapping."""
    if _depth >= max_depth:
        return {"truncated": True}
    output: dict[str, Any] = {}
    for index, (key, value) in enumerate(dict(data).items()):
        if index >= max_items:
            output["truncated"] = True
            break
        key_text = str(key)
        output[key_text] = (
            REDACTED
            if is_sensitive_key(key_text)
            else redact_value(value, max_depth=max_depth, max_items=max_items, _depth=_depth + 1)
        )
    return output


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return safe headers for logs and audit metadata."""
    return {
        str(key): (REDACTED if is_sensitive_key(key) else redact_text(str(value)))
        for key, value in dict(headers).items()
    }


def safe_log_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """Return redacted logging extra fields."""
    return redact_dict(extra, max_depth=4, max_items=50)

