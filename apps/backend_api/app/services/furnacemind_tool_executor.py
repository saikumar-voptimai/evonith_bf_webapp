"""Safe FurnaceMind tool execution service."""

from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.furnacemind_safety_service import FurnaceMindSafetyService, warning
from apps.backend_api.app.services.furnacemind_tool_registry import FurnaceMindToolRegistry


class FurnaceMindToolExecutor:
    """Execute a bounded set of safe read-only tools."""

    def __init__(
        self,
        *,
        registry: FurnaceMindToolRegistry | None = None,
        settings: BackendSettings | None = None,
        safety: FurnaceMindSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.registry = registry or FurnaceMindToolRegistry(self.settings)
        self.safety = safety or FurnaceMindSafetyService(self.settings)

    def execute_many(self, tool_calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not tool_calls:
            return [], []
        if len(tool_calls) > self.settings.furnacemind_max_tool_calls_per_run:
            raise ApiError("FURNACEMIND_TOOL_LIMIT_EXCEEDED", "Too many FurnaceMind tool calls requested.", 429)
        results = []
        warnings: list[dict[str, Any]] = []
        for item in tool_calls:
            try:
                results.append(self.execute(str(item.get("name") or ""), item.get("input") or {}))
            except ApiError as exc:
                if exc.code == "FURNACEMIND_TOOLS_DISABLED":
                    warnings.append(warning(exc.code, exc.message))
                    continue
                raise
        return results, warnings

    def execute(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        clean_name = str(name or "").strip()
        if not self.settings.furnacemind_tools_enabled:
            raise ApiError("FURNACEMIND_TOOLS_DISABLED", "FurnaceMind tools are disabled.", 409)
        if any(token in clean_name.lower() for token in ("python", "shell", "exec", "eval", "code")):
            raise ApiError("FURNACEMIND_TOOL_UNSAFE", "Requested FurnaceMind tool is unsafe.", 403)
        if not self.registry.is_allowed(clean_name):
            raise ApiError("FURNACEMIND_TOOL_NOT_ALLOWED", "FurnaceMind tool is not allowed.", 403)
        schema = self.registry.schema_for(clean_name)
        if schema is None:
            raise ApiError("FURNACEMIND_TOOL_NOT_ALLOWED", "FurnaceMind tool is not allowed.", 403)
        started = time.perf_counter()
        try:
            validated = schema.model_validate(payload)
        except ValidationError as exc:
            raise ApiError("FURNACEMIND_TOOL_INPUT_INVALID", "FurnaceMind tool input is invalid.", 422, {"errors": exc.errors()}) from exc
        try:
            output = self._dispatch(clean_name, validated.model_dump())
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError("FURNACEMIND_TOOL_EXECUTION_FAILED", "FurnaceMind tool execution failed.", 500) from exc
        duration_ms = int((time.perf_counter() - started) * 1000)
        if duration_ms > self.settings.furnacemind_tool_timeout_seconds * 1000:
            raise ApiError("FURNACEMIND_TOOL_TIMEOUT", "FurnaceMind tool timed out.", 504)
        return {
            "tool_name": clean_name,
            "status": "completed",
            "output": self.safety.redact(output),
            "error_code": None,
            "error_message": None,
            "duration_ms": duration_ms,
            "warnings": [],
        }

    def _dispatch(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name == "data_summary":
            rows = payload.get("rows") or []
            columns = sorted({key for row in rows if isinstance(row, dict) for key in row})
            return {"row_count": len(rows), "column_count": len(columns), "columns": columns[:50]}
        if name == "anomaly_summary":
            rows = payload.get("rows") or []
            numeric_values: list[float] = []
            for row in rows:
                if isinstance(row, dict):
                    numeric_values.extend(float(value) for value in row.values() if isinstance(value, (int, float)))
            return {
                "numeric_value_count": len(numeric_values),
                "max": max(numeric_values) if numeric_values else None,
                "min": min(numeric_values) if numeric_values else None,
            }
        return {
            "summary": f"{name} is available through its dedicated backend API; FurnaceMind returns a safe availability note in Phase 9.",
            "query": payload.get("query"),
        }
