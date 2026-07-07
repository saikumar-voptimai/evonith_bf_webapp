"""Allowlisted FurnaceMind tool registry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings


class DataSummaryInput(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)


class QueryInput(BaseModel):
    query: str | None = None


_TOOL_DESCRIPTIONS: dict[str, tuple[str, type[BaseModel]]] = {
    "data_summary": ("Summarize provided JSON rows without external access.", DataSummaryInput),
    "anomaly_summary": ("Create a simple numeric anomaly summary from provided rows.", DataSummaryInput),
    "material_balance_summary": ("Return a safe Material Balance availability summary.", QueryInput),
    "recommendations_summary": ("Return a safe Recommendations availability summary.", QueryInput),
    "blend_optimizer_summary": ("Return a safe Blend Optimizer availability summary.", QueryInput),
}


class FurnaceMindToolRegistry:
    """Expose safe allowlisted tool metadata."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def allowed_tool_names(self) -> list[str]:
        configured = [str(item).strip() for item in self.settings.furnacemind_allowed_tools if str(item).strip()]
        return [name for name in configured if name in _TOOL_DESCRIPTIONS]

    def list_tools(self) -> list[dict[str, Any]]:
        return [self.info(name) for name in self.allowed_tool_names()]

    def info(self, name: str) -> dict[str, Any]:
        description, schema = _TOOL_DESCRIPTIONS[name]
        return {
            "name": name,
            "description": description,
            "enabled": bool(self.settings.furnacemind_tools_enabled),
            "input_schema": schema.model_json_schema(),
            "safe_for_user": True,
            "requires_auth": bool(self.settings.furnacemind_require_auth),
            "timeout_seconds": self.settings.furnacemind_tool_timeout_seconds,
        }

    def schema_for(self, name: str) -> type[BaseModel] | None:
        item = _TOOL_DESCRIPTIONS.get(name)
        return item[1] if item else None

    def is_allowed(self, name: str) -> bool:
        return name in self.allowed_tool_names()
