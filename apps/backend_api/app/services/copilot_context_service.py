"""Build compact Copilot analysis context."""

from __future__ import annotations

from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.services.copilot_safety_service import CopilotSafetyService, warning


class CopilotContextService:
    """Create deterministic redacted context objects for analysis."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        safety: CopilotSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.safety = safety or CopilotSafetyService(self.settings)

    def build_context(
        self,
        *,
        question: str,
        data: dict[str, Any] | None,
        anomaly: dict[str, Any] | None,
        analysis_mode: str,
        warnings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        context = {
            "question": question.strip(),
            "analysis_mode": analysis_mode,
            "data_summary": (data or {}).get("summary", {}),
            "data_columns": (data or {}).get("columns", [])[:50],
            "sample_rows": (data or {}).get("rows", [])[: min(20, self.settings.copilot_max_context_rows)],
            "anomaly_summary": (anomaly or {}).get("summary", {}),
            "top_anomaly_signals": (anomaly or {}).get("signals", [])[:10],
            "warnings": list(warnings or []) + list((data or {}).get("warnings", [])) + list((anomaly or {}).get("warnings", [])),
        }
        if not self.settings.copilot_allow_raw_data_to_llm and context["sample_rows"]:
            context["sample_rows"] = []
            context["warnings"].append(
                warning(
                    "COPILOT_RAW_DATA_NOT_ALLOWED",
                    "Raw data rows were excluded from provider context by configuration.",
                )
            )
        return self.safety.redact(context) if self.settings.copilot_enable_data_redaction else context
