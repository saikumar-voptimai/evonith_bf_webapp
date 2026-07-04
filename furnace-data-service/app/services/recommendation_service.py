"""Backend service for deterministic, non-LLM recommendations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from app.services.compute_artifact_service import ComputeArtifactService
from app.services.material_balance_service import _warning, table_data


class RecommendationService:
    """Create bounded recommendation results without LLM provider calls."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        artifact_service: ComputeArtifactService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)

    def config(self) -> dict[str, Any]:
        return {
            "sources": ["input_data", "static_dataset"],
            "max_items": self.settings.recommendations_max_items,
            "explanations_enabled": self.settings.recommendations_enable_explanations,
            "llm_available": False,
            "categories": ["stability", "fuel", "burden", "quality", "safety"],
        }

    def run(self, payload: dict[str, Any], *, route_prefix: str = "/recommendations") -> dict[str, Any]:
        max_items = min(
            self.settings.recommendations_max_items,
            max(1, int(payload.get("max_items") or self.settings.recommendations_max_items)),
        )
        input_data = payload.get("input_data") or {}
        if input_data is not None and not isinstance(input_data, dict):
            raise ApiError(
                "RECOMMENDATIONS_INPUT_INVALID",
                "Recommendation input_data must be an object.",
                status_code=422,
            )

        signals = input_data.get("signals") if isinstance(input_data, dict) else None
        items = self._items_from_signals(signals or input_data, max_items=max_items)
        warnings = []
        if payload.get("include_explanations", True) and not self.settings.recommendations_enable_explanations:
            warnings.append(
                _warning(
                    "RECOMMENDATIONS_EXPLANATIONS_DISABLED",
                    "Explanations are disabled by configuration.",
                )
            )
        rows = [
            {
                "id": item["id"],
                "title": item["title"],
                "priority": item["priority"],
                "category": item["category"],
                "confidence": item["confidence"],
            }
            for item in items
        ]
        artifacts = []
        table = table_data(rows, self.settings.compute_max_json_rows)
        if table["truncated"]:
            metadata = self.artifacts.create_csv_artifact(
                workflow="recommendations",
                filename_prefix="recommendations",
                rows=rows,
            )
            artifacts.append(self.artifacts.response(metadata, route_prefix))
        return {
            "items": items,
            "summary": {
                "source": payload.get("source") or "input_data",
                "item_count": len(items),
                "truncated_to": max_items,
                "llm_used": False,
            },
            "tables": {"recommendations": table},
            "charts": {},
            "warnings": warnings,
            "artifacts": artifacts,
            "computed_at": datetime.now(timezone.utc),
        }

    def _items_from_signals(self, signals: Any, *, max_items: int) -> list[dict[str, Any]]:
        if not isinstance(signals, dict) or not signals:
            return [
                {
                    "id": "rec_baseline_review",
                    "title": "Review current operating window",
                    "description": "No explicit signals were supplied; review current furnace KPIs before changing controls.",
                    "priority": "medium",
                    "category": "stability",
                    "confidence": 0.5,
                    "expected_impact": {},
                    "actions": ["Check latest process and burden stability indicators."],
                    "evidence": {"source": "fallback"},
                    "warnings": [],
                }
            ][:max_items]

        items: list[dict[str, Any]] = []
        for idx, (name, value) in enumerate(signals.items(), start=1):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            priority = "high" if abs(numeric) >= 10 else "medium" if abs(numeric) >= 3 else "low"
            direction = "high" if numeric > 0 else "low" if numeric < 0 else "stable"
            items.append(
                {
                    "id": f"rec_{idx:03d}",
                    "title": f"Review {name}",
                    "description": f"{name} is {direction} relative to the supplied target signal.",
                    "priority": priority,
                    "category": "stability",
                    "confidence": min(0.95, 0.45 + abs(numeric) / 100.0),
                    "expected_impact": {"signal_delta": numeric},
                    "actions": [f"Validate {name} trend before applying control changes."],
                    "evidence": {"signal": name, "value": value, "llm_used": False},
                    "warnings": [],
                }
            )
            if len(items) >= max_items:
                break
        return items
