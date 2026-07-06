"""Copilot data-window retrieval and summarization."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from app.api.v1.schemas.data import DataQueryRequest
from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from app.services.copilot_safety_service import CopilotSafetyService, warning
from app.services.serialization import dataframe_to_preview


class CopilotDataService:
    """Fetch compact, JSON-safe data windows for Copilot."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        safety: CopilotSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.safety = safety or CopilotSafetyService(self.settings)

    def dataframe_from_input(self, value: Any) -> pd.DataFrame:
        if value is None:
            return pd.DataFrame()
        if isinstance(value, list):
            if not all(isinstance(item, dict) for item in value):
                raise ApiError("COPILOT_DATA_QUERY_INVALID", "input_data list must contain objects.", status_code=422)
            return pd.DataFrame(value)
        if isinstance(value, dict):
            rows = value.get("rows")
            if isinstance(rows, list):
                return self.dataframe_from_input(rows)
            if all(isinstance(item, (list, tuple)) for item in value.values()):
                return pd.DataFrame(value)
            return pd.DataFrame([value])
        raise ApiError("COPILOT_DATA_QUERY_INVALID", "Unsupported input_data shape.", status_code=422)

    def fetch_recent_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = str(payload.get("source") or "online").strip().lower()
        limit = min(
            self.settings.copilot_max_json_rows,
            max(0, int(payload.get("limit") or 500)),
        )
        warnings: list[dict[str, Any]] = []

        try:
            if source in {"input_data", "mock", "test"}:
                rows = (payload.get("filters") or {}).get("rows", [])
                df = self.dataframe_from_input(rows)
            else:
                from app.services import data_service

                query = DataQueryRequest(
                    source=source,
                    start_time=payload.get("start_time"),
                    end_time=payload.get("end_time"),
                    columns=payload.get("columns"),
                    filters=payload.get("filters") or {},
                    limit=limit,
                    timezone=payload.get("timezone") or "Asia/Kolkata",
                )
                df = data_service.fetch_dataframe(query)
                if payload.get("columns") and source != "online":
                    missing = [col for col in payload["columns"] if col not in df.columns]
                    if missing:
                        raise ApiError(
                            "COPILOT_DATA_QUERY_INVALID",
                            f"Unknown column(s): {missing}",
                            status_code=400,
                        )
                    df = df[payload["columns"]]
        except ApiError:
            raise
        except ValueError as exc:
            raise ApiError("COPILOT_DATA_QUERY_INVALID", str(exc), status_code=400) from exc
        except Exception as exc:
            raise ApiError("COPILOT_DATA_FETCH_FAILED", "Copilot data fetch failed.", status_code=503) from exc

        if df.empty:
            warnings.append(warning("COPILOT_DATA_EMPTY", "No Copilot data rows were returned."))

        columns, rows, row_count, truncated = dataframe_to_preview(
            df,
            limit=limit,
            include_index=isinstance(df.index, pd.DatetimeIndex),
        )
        capped_rows, cap_warnings, cap_truncated = self.safety.cap_rows(rows, limit=limit)
        warnings.extend(cap_warnings)
        return {
            "columns": [column.model_dump() for column in columns],
            "rows": capped_rows,
            "row_count": row_count,
            "returned_rows": len(capped_rows),
            "truncated": bool(truncated or cap_truncated),
            "summary": self.summarize_dataframe(df),
            "warnings": warnings,
        }

    def summarize_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"row_count": 0, "column_count": 0, "numeric_columns": []}
        numeric = df.select_dtypes(include="number")
        stats: dict[str, Any] = {}
        for column in numeric.columns[:25]:
            series = pd.to_numeric(numeric[column], errors="coerce").dropna()
            if series.empty:
                continue
            stats[str(column)] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "latest": float(series.iloc[-1]),
            }
        start = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
        end = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": list(stats),
            "numeric_stats": stats,
            "start_time": start.isoformat() if isinstance(start, (pd.Timestamp, datetime)) else None,
            "end_time": end.isoformat() if isinstance(end, (pd.Timestamp, datetime)) else None,
        }
