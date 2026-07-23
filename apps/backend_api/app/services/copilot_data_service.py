"""Copilot data-window retrieval and summarization."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from apps.backend_api.app.api.v1.schemas.data import OfflineDataQuery, OnlineDataQuery
from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.copilot_safety_service import CopilotSafetyService, warning
from apps.backend_api.app.services.serialization import dataframe_to_preview


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

    @staticmethod
    def _aware_datetime(value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _time_range(self, payload: dict[str, Any], filters: dict[str, Any]) -> dict[str, Any]:
        start = self._aware_datetime(payload.get("start_time"))
        end = self._aware_datetime(payload.get("end_time"))
        if start is not None and end is not None:
            return {"kind": "absolute", "start": start, "end": end}
        preset = str(filters.get("preset_id") or filters.get("preset") or "last_1_hour")
        return {"kind": "preset", "preset_id": preset.strip().lower().replace(" ", "_")}

    def _typed_query(self, source: str, payload: dict[str, Any], limit: int):
        filters = payload.get("filters") or {}
        if not isinstance(filters, dict):
            raise ApiError("COPILOT_DATA_QUERY_INVALID", "filters must be an object.", status_code=422)
        time_range = self._time_range(payload, filters)
        columns = payload.get("columns") if isinstance(payload.get("columns"), list) else None
        if source == "online":
            measurements = filters.get("measurements")
            if not isinstance(measurements, list) or not measurements:
                # Legacy callers used ``columns`` for measurement IDs; preserve
                # that only when every value is a valid public measurement ID.
                if columns and all(isinstance(value, str) and value in {"process_params", "temperature_profile", "heatload_delta_t", "cooling_water", "delta_t", "miscellaneous"} for value in columns):
                    measurements = columns
                    fields = None
                else:
                    measurements = ["process_params"]
                    fields = filters.get("fields") or columns
            else:
                fields = filters.get("fields") or columns
            return OnlineDataQuery(
                source="online",
                measurements=measurements,
                time_range=time_range,
                aggregation=None,
                fields=fields,
                limit=max(1, limit),
                offset=0,
            )
        if source == "offline":
            report_id = str(filters.get("report_id") or filters.get("report_type") or payload.get("report_type") or "hm_slag")
            return OfflineDataQuery(
                source="offline",
                selection={"kind": "report", "report_id": report_id.strip().lower()},
                time_range=time_range,
                aggregation=None,
                fields=columns,
                limit=max(1, limit),
                offset=0,
            )
        raise ApiError("COPILOT_DATA_QUERY_INVALID", "Unsupported data source.", status_code=422)

    def fetch_recent_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = str(payload.get("source") or "online").strip().lower()
        limit = min(self.settings.copilot_max_json_rows, max(1, int(payload.get("limit") or 500)))
        warnings: list[dict[str, Any]] = []
        try:
            if source in {"input_data", "mock", "test"}:
                rows = (payload.get("filters") or {}).get("rows", [])
                df = self.dataframe_from_input(rows)
            else:
                from apps.backend_api.app.services import data_service

                df = data_service.fetch_dataframe(
                    self._typed_query(source, payload, limit),
                    settings=self.settings,
                )
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
