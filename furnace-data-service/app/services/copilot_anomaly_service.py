"""Copilot anomaly scoring service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from app.services.copilot_data_service import CopilotDataService
from app.services.copilot_safety_service import warning
from app.services.material_balance_service import _ensure_src_package, table_data


def severity_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


class CopilotAnomalyService:
    """Wrap existing channeling logic with a deterministic numeric fallback."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        data_service: CopilotDataService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.data_service = data_service or CopilotDataService(settings=self.settings)

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            df = self._dataframe(payload)
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError("COPILOT_ANOMALY_INPUT_INVALID", "Invalid anomaly input.", status_code=422) from exc
        if df.empty:
            raise ApiError("COPILOT_ANOMALY_DATA_EMPTY", "No data available for anomaly analysis.", status_code=422)

        try:
            signals = self._score_numeric_columns(df)
            warnings = []
            channeling_signal = self._channeling_signal(df)
            if channeling_signal is not None:
                signals.insert(0, channeling_signal)
            elif not signals:
                warnings.append(warning("COPILOT_ANOMALY_DATA_EMPTY", "No numeric anomaly signals were available."))
            signals = sorted(signals, key=lambda item: item.get("score") or 0.0, reverse=True)
            overall = max((float(item.get("score") or 0.0) for item in signals), default=None)
        except Exception as exc:
            raise ApiError(
                "COPILOT_ANOMALY_CALCULATION_FAILED",
                "Copilot anomaly calculation failed.",
                status_code=500,
            ) from exc

        rows = [
            {
                "name": item["name"],
                "score": item.get("score"),
                "severity": item.get("severity"),
                "value": item.get("value"),
                "baseline": item.get("baseline"),
            }
            for item in signals
        ]
        return {
            "overall_score": overall,
            "severity": severity_from_score(overall),
            "signals": signals[:50],
            "summary": {
                "signal_count": len(signals),
                "row_count": len(df),
                "column_count": len(df.columns),
            },
            "tables": {"signals": table_data(rows, self.settings.copilot_max_json_rows)},
            "charts": {"scores": {"series": [{"name": "Anomaly score", "x": [r["name"] for r in rows], "y": [r["score"] for r in rows]}]}},
            "warnings": warnings,
            "computed_at": datetime.now(timezone.utc),
        }

    def _dataframe(self, payload: dict[str, Any]) -> pd.DataFrame:
        if payload.get("input_data") is not None:
            try:
                df = self.data_service.dataframe_from_input(payload.get("input_data"))
            except ApiError as exc:
                raise ApiError(
                    "COPILOT_ANOMALY_INPUT_INVALID",
                    "Invalid anomaly input.",
                    status_code=422,
                ) from exc
        else:
            recent = self.data_service.fetch_recent_data(
                {
                    "source": payload.get("source") or "online",
                    "start_time": payload.get("start_time"),
                    "end_time": payload.get("end_time"),
                    "columns": payload.get("columns"),
                    "filters": payload.get("options") or {},
                    "limit": self.settings.copilot_max_context_rows,
                    "timezone": payload.get("timezone"),
                }
            )
            df = self.data_service.dataframe_from_input(recent["rows"])
        columns = payload.get("columns")
        if columns:
            present = [column for column in columns if column in df.columns]
            if not present:
                raise ApiError("COPILOT_ANOMALY_INPUT_INVALID", "None of the requested columns exist.", status_code=422)
            df = df[present]
        return df

    def _score_numeric_columns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        numeric = df.select_dtypes(include="number")
        for column in numeric.columns:
            series = pd.to_numeric(numeric[column], errors="coerce").dropna()
            if series.empty:
                continue
            latest = float(series.iloc[-1])
            baseline = float(series.mean())
            spread = float(series.std()) if len(series) > 1 else 0.0
            score = min(1.0, abs(latest - baseline) / (3.0 * spread)) if spread > 0 else 0.0
            signals.append(
                {
                    "name": str(column),
                    "score": round(score, 4),
                    "severity": severity_from_score(score),
                    "value": latest,
                    "baseline": baseline,
                    "unit": None,
                    "explanation": f"Latest value compared with the window mean for {column}.",
                    "evidence": {"rows": len(series), "std": spread},
                }
            )
        return signals

    def _channeling_signal(self, df: pd.DataFrame) -> dict[str, Any] | None:
        try:
            _ensure_src_package("utils")
            from utils.anomaly_propensity import Channeling

            indexed = df.copy()
            if not isinstance(indexed.index, pd.DatetimeIndex):
                for candidate in ("time", "timestamp", "index"):
                    if candidate in indexed.columns:
                        indexed[candidate] = pd.to_datetime(indexed[candidate], errors="coerce")
                        indexed = indexed.dropna(subset=[candidate]).set_index(candidate)
                        break
            if not isinstance(indexed.index, pd.DatetimeIndex):
                return None
            scores = Channeling().score_timeseries(indexed)
            if scores.empty or "channeling_score" not in scores:
                return None
            score = float(pd.to_numeric(scores["channeling_score"], errors="coerce").dropna().iloc[-1])
            return {
                "name": "channeling_score",
                "score": round(min(1.0, max(0.0, score)), 4),
                "severity": severity_from_score(score),
                "value": score,
                "baseline": None,
                "unit": None,
                "explanation": "Channeling propensity from existing detector.",
                "evidence": {"n_windows": len(scores)},
            }
        except Exception:
            return None
