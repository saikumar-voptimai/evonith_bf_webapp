"""Dashboard business service."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from apps.backend_api.app.core.errors import ApiError


FetchOnlineDf = Callable[..., Any]


class DashboardService:
    """Read dashboard aggregates from the online telemetry source."""

    _WINDOWS = {"1h": "last 1 hour"}
    _BUCKETS = {"15m": "15 minutes"}
    _METRICS = {
        "production_rate": ("production_per_hour", "t/h"),
        "fuel_rate": ("fuel_rate", "kg/tHM"),
        "eta_co": ("body_etaco", "%"),
        "blast_volume": ("hot_blast_vol_nm3h", "Nm3/h"),
    }

    def __init__(
        self,
        *,
        fetcher: FetchOnlineDf | None = None,
        cache_ttl_seconds: int = 90,
    ) -> None:
        self._fetcher = fetcher
        self._cache_ttl_seconds = max(60, min(120, int(cache_ttl_seconds)))
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any], list[str]]] = {}

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc).replace(microsecond=0)

    def _fetch_online_df(self, *, window: str, bucket: str) -> Any:
        fetcher = self._fetcher
        if fetcher is None:
            from furnace_data.influx.online import fetch_online_df

            fetcher = fetch_online_df
        return fetcher(
            selected_measurements=["process_params"],
            time_range=self._WINDOWS[window],
            window_by=self._BUCKETS[bucket],
            column_naming="field",
        )

    def get_kpis(self, *, window: str = "1h", bucket: str = "15m") -> tuple[dict[str, Any], list[str]]:
        """Return aggregate dashboard KPIs and non-fatal warnings."""
        if window not in self._WINDOWS:
            raise ApiError("VALIDATION_ERROR", "Unsupported dashboard window.", status_code=422)
        if bucket not in self._BUCKETS:
            raise ApiError("VALIDATION_ERROR", "Unsupported dashboard bucket.", status_code=422)

        cache_key = (window, bucket)
        now_monotonic = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] > now_monotonic:
            return cached[1], list(cached[2])

        try:
            df = self._fetch_online_df(window=window, bucket=bucket)
        except Exception as exc:
            raise ApiError(
                "DASHBOARD_DATA_UNAVAILABLE",
                "Dashboard KPI data is unavailable.",
                status_code=503,
            ) from exc

        warnings: list[str] = []
        if df is None or getattr(df, "empty", True):
            warnings.append("No online KPI samples were returned for the requested window.")
            payload = self._empty_payload(window=window, bucket=bucket)
        else:
            payload = self._payload_from_frame(df, window=window, bucket=bucket)

        self._cache[cache_key] = (
            now_monotonic + self._cache_ttl_seconds,
            payload,
            list(warnings),
        )
        return payload, warnings

    def _empty_payload(self, *, window: str, bucket: str) -> dict[str, Any]:
        return {
            "as_of": self._now_utc(),
            "window": window,
            "bucket": bucket,
            "sample_count": 0,
            "metrics": {
                name: {"value": None, "unit": unit}
                for name, (_column, unit) in self._METRICS.items()
            },
        }

    def _payload_from_frame(self, df: Any, *, window: str, bucket: str) -> dict[str, Any]:
        sample_count = int(len(df.index))
        as_of = self._now_utc()
        if getattr(df, "index", None) is not None and sample_count:
            try:
                as_of_value = df.index.max()
                as_of = as_of_value.to_pydatetime().astimezone(timezone.utc).replace(microsecond=0)
            except Exception:
                as_of = self._now_utc()

        metrics: dict[str, dict[str, float | str | None]] = {}
        for name, (column, unit) in self._METRICS.items():
            value = None
            if column in df.columns:
                series = df[column].dropna()
                if not series.empty:
                    value = round(float(series.mean()), 1)
            metrics[name] = {"value": value, "unit": unit}

        return {
            "as_of": as_of,
            "window": window,
            "bucket": bucket,
            "sample_count": sample_count,
            "metrics": metrics,
        }
