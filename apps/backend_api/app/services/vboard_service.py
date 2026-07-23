"""Backend V-Board orchestration service."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from typing import Any, Callable

from apps.backend_api.app.core.errors import ApiError
from furnace_data.vboard.catalog import (
    CATALOG_VERSION,
    PROCESSING_POLICY_ID,
    auto_resolution_window_id,
    load_vboard_catalog,
    query_window_for_window_id,
    resolution_windows_by_id,
    rows_by_id,
)
from furnace_data.vboard.models import ResolvedRange
from furnace_data.vboard.repository import VBoardRepository
from furnace_data.vboard.time_ranges import VBoardTimeRangeError, resolve_time_range
from furnace_data.vboard.transforms import (
    transform_heatload_contour,
    transform_heatload_timeseries,
    transform_temperature_contour,
)


class VBoardService:
    """API-first V-Board service returning frontend-neutral domain data."""

    def __init__(
        self,
        *,
        settings: Any | None = None,
        repository: VBoardRepository | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or VBoardRepository(source="historical")
        self.clock = clock
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.last_metrics: dict[str, Any] = {}

    def get_catalog(self) -> dict[str, Any]:
        return load_vboard_catalog(
            max_absolute_range_days=self._setting("vboard_max_absolute_range_days", 31),
            max_timeseries_points_per_quadrant=self._setting(
                "vboard_max_timeseries_points", 2000
            ),
            max_source_rows=self._setting("vboard_max_source_rows", 200000),
            processing_policy_id=self._setting(
                "vboard_processing_policy", PROCESSING_POLICY_ID
            ),
            display_timezone=self._setting("vboard_default_timezone", "Asia/Kolkata"),
        )

    def query_contours(self, request: Any) -> dict[str, Any]:
        start_perf = time.perf_counter()
        catalog = self.get_catalog()
        time_range = _model_dump(request.time_range)
        resolved = self._resolve_range(time_range)
        cache_key = self._cache_key(
            "contours",
            {
                "time_range": self._cache_range_identity(time_range, resolved),
                "catalog_version": CATALOG_VERSION,
                "policy": self._policy_id(),
            },
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._record_metrics("contours", start_perf, cache="hit", source_calls={})
            return cached

        source_calls = {"temperature": 0, "heatload": 0}

        def fetch_temperature():
            source_calls["temperature"] += 1
            return self.repository.fetch_temperature_contour(resolved.start, resolved.end)

        def fetch_heatload():
            source_calls["heatload"] += 1
            return self.repository.fetch_heatload_contour(resolved.start, resolved.end)

        with ThreadPoolExecutor(max_workers=2) as executor:
            temperature_future = executor.submit(fetch_temperature)
            heatload_future = executor.submit(fetch_heatload)
            temperature_frame, temperature_error = self._future_result(
                temperature_future, "temperature"
            )
            heatload_frame, heatload_error = self._future_result(heatload_future, "heatload")

        temperature = None
        heatload = None
        if temperature_error is None:
            try:
                temperature = transform_temperature_contour(
                    temperature_frame,
                    catalog=catalog,
                )
            except Exception as exc:  # transformation failures are domain errors
                temperature_error = exc
        if heatload_error is None:
            try:
                heatload = transform_heatload_contour(heatload_frame, catalog=catalog)
            except Exception as exc:
                heatload_error = exc

        if temperature_error is not None and heatload_error is not None:
            self._record_metrics(
                "contours",
                start_perf,
                cache="miss",
                source_calls=source_calls,
                error_code="VBOARD_DATA_UNAVAILABLE",
            )
            raise ApiError(
                "VBOARD_DATA_UNAVAILABLE",
                "Both V-Board contour sources are unavailable.",
                status_code=503,
            )

        if temperature is None:
            temperature = self._temperature_unavailable(catalog)
        elif heatload_error is not None:
            temperature["warnings"] = list(temperature.get("warnings") or [])
        if heatload is None:
            heatload = self._heatload_unavailable(catalog)

        if temperature_error is not None:
            heatload.setdefault("warnings", []).append(
                "Temperature data source is unavailable for this contour request."
            )
        if heatload_error is not None:
            temperature.setdefault("warnings", []).append(
                "Heat-load data source is unavailable for this contour request."
            )

        data = {
            "generated_at": _iso_z(self._now()),
            "resolved_range": resolved.to_dict(),
            "catalog_version": catalog["catalog_version"],
            "processing_policy_id": catalog["processing_policy"]["id"],
            "temperature": temperature,
            "heatload": heatload,
        }
        self._cache_set(cache_key, data, ttl_seconds=self._cache_ttl(time_range))
        self._record_metrics(
            "contours", start_perf, cache="miss", source_calls=source_calls
        )
        return data

    def query_heatload_timeseries(self, request: Any) -> dict[str, Any]:
        start_perf = time.perf_counter()
        catalog = self.get_catalog()
        row_id = str(request.row_id or "").strip().upper()
        row = rows_by_id().get(row_id)
        if row is None:
            raise ApiError("INVALID_VBOARD_ROW", "Unknown V-Board row.", status_code=400)

        time_range = _model_dump(request.time_range)
        resolved = self._resolve_range(time_range)
        window_id = self._resolve_window_id(_model_dump(request.resolution), resolved)
        window = resolution_windows_by_id()[window_id]
        cache_key = self._cache_key(
            "heatload-timeseries",
            {
                "row_id": row_id,
                "time_range": self._cache_range_identity(time_range, resolved),
                "window_id": window_id,
                "catalog_version": CATALOG_VERSION,
                "policy": self._policy_id(),
            },
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._record_metrics("heatload-timeseries", start_perf, cache="hit", source_calls={})
            return cached

        source_calls = {"heatload": 0}
        try:
            source_calls["heatload"] += 1
            frame = self.repository.fetch_heatload_timeseries(
                resolved.start,
                resolved.end,
                row_id=row_id,
                window_by=query_window_for_window_id(window_id),
            )
        except Exception as exc:
            self._record_metrics(
                "heatload-timeseries",
                start_perf,
                cache="miss",
                source_calls=source_calls,
                error_code="VBOARD_HEATLOAD_UNAVAILABLE",
            )
            raise ApiError(
                "VBOARD_HEATLOAD_UNAVAILABLE",
                "The V-Board heat-load source is unavailable.",
                status_code=503,
            ) from exc

        if len(frame.index) > self._setting("vboard_max_source_rows", 200000):
            raise ApiError(
                "VBOARD_RESULT_TOO_LARGE",
                "The V-Board source result exceeds the configured row limit.",
                status_code=413,
            )

        try:
            transformed = transform_heatload_timeseries(
                frame,
                row_id=row_id,
                resolved_window_seconds=window.seconds,
                max_points_per_quadrant=self._setting("vboard_max_timeseries_points", 2000),
                catalog=catalog,
                processing_policy_id=catalog["processing_policy"]["id"],
            )
        except Exception as exc:
            raise ApiError(
                "VBOARD_PROCESSING_ERROR",
                "The V-Board heat-load time-series result could not be processed.",
                status_code=500,
            ) from exc

        data = {
            "generated_at": _iso_z(self._now()),
            "resolved_range": resolved.to_dict(),
            "row": row,
            **transformed,
        }
        self._cache_set(cache_key, data, ttl_seconds=self._cache_ttl(time_range))
        self._record_metrics(
            "heatload-timeseries", start_perf, cache="miss", source_calls=source_calls
        )
        return data

    def _resolve_range(self, time_range: dict[str, Any]) -> ResolvedRange:
        try:
            return resolve_time_range(
                time_range,
                now_utc=self._now(),
                max_absolute_range_days=self._setting("vboard_max_absolute_range_days", 31),
                clock_skew_seconds=self._setting("vboard_clock_skew_seconds", 300),
            )
        except VBoardTimeRangeError as exc:
            status = 400 if exc.code != "VBOARD_RANGE_TOO_LARGE" else 413
            raise ApiError(exc.code, str(exc), status_code=status) from exc

    def _resolve_window_id(
        self,
        resolution: dict[str, Any],
        resolved: ResolvedRange,
    ) -> str:
        mode = str(resolution.get("mode") or "auto").strip().lower()
        if mode == "auto":
            duration_seconds = int((resolved.end - resolved.start).total_seconds())
            return auto_resolution_window_id(duration_seconds)
        if mode == "fixed":
            window_id = str(resolution.get("window_id") or "").strip()
            if window_id not in resolution_windows_by_id():
                raise ApiError(
                    "INVALID_VBOARD_RESOLUTION",
                    "Unknown V-Board resolution window.",
                    status_code=400,
                )
            return window_id
        raise ApiError(
            "INVALID_VBOARD_RESOLUTION",
            "Unknown V-Board resolution mode.",
            status_code=400,
        )

    def _future_result(self, future: Any, source: str) -> tuple[Any | None, Exception | None]:
        try:
            return future.result(timeout=self._setting("vboard_query_timeout_seconds", 20)), None
        except TimeoutError as exc:
            future.cancel()
            return None, exc
        except Exception as exc:
            return None, exc

    def _temperature_unavailable(self, catalog: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "unit": catalog["display"]["temperature_unit"],
            "levels": [],
            "missing_level_ids": [level["id"] for level in catalog["temperature_levels"]],
            "warnings": ["Temperature data source is unavailable for this contour request."],
        }

    def _heatload_unavailable(self, catalog: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "unit": catalog["display"]["heatload_unit"],
            "display_label": catalog["display"]["heatload_label"],
            "rows": [],
            "missing_row_ids": [row["id"] for row in catalog["rows"]],
            "warnings": ["Heat-load data source is unavailable for this contour request."],
        }

    def _now(self) -> datetime:
        value = self.clock() if self.clock else datetime.now(timezone.utc)
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default) if self.settings is not None else default

    def _policy_id(self) -> str:
        return str(self._setting("vboard_processing_policy", PROCESSING_POLICY_ID))

    def _cache_ttl(self, time_range: dict[str, Any]) -> int:
        if time_range.get("kind") == "preset":
            return int(self._setting("vboard_cache_ttl_seconds", 60))
        return int(self._setting("vboard_historical_cache_ttl_seconds", 300))

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._cache_lock:
            item = self._cache.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(value)

    def _cache_set(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        max_items = int(self._setting("vboard_cache_max_items", 128))
        with self._cache_lock:
            self._cache[key] = (time.time() + max(1, ttl_seconds), copy.deepcopy(value))
            self._cache.move_to_end(key)
            while len(self._cache) > max_items:
                self._cache.popitem(last=False)

    def _cache_key(self, namespace: str, payload: dict[str, Any]) -> str:
        return f"{namespace}:{json.dumps(payload, sort_keys=True, default=str)}"

    def _cache_range_identity(
        self,
        time_range: dict[str, Any],
        resolved: ResolvedRange,
    ) -> dict[str, Any]:
        if time_range.get("kind") == "preset":
            return {"kind": "preset", "preset_id": time_range.get("preset_id")}
        return {"kind": "absolute", "start": resolved.to_dict()["start"], "end": resolved.to_dict()["end"]}

    def _record_metrics(
        self,
        operation: str,
        started_at: float,
        *,
        cache: str,
        source_calls: dict[str, int],
        error_code: str | None = None,
    ) -> None:
        self.last_metrics = {
            "operation": operation,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "cache": cache,
            "source_calls": dict(source_calls),
            "error_code": error_code,
        }


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
