from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from apps.backend_api.app.api.v1.schemas.vboard import (
    VBoardContoursRequest,
    VBoardHeatloadTimeseriesRequest,
)
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.vboard_service import VBoardService
from furnace_data.vboard.catalog import (
    heatload_fields_for_row,
    heatload_fields_for_rows,
    load_vboard_catalog,
)


NOW = datetime(2026, 7, 23, 4, 30, tzinfo=timezone.utc)


class FakeRepository:
    def __init__(self, *, fail_temperature=False, fail_heatload=False):
        self.fail_temperature = fail_temperature
        self.fail_heatload = fail_heatload
        self.temperature_calls = 0
        self.heatload_contour_calls = 0
        self.heatload_timeseries_calls = 0
        self.last_selected_fields = {}

    def fetch_temperature_contour(self, start, end):
        self.temperature_calls += 1
        if self.fail_temperature:
            raise OSError("temperature down")
        data = {}
        for level in load_vboard_catalog()["temperature_levels"]:
            for idx in range(level["sensor_count"]):
                suffix = "abcdefghijklmn"[idx]
                field = f"temp_{level['id']}_{suffix}"
                data[f"{field}_mean"] = [300.0 + idx]
                data[f"{field}_min"] = [290.0 + idx]
                data[f"{field}_max"] = [310.0 + idx]
        return pd.DataFrame(data)

    def fetch_heatload_contour(self, start, end):
        self.heatload_contour_calls += 1
        if self.fail_heatload:
            raise OSError("heatload down")
        self.last_selected_fields["heatload_delta_t"] = heatload_fields_for_rows()
        data = {}
        for field in heatload_fields_for_rows():
            data[f"{field}_mean"] = [0.4]
            data[f"{field}_min"] = [0.2]
            data[f"{field}_max"] = [0.6]
        return pd.DataFrame(data)

    def fetch_heatload_timeseries(self, start, end, *, row_id, window_by):
        self.heatload_timeseries_calls += 1
        if self.fail_heatload:
            raise OSError("heatload down")
        self.last_selected_fields["heatload_delta_t"] = heatload_fields_for_row(row_id)
        return pd.DataFrame(
            {
                "time": pd.date_range(start, periods=3, freq="1min", tz="UTC"),
                **{field: [0.2, 0.3, 0.4] for field in heatload_fields_for_row(row_id)},
            }
        )


def _settings(**changes):
    defaults = {
        "vboard_max_absolute_range_days": 31,
        "vboard_max_timeseries_points": 2000,
        "vboard_max_source_rows": 200000,
        "vboard_processing_policy": "legacy_v1",
        "vboard_default_timezone": "Asia/Kolkata",
        "vboard_cache_ttl_seconds": 60,
        "vboard_historical_cache_ttl_seconds": 300,
        "vboard_query_timeout_seconds": 20,
        "vboard_clock_skew_seconds": 300,
        "vboard_cache_max_items": 128,
    }
    defaults.update(changes)
    return SimpleNamespace(**defaults)


def _service(repository=None, **settings):
    return VBoardService(
        settings=_settings(**settings),
        repository=repository or FakeRepository(),
        clock=lambda: NOW,
    )


def test_presets_resolve_exact_duration_and_cache_hits():
    repo = FakeRepository()
    service = _service(repo)
    request = VBoardContoursRequest(
        time_range={"kind": "preset", "preset_id": "last_1_hour"}
    )

    first = service.query_contours(request)
    second = service.query_contours(request)

    assert first == second
    assert first["resolved_range"]["start"] == "2026-07-23T03:30:00Z"
    assert repo.temperature_calls == 1
    assert repo.heatload_contour_calls == 1
    assert service.last_metrics["cache"] == "hit"


def test_absolute_ist_range_converts_to_utc_and_rejects_naive():
    service = _service()
    request = VBoardContoursRequest(
        time_range={
            "kind": "absolute",
            "start": "2026-07-23T08:00:00+05:30",
            "end": "2026-07-23T10:00:00+05:30",
        }
    )

    result = service.query_contours(request)

    assert result["resolved_range"]["start"] == "2026-07-23T02:30:00Z"
    assert result["resolved_range"]["end"] == "2026-07-23T04:30:00Z"

    with pytest.raises(ApiError) as exc_info:
        service.query_contours(
            VBoardContoursRequest(
                time_range={
                    "kind": "absolute",
                    "start": "2026-07-23T08:00:00",
                    "end": "2026-07-23T10:00:00+05:30",
                }
            )
        )
    assert exc_info.value.code == "INVALID_TIME_RANGE"


def test_oversized_range_uses_stable_error_code():
    service = _service(vboard_max_absolute_range_days=1)

    with pytest.raises(ApiError) as exc_info:
        service.query_contours(
            VBoardContoursRequest(
                time_range={
                    "kind": "absolute",
                    "start": "2026-07-20T00:00:00+00:00",
                    "end": "2026-07-23T00:00:00+00:00",
                }
            )
        )

    assert exc_info.value.code == "VBOARD_RANGE_TOO_LARGE"
    assert exc_info.value.status_code == 413


def test_partial_and_complete_source_failures_are_isolated():
    service = _service(FakeRepository(fail_heatload=True))
    request = VBoardContoursRequest(
        time_range={"kind": "preset", "preset_id": "last_1_hour"}
    )

    result = service.query_contours(request)

    assert result["temperature"]["status"] == "ok"
    assert result["heatload"]["status"] == "unavailable"

    both_failed = _service(FakeRepository(fail_temperature=True, fail_heatload=True))
    with pytest.raises(ApiError) as exc_info:
        both_failed.query_contours(request)
    assert exc_info.value.code == "VBOARD_DATA_UNAVAILABLE"
    assert exc_info.value.status_code == 503


def test_timeseries_validates_row_and_selects_only_four_fields():
    repo = FakeRepository()
    service = _service(repo)
    request = VBoardHeatloadTimeseriesRequest(
        row_id="R6",
        time_range={"kind": "preset", "preset_id": "last_6_hours"},
        resolution={"mode": "auto"},
    )

    result = service.query_heatload_timeseries(request)

    assert result["row"]["id"] == "R6"
    assert result["resolved_window_seconds"] == 60
    assert repo.heatload_timeseries_calls == 1
    assert repo.last_selected_fields["heatload_delta_t"] == heatload_fields_for_row("R6")

    with pytest.raises(ApiError) as exc_info:
        service.query_heatload_timeseries(
            VBoardHeatloadTimeseriesRequest(
                row_id="R5",
                time_range={"kind": "preset", "preset_id": "last_6_hours"},
            )
        )
    assert exc_info.value.code == "INVALID_VBOARD_ROW"
