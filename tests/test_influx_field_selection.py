"""Backward-compatible field selection in the shared Influx fetch path."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.fetchers.ts_data_fetcher import TimeSeriesDataFetcher
from furnace_data.influx.query import query_builder

START = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)
END = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)


def test_query_builder_default_still_selects_every_raw_field() -> None:
    query = query_builder("process_params", START, END, type="ts", window_by=None)
    assert query.startswith("SELECT * FROM process_params ")


def test_query_builder_selects_only_validated_canonical_fields() -> None:
    query = query_builder(
        "process_params",
        START,
        END,
        type="windowed-average",
        window_by="5 minutes",
        fields=("fuel_rate", "coal_rate_actual_value"),
    )
    assert "MEAN(fuel_rate) AS fuel_rate" in query
    assert "MEAN(coal_rate_actual_value) AS coal_rate_actual_value" in query
    assert "GROUP BY time(5m) fill(null)" in query
    assert "hot_blast_press" not in query


@pytest.mark.parametrize(
    "fields",
    [(), ("not_configured",), ("fuel_rate) FROM secrets; --",)],
)
def test_query_builder_rejects_empty_or_unconfigured_field_selection(fields) -> None:
    with pytest.raises(ValueError):
        query_builder(
            "process_params",
            START,
            END,
            type="ts",
            window_by=None,
            fields=fields,
        )


def test_time_series_fetcher_existing_callers_need_no_fields_argument(
    monkeypatch,
) -> None:
    fetcher = TimeSeriesDataFetcher("process_params", debug=False, source="historical")
    captured = {}

    def fake_fetch(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "time": pd.DatetimeIndex([END]),
                "fuel_rate": [500.0],
                "transport_tag": ["ignored"],
            }
        )

    monkeypatch.setattr(fetcher, "fetch_averaged_data", fake_fetch)

    result = fetcher.fetch_data("over selected range", START, END)

    assert captured["fields"] is None
    assert list(result.columns) == ["time", "fuel_rate"]


def test_time_series_fetcher_forwards_optional_field_selection(monkeypatch) -> None:
    fetcher = TimeSeriesDataFetcher("process_params", debug=False, source="historical")
    captured = {}

    def fake_fetch(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": pd.DatetimeIndex([END]), "fuel_rate": [500.0]})

    monkeypatch.setattr(fetcher, "fetch_averaged_data", fake_fetch)

    fetcher.fetch_data(
        "over selected range",
        START,
        END,
        fields=("fuel_rate",),
    )

    assert captured["fields"] == ("fuel_rate",)
