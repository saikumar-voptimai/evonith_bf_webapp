"""Safe V-Board InfluxDB repository helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from furnace_data.vboard.catalog import (
    heatload_fields_for_row,
    heatload_fields_for_rows,
    temperature_fields,
)


class VBoardRepository:
    """Read only allowlisted V-Board source fields from InfluxDB."""

    def __init__(self, *, source: str = "historical") -> None:
        self.source = source
        self.temperature_calls = 0
        self.heatload_contour_calls = 0
        self.heatload_timeseries_calls = 0
        self.last_selected_fields: dict[str, list[str]] = {}

    def fetch_temperature_contour(self, start: datetime, end: datetime) -> pd.DataFrame:
        self.temperature_calls += 1
        fields = temperature_fields()
        self.last_selected_fields["temperature_profile"] = fields
        return self._fetch(
            measurement="temperature_profile",
            start=start,
            end=end,
            request_type="avg-min-max",
            selected_fields=fields,
        )

    def fetch_heatload_contour(self, start: datetime, end: datetime) -> pd.DataFrame:
        self.heatload_contour_calls += 1
        fields = heatload_fields_for_rows()
        self.last_selected_fields["heatload_delta_t"] = fields
        return self._fetch(
            measurement="heatload_delta_t",
            start=start,
            end=end,
            request_type="avg-min-max",
            selected_fields=fields,
        )

    def fetch_heatload_timeseries(
        self,
        start: datetime,
        end: datetime,
        *,
        row_id: str,
        window_by: str,
    ) -> pd.DataFrame:
        self.heatload_timeseries_calls += 1
        fields = heatload_fields_for_row(row_id)
        self.last_selected_fields["heatload_delta_t"] = fields
        return self._fetch(
            measurement="heatload_delta_t",
            start=start,
            end=end,
            request_type="windowed-average",
            window_by=window_by,
            selected_fields=fields,
        )

    def _fetch(
        self,
        *,
        measurement: str,
        start: datetime,
        end: datetime,
        request_type: str,
        selected_fields: Iterable[str],
        window_by: str | None = None,
    ) -> pd.DataFrame:
        # Import lazily so importing backend routes does not initialise the
        # Influx client dependency or a source connection.
        from furnace_data.influx.base import BaseDataFetcher

        fetcher = BaseDataFetcher(measurement, source=self.source)
        return fetcher.fetch_averaged_data(
            "over selected range",
            start,
            end,
            request_type=request_type,
            window_by=window_by,
            selected_fields=list(selected_fields),
        )
