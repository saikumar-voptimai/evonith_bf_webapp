"""Online InfluxDB multi-measurement fetcher.

Provides
--------
fetch_online_df   Fetch and outer-join multiple online measurements into
                  one time-indexed DataFrame (IST index).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import pandas as pd

from furnace_data.influx.base import BaseDataFetcher
from furnace_data.influx.query import (
    TIMEDELTAS,
    display_column_name,
    measurement_label,
)

# Pandas-compatible resampling frequency aliases (human-readable → pandas freq string).
_PANDAS_FREQ_MAP: dict[str, str] = {
    "1 minute":  "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour":    "1h",
    "6 hours":   "6h",
    "8 hours":   "8h",
    "12 hours":  "12h",
    "1 day":     "1d",
}

ColumnNaming = Literal["display", "field", "prefixed_field"]
_INFLUX_METADATA_COLUMNS = {"result", "table"}


def _is_influx_metadata_column(column: object) -> bool:
    name = str(column)
    return name in _INFLUX_METADATA_COLUMNS or name.startswith("iox::")


def fetch_online_df(
    selected_measurements: List[str],
    time_range: str,
    request_type: str = "windowed-average",
    window_by: str = "15 minutes",
    measurement_labels: Optional[Dict[str, str]] = None,
    field_labels: Optional[Dict[str, str]] = None,
    frequency_map: Optional[Dict[str, str]] = None,
    start_time_override: Optional[datetime] = None,
    end_time_override: Optional[datetime] = None,
    column_naming: ColumnNaming = "display",
) -> pd.DataFrame:
    """Fetch and merge online (real-time) measurements from InfluxDB.

    Queries each measurement in *selected_measurements* over the resolved
    lookback window and outer-joins all results into one time-indexed DataFrame.
    By default, column names keep the legacy
    ``"<Measurement Label> - <Field Label>"`` format.

    Args:
        selected_measurements: List of measurement keys
            (e.g. ``["process_params", "heatload_delta_t"]``).
        time_range:  Preset lookback string from
            :data:`~furnace_data.influx.query.TIMEDELTAS`
            (e.g. ``"last 8 hours"``).
        request_type:     InfluxQL aggregation type; default
            ``"windowed-average"``.
        window_by:        Aggregation window; default ``"15 minutes"``.
        measurement_labels: Optional override for measurement -> display label
            mapping. Falls back to labels derived from measurement keys.
        field_labels: Optional override for field -> display label mapping.
            Falls back to config-derived defaults.
        frequency_map: Optional override for window string → pandas freq string
            (e.g. ``{"15 minutes": "15min"}``).  Falls back to _PANDAS_FREQ_MAP.
        start_time_override: UTC datetime for the start of the fetch window.
            When provided, overrides the ``time_range`` lookback.
        end_time_override: UTC datetime for the end of the fetch window.
            Defaults to now when ``start_time_override`` is set but this is not.
        column_naming: Output column convention:
            ``"display"`` keeps the legacy ``"<Measurement> - <Label>"`` shape,
            ``"field"`` returns canonical InfluxDB fields, and
            ``"prefixed_field"`` returns ``"<Measurement> - <field>"``.

    Returns:
        IST-indexed :class:`pandas.DataFrame` with deduplicated columns, or an
        empty DataFrame if nothing was fetched.
    """
    if not selected_measurements:
        return pd.DataFrame()

    if column_naming not in ("display", "field", "prefixed_field"):
        raise ValueError(
            "column_naming must be 'display', 'field', or 'prefixed_field'."
        )

    m_labels = measurement_labels or {}
    f_labels = field_labels or {}
    freq_map = frequency_map or _PANDAS_FREQ_MAP

    now = datetime.now(timezone.utc)
    if start_time_override is not None:
        start_time = start_time_override
        end_time = end_time_override if end_time_override is not None else now
    else:
        start_time = now - TIMEDELTAS[time_range]
        end_time = now

    combined_df = pd.DataFrame()

    for meas in selected_measurements:
        fetcher = BaseDataFetcher(meas)
        df_meas = fetcher.fetch_averaged_data(
            recent_data_of="over selected range",
            start_time=start_time,
            end_time=end_time,
            request_type=request_type,
            window_by=window_by,
        )

        if df_meas is None or df_meas.empty:
            continue

        if "time" not in df_meas.columns:
            continue

        df_meas["time"] = pd.to_datetime(df_meas["time"], errors="coerce", utc=True)
        df_meas = df_meas.dropna(subset=["time"])
        if df_meas.empty:
            continue

        df_meas = df_meas.set_index("time").sort_index()
        df_meas = df_meas.drop(
            columns=[
                col for col in df_meas.columns if _is_influx_metadata_column(col)
            ],
            errors="ignore",
        )
        if df_meas.empty:
            continue

        def _rename(col: str) -> str:
            if column_naming == "field":
                return col
            if column_naming == "prefixed_field":
                return f"{m_labels.get(meas, measurement_label(meas))} - {col}"
            return display_column_name(
                meas,
                col,
                measurement_labels=m_labels,
                field_label_overrides=f_labels,
            )

        df_meas = df_meas.rename(columns={col: _rename(col) for col in df_meas.columns})

        combined_df = (
            df_meas if combined_df.empty else combined_df.join(df_meas, how="outer")
        )

    if combined_df.empty:
        return combined_df

    if not isinstance(combined_df.index, pd.DatetimeIndex):
        return combined_df

    freq = freq_map.get(window_by, window_by)
    if freq and freq.replace(" ", "").lower() != "none":
        combined_df = combined_df.resample(freq).mean(numeric_only=True)
        if combined_df.empty:
            return combined_df
        combined_df.index = combined_df.index + pd.Timedelta(freq)
        combined_df = combined_df.rename(
            index={combined_df.index[-1]: pd.Timestamp(end_time).round("1min")}
        )

    combined_df = combined_df.sort_index()
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # Convert index to IST
    combined_df.index = combined_df.index.tz_convert("Asia/Kolkata")
    combined_df.index.name = "time (IST)"

    return combined_df
