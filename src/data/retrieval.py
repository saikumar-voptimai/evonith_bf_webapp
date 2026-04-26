"""InfluxDB data retrieval helpers for both online and offline buckets.

``clean_rm_data`` and ``fetch_offline_data`` are thin shims that delegate to
``furnace_data.influx.offline``; ``fetch_online_df`` retains webapp-specific
IST conversion, column renaming, and resampling logic.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd

from furnace_data.influx.base import BaseDataFetcher
from furnace_data.influx.offline import clean_rm_data, fetch_offline_data  # noqa: F401
from furnace_data.influx.query import TIMEDELTAS  # noqa: F401


def fetch_online_df(
    selected_measurements: List[str],
    time_range: str,
    FREQUENCY_TO_TIMEDTA: Dict,
    MEASUREMENT_LABELS: Dict,
    FIELD_LABELS: Dict,
    request_type: str = "windowed-average",
    window_by: str = "15 minutes",
) -> pd.DataFrame:
    """Fetch and merge online (real-time) measurements from InfluxDB.

    Queries each measurement in *selected_measurements* over the resolved time
    window and concatenates the results into a single time-indexed DataFrame.
    Column names are prefixed with the measurement label and the human-readable
    field label.

    Args:
        selected_measurements: List of measurement keys to fetch (must be present
            in *MEASUREMENT_LABELS*).
        time_range:            Preset lookback string (e.g. ``"last 8 hours"``).
        FREQUENCY_TO_TIMEDTA:  Mapping from frequency strings to
            :class:`datetime.timedelta` objects.
        MEASUREMENT_LABELS:    Mapping from measurement key to display label.
        FIELD_LABELS:          Mapping from InfluxDB field names to display names.
        request_type:          InfluxQL query type; defaults to
            ``"windowed-average"``.
        window_by:             Aggregation window size; defaults to
            ``"15 minutes"``.

    Returns:
        A time-indexed ``pd.DataFrame`` with one column per fetched field,
        or an empty ``DataFrame`` if *selected_measurements* is empty.
    """
    datafetchers = {key: BaseDataFetcher(key) for key in MEASUREMENT_LABELS.keys()}

    if not selected_measurements:
        return pd.DataFrame()

    combined_df = pd.DataFrame()

    now = datetime.now(timezone.utc)
    start_time = (
        now - TIMEDELTAS[time_range]
    )  # assumes TIMEDELTAS has "last 8 hours", etc.
    end_time = now

    for meas in selected_measurements:
        if meas not in datafetchers:
            continue

        df_meas = datafetchers[meas].fetch_averaged_data(
            recent_data_of="over selected range",
            start_time=start_time,
            end_time=end_time,
            request_type=request_type,
            window_by=window_by,
        )

        if df_meas is None or df_meas.empty:
            continue

        # Ensure time column exists and becomes a DatetimeIndex
        if "time" not in df_meas.columns:
            # If your fetcher returns the time as index already, you can adapt here.
            # For now, skip to avoid breaking resample later.
            continue

        df_meas["time"] = pd.to_datetime(df_meas["time"], errors="coerce", utc=True)
        df_meas = df_meas.dropna(subset=["time"])
        if df_meas.empty:
            continue

        df_meas = df_meas.set_index("time", drop=True).sort_index()

        # Rename columns to "Measurement - Field"
        human_meas = MEASUREMENT_LABELS.get(meas, meas)

        def _rename(col: str) -> str:
            human_field = FIELD_LABELS.get(col, col)
            return f"{human_meas} - {human_field}"

        df_meas = df_meas.rename(columns={col: _rename(col) for col in df_meas.columns})

        combined_df = (
            df_meas if combined_df.empty else combined_df.join(df_meas, how="outer")
        )

    # CRITICAL GUARD: if nothing fetched, return empty BEFORE resample
    if combined_df.empty:
        return combined_df

    # Make sure we can resample
    if not isinstance(combined_df.index, pd.DatetimeIndex):
        # If something unexpected happened, don't crash the app; just return what we have.
        return combined_df

    # Resolve frequency
    # Example: average_range="15 minutes" -> "15min"
    freq = FREQUENCY_TO_TIMEDTA.get(window_by, window_by)

    # If user selected "None", skip resampling
    if freq is None:
        combined_df = combined_df.sort_index()
    else:
        combined_df = combined_df.resample(freq).mean(numeric_only=True)
        if combined_df.empty:
            return combined_df

        combined_df.index = combined_df.index + pd.Timedelta(freq)
        combined_df = combined_df.rename(
            index={combined_df.index[-1]: pd.Timestamp(end_time).round("1min")}
        )

    combined_df = combined_df.sort_index()
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # Convert to IST
    combined_df.index = combined_df.index.tz_convert("Asia/Kolkata")
    combined_df.index.name = "time (IST)"

    return combined_df
