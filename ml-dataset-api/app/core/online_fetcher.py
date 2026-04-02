"""
online_fetcher.py — wraps BaseDataFetcher for the /data/online endpoint.

Supports fetching 1–6 measurements, outer-joining them on the time index.
All measurements share the same time window and query type.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import pytz

from app.core.influx_client import BaseDataFetcher
from app.core.config_loader import load_config

log = logging.getLogger(__name__)

config = load_config("setting_ds_dv.yml")

# All valid online measurements
ONLINE_MEASUREMENTS: List[str] = [
    "process_params",
    "temperature_profile",
    "heatload_delta_t",
    "cooling_water",
    "delta_t",
    "miscellaneous",
]

PRESET_TIMEDELTAS = {
    "last 1 minute":  timedelta(minutes=1),
    "last 5 minutes": timedelta(minutes=5),
    "last 15 minutes": timedelta(minutes=15),
    "last 30 minutes": timedelta(minutes=30),
    "last 1 hour":    timedelta(hours=1),
    "last 6 hours":   timedelta(hours=6),
    "last 8 hours":   timedelta(hours=8),
    "last 12 hours":  timedelta(hours=12),
    "last 1 day":     timedelta(days=1),
    "last 3 days":    timedelta(days=3),
    "last 1 week":    timedelta(weeks=1),
    "last 2 weeks":   timedelta(weeks=2),
    "last 1 month":   timedelta(days=30),
    "last 2 months":  timedelta(days=60),
    "last 3 months":  timedelta(days=90),
}

_IST = pytz.timezone("Asia/Kolkata")


def list_measurements() -> dict:
    """Return all measurements and their field names from YAML config."""
    data_mapping = config.get("data_mapping", {})
    return {
        m: list(data_mapping.get(m, {}).values())
        for m in ONLINE_MEASUREMENTS
    }


def fetch_online(
    measurements: List[str],
    query_type: str,
    window: Optional[str],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    preset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch one or more measurements from the online InfluxDB bucket and
    outer-join them on the time index.

    Args:
        measurements: subset of ONLINE_MEASUREMENTS
        query_type:   "ts" | "windowed-average" | "average" | "avg-min-max"
        window:       e.g. "1h", "15m" (only for windowed-average)
        start_time:   UTC-aware datetime (ignored when preset is given)
        end_time:     UTC-aware datetime (ignored when preset is given)
        preset:       e.g. "last 8 hours" (takes priority over start/end)

    Returns:
        DataFrame with IST-indexed timestamps, columns from all requested measurements.
    """
    invalid = [m for m in measurements if m not in ONLINE_MEASUREMENTS]
    if invalid:
        raise ValueError(f"Unknown measurement(s): {invalid}. Valid: {ONLINE_MEASUREMENTS}")

    # Resolve time window
    now = datetime.now(timezone.utc)
    if preset:
        delta = PRESET_TIMEDELTAS.get(preset.lower())
        if delta is None:
            raise ValueError(f"Unknown preset '{preset}'. Valid: {list(PRESET_TIMEDELTAS)}")
        start_time = now - delta
        end_time = now
    elif start_time is None or end_time is None:
        raise ValueError("Provide either 'preset' or both 'start_time' and 'end_time'.")

    # Ensure UTC-aware
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    frames: List[pd.DataFrame] = []

    for measurement in measurements:
        fetcher = BaseDataFetcher(
            variable_tag=measurement,
            database="bf2_evonith_raw",
            token="INFLUX_ONLINE_TOKEN",
        )
        try:
            df = fetcher.fetch_averaged_data(
                recent_data_of="over selected range",
                start_time=start_time,
                end_time=end_time,
                request_type=query_type,
                window_by=window,
            )
            if df is not None and not df.empty:
                # Normalise time index to IST, tz-naive
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], utc=True)
                    df = df.set_index("time")
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(_IST).tz_localize(None)
                df.index.name = "time"
                # Drop InfluxDB metadata columns that clash on multi-measurement joins
                df = df.drop(columns=[c for c in df.columns if c.startswith("iox::")], errors="ignore")
                frames.append(df)
                log.info("Fetched %s: %d rows, %d cols", measurement, len(df), len(df.columns))
        except Exception as e:
            log.warning("Failed to fetch %s: %s", measurement, e)

    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0].sort_index()

    result = frames[0]
    for df in frames[1:]:
        result = result.join(df, how="outer")

    return result.sort_index()
