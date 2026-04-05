"""
InfluxDB client — ported from src/data_fetchers/base_data_fetcher.py
and src/utils/helper_functions_explorer/data_retrieval.py.

Provides:
  - query_builder()        — builds InfluxQL queries
  - BaseDataFetcher        — low-level InfluxDB fetch
  - fetch_offline_data()   — convenience wrapper for offline bucket queries
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import certifi
import pandas as pd
import pytz
from influxdb_client_3 import InfluxDBClient3, flight_client_options

from app.core.config_loader import load_config

log = logging.getLogger(__name__)

# Load TLS certs once
with open(certifi.where(), "r") as _fh:
    _TLS_CERT = _fh.read()

config = load_config("setting_ds_dv.yml")

WINDOWING = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "10 minutes": "10m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "6 hours": "6h",
    "12 hours": "12h",
    "1 day": "1d",
}


def query_builder(
    measurement: str,
    start: datetime,
    stop: datetime,
    type: str = "average",
    window_by: str = "1h",
) -> str:
    """Build an InfluxQL query for the given measurement and time range."""
    if window_by is not None:
        window_by = WINDOWING.get(window_by, window_by)
        window_by = None if window_by.replace(" ", "").lower() == "none" else window_by

    start_iso = start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    end_iso = stop.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    var_map = config["data_mapping"].get(measurement, {})
    var_map_inv = {v: k for k, v in var_map.items()}

    if type == "windowed-average" and window_by is None:
        type = "ts"

    if type == "ts":
        return (
            f"SELECT * FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time <= '{end_iso}'"
        )
    elif type == "average":
        avg_str = [f"MEAN({col}) AS {col}" for col in var_map_inv.keys()]
        return (
            f"SELECT {', '.join(avg_str)} FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time < '{end_iso}'"
        )
    elif type == "avg-min-max":
        parts = [
            f'MEAN({col}) AS "{col}_mean", MIN({col}) AS "{col}_min", MAX({col}) AS "{col}_max"'
            for col in var_map_inv.keys()
        ]
        return (
            f"SELECT {', '.join(parts)} FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time < '{end_iso}'"
        )
    elif type == "windowed-average":
        avg_str = [f"MEAN({col}) AS {col}" for col in var_map_inv.keys()]
        return (
            f"SELECT {', '.join(avg_str)} FROM {measurement} "
            f"WHERE time >= '{start_iso}' AND time < '{end_iso}' "
            f"GROUP BY time({window_by}) fill(null)"
        )
    else:
        raise ValueError(f"Invalid query type: {type}. Use 'ts', 'average', 'avg-min-max', or 'windowed-average'.")


class BaseDataFetcher:
    """Low-level InfluxDB data fetcher."""

    def __init__(
        self,
        variable_tag: str,
        database: str = "bf2_evonith_raw",
        host: str = "https://eu-central-1-1.aws.cloud2.influxdata.com",
        org: str = "Blast Furnace, Evonith",
        token: str = "INFLUX_ONLINE_TOKEN",
    ):
        self.variable_tag = variable_tag
        self.measurement_type = variable_tag
        self.var_map = config["data_mapping"].get(self.measurement_type, {})

        self.database = database
        self.host = host
        self.org = org
        self.token = token

    def fetch_averaged_data(
        self,
        recent_data_of: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        request_type: str = "ts",
        window_by: str = "1h",
    ) -> pd.DataFrame:
        """Fetch data from InfluxDB for the given time range."""
        now = datetime.now(timezone.utc)
        recent_norm = recent_data_of.strip().lower()

        if recent_norm == "over selected range":
            if not start_time or not end_time:
                raise ValueError("Both start_time and end_time must be provided.")
        else:
            raise ValueError(
                f"Unsupported recent_data_of value: {recent_data_of}. "
                "Use 'over selected range' with explicit start/end times."
            )

        db = self.database
        host = self.host
        org = self.org
        token_val = os.environ.get(self.token, "")

        query = query_builder(
            self.measurement_type,
            start_time,
            end_time,
            type=request_type,
            window_by=window_by,
        )

        client = InfluxDBClient3(
            host=host,
            database=db,
            org=org,
            token=token_val,
            flight_client_options=flight_client_options(tls_root_certs=_TLS_CERT),
        )
        try:
            df = client.query(query=query, mode="pandas", language="influxql")
        finally:
            client.close()

        return df


def fetch_offline_data(
    measurement: str,
    time_range: tuple,
    database: str = "bf2_evonith_offline_utc",
) -> pd.DataFrame:
    """
    Fetch offline data from InfluxDB for an explicit (start, end) time range.
    Both timestamps should be timezone-aware (UTC).
    """
    dfetch = BaseDataFetcher(
        variable_tag=measurement,
        database=database,
        token="INFLUX_OFFLINE_TOKEN",
    )

    start = pd.to_datetime(time_range[0], utc=True)
    end = pd.to_datetime(time_range[1], utc=True)

    df = dfetch.fetch_averaged_data(
        recent_data_of="over selected range",
        start_time=start,
        end_time=end,
        request_type="ts",
        window_by=None,
    )

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")

    return df.sort_index()
