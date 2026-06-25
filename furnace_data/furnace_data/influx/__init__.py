"""InfluxDB access layer.

Public API
----------
query_builder        Build an InfluxQL query string.
TIMEDELTAS           Preset lookback string → timedelta mapping.
WINDOWING            Human-readable window string → InfluxQL window mapping.
BaseDataFetcher      Low-level per-measurement InfluxDB fetcher.
fetch_online_df      Fetch and merge multiple online measurements.
"""

from furnace_data.influx.base import BaseDataFetcher
from furnace_data.influx.online import fetch_online_df
from furnace_data.influx.query import (
    TIMEDELTAS,
    WINDOWING,
    human_labels,
    influx_fields,
    query_builder,
)

__all__ = [
    "BaseDataFetcher",
    "fetch_online_df",
    "query_builder",
    "human_labels",
    "influx_fields",
    "TIMEDELTAS",
    "WINDOWING",
]
