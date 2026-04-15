"""InfluxDB access layer.

Public API
----------
query_builder        Build an InfluxQL query string.
TIMEDELTAS           Preset lookback string → timedelta mapping.
WINDOWING            Human-readable window string → InfluxQL window mapping.
BaseDataFetcher      Low-level per-measurement InfluxDB fetcher.
fetch_online_df      Fetch and merge multiple online measurements.
fetch_offline_data   Fetch offline bucket data.
clean_rm_data        Clean raw-material DataFrame (rename ore prefixes, drop all-NaN groups).
"""

from furnace_data.influx.base import BaseDataFetcher
from furnace_data.influx.offline import clean_rm_data, fetch_offline_data
from furnace_data.influx.online import fetch_online_df
from furnace_data.influx.query import TIMEDELTAS, WINDOWING, query_builder

__all__ = [
    "BaseDataFetcher",
    "clean_rm_data",
    "fetch_offline_data",
    "fetch_online_df",
    "query_builder",
    "TIMEDELTAS",
    "WINDOWING",
]
