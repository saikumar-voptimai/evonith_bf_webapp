"""Offline InfluxDB bucket fetcher and RM data cleaning.

Provides
--------
fetch_offline_data   Fetch from ``bf2_evonith_offline_utc`` (or any offline bucket).
clean_rm_data        Remove all-NaN ore groups and rename ore prefixes.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Tuple, Union

import pandas as pd

from furnace_data.config import load_config
from furnace_data.influx.base import BaseDataFetcher
from furnace_data.influx.query import TIMEDELTAS

_config = load_config("setting_ds_dv.yml")
_DEFAULT_OFFLINE_DB: str = (
    _config.get("influx_offline", {}) or {}
).get("database", "bf2_evonith_offline_utc")

# Map from generic ore prefix to site-specific name
_ORE_NAME_MAP: dict[str, str] = {
    "ore_1": "NMDC_DONAMALAI",
    "ore_2": "LLOYDS",
    "ore_3": "GEOMIN",
    "ore_4": "JRS_VENTURES",
    "ore_5": "NMDC_ROM",
}


def clean_rm_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw-material DataFrame.

    Drops ore groups where every row is NaN and renames ``ore_N_`` column
    prefixes to their site-specific names (e.g. ``NMDC_DONAMALAI``).

    Args:
        df: Raw DataFrame as returned by :func:`fetch_offline_data`.

    Returns:
        Cleaned DataFrame with renamed ore columns.
    """
    ore_groups: dict[str, list[str]] = {}
    for col in df.columns:
        match = re.match(r"(ore_\d+)_", col)
        if match:
            ore_groups.setdefault(match.group(1), []).append(col)

    drop_cols: list[str] = []
    for ore_key, cols in ore_groups.items():
        if df[cols].isnull().all(axis=1).all():
            drop_cols.extend(cols)

    df_clean = df.drop(columns=drop_cols, errors="ignore")

    rename_map: dict[str, str] = {}
    for col in df_clean.columns:
        for ore_prefix, ore_name in _ORE_NAME_MAP.items():
            if col.startswith(ore_prefix):
                rename_map[col] = col.replace(ore_prefix, ore_name)

    return df_clean.rename(columns=rename_map)


def fetch_offline_data(
    measurement: str,
    time_range: Union[str, Tuple],
    database: str = _DEFAULT_OFFLINE_DB,
) -> pd.DataFrame:
    """Fetch data from an offline InfluxDB bucket.

    Args:
        measurement: Measurement name (e.g. ``"rm_updated_data"``).
        time_range:  One of:

            * A preset string from :data:`~furnace_data.influx.query.TIMEDELTAS`
              (e.g. ``"last 1 month"``).
            * A ``(start, end)`` tuple of UTC-aware datetimes or ISO strings.
            * The string ``"full"`` — fetches from 2023-01-01 to now.

        database:    InfluxDB database / bucket name.  Defaults to the
                     ``influx_offline.database`` config value.

    Returns:
        Time-indexed :class:`pandas.DataFrame` (UTC-aware index).
    """
    fetcher = BaseDataFetcher(
        variable_tag=measurement,
        database=database,
        token="INFLUX_OFFLINE_TOKEN",
    )

    now = datetime.now(timezone.utc)

    if isinstance(time_range, str) and time_range.strip().lower() == "full":
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = now
        label = "over selected range"

    elif isinstance(time_range, tuple) and len(time_range) == 2:
        start = pd.to_datetime(time_range[0], utc=True)
        end = pd.to_datetime(time_range[1], utc=True)
        label = "over selected range"

    else:
        # Preset string — let BaseDataFetcher resolve it
        df = fetcher.fetch_averaged_data(
            recent_data_of=time_range,
            request_type="ts",
            window_by=None,
        )
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
        return df.sort_index()

    df = fetcher.fetch_averaged_data(
        recent_data_of=label,
        start_time=start,
        end_time=end,
        request_type="ts",
        window_by=None,
    )

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")

    return df.sort_index()
