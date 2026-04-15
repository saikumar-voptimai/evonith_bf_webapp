"""Unified BaseDataFetcher for InfluxDB v3 (flight protocol).

This is the single authoritative implementation, merging:
- ``src/data/fetchers/base_data_fetcher.py``  (webapp, richer — wins)
- ``ml_dataset_core/influx_client.py``         (API core, subset)

Both callers now import from here.  The webapp ``src/data/fetchers/`` shim
re-exports this class so existing import paths are not broken immediately.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import certifi
import pandas as pd
import pytz

from influxdb_client_3 import InfluxDBClient3, flight_client_options

from furnace_data.config import load_config
from furnace_data.influx.query import TIMEDELTAS, query_builder

log = logging.getLogger(__name__)

# Load TLS certificate once at module import.
with open(certifi.where(), "r") as _fh:
    _TLS_CERT = _fh.read()

_config = load_config("setting_ds_dv.yml")


class BaseDataFetcher:
    """Low-level InfluxDB data fetcher.

    Supports preset lookback strings (e.g. ``"last 8 hours"``), explicit
    ``(start_time, end_time)`` ranges, and ``"over selected range"`` for
    explicit ranges.

    Args:
        variable_tag: Key in ``data_mapping`` / ``data_tags`` config sections
            (e.g. ``"process_params"``).
        debug:        If ``True``, ``fetch_averaged_data`` returns dummy data
            instead of hitting InfluxDB.  Intended for local UI development.
        source:       ``"live"`` or ``"historical"``.  Subclasses may use this
            to switch between fetch strategies.
        database:     InfluxDB database / bucket name.
        host:         InfluxDB host URL.
        org:          InfluxDB organisation name.
        token:        Environment variable name whose value is the InfluxDB token.
    """

    def __init__(
        self,
        variable_tag: str,
        debug: bool = False,
        source: str = "live",
        database: str = "bf2_evonith_raw",
        host: str = "https://eu-central-1-1.aws.cloud2.influxdata.com",
        org: str = "Blast Furnace, Evonith",
        token: str = "INFLUX_ONLINE_TOKEN",
    ) -> None:
        self.debug = debug
        self.source = source.strip().lower()
        self.timezone = pytz.timezone("Asia/Kolkata")
        self.variable_tag = variable_tag
        self.variables: List[str] = _config.get("data_tags", {}).get(variable_tag, [])
        self.measurement_type = variable_tag
        self.var_map: Dict[str, str] = _config.get("data_mapping", {}).get(
            self.measurement_type, {}
        )
        self.database = database
        self.host = host
        self.org = org
        self.token = token

    # ------------------------------------------------------------------
    # Main fetch method
    # ------------------------------------------------------------------

    def fetch_averaged_data(
        self,
        recent_data_of: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        request_type: str = "ts",
        window_by: Optional[str] = "1h",
    ) -> pd.DataFrame:
        """Fetch data from InfluxDB for the requested time window.

        Args:
            recent_data_of: One of the preset keys in :data:`TIMEDELTAS`
                (e.g. ``"last 8 hours"``), ``"over selected range"`` (requires
                *start_time* and *end_time*), or ``"live"`` (raises
                :class:`NotImplementedError` — subclass override needed).
            start_time:  UTC-aware start time (required for ``"over selected range"``).
            end_time:    UTC-aware end time (required for ``"over selected range"``).
            request_type: InfluxQL query type; see :func:`~furnace_data.influx.query.query_builder`.
            window_by:   Aggregation window for ``"windowed-average"`` type.

        Returns:
            :class:`pandas.DataFrame` with a ``"time"`` column (UTC-aware).
        """
        recent_norm = recent_data_of.strip().lower()

        if recent_norm == "live":
            return self.fetch_live_data()

        now = datetime.now(timezone.utc)

        if recent_norm == "over selected range":
            if not start_time or not end_time:
                raise ValueError(
                    "Both start_time and end_time must be provided for 'over selected range'."
                )
        elif recent_norm in TIMEDELTAS:
            start_time = now - TIMEDELTAS[recent_norm]
            end_time = now
        else:
            raise ValueError(
                f"Unrecognised recent_data_of value: {recent_data_of!r}. "
                f"Use a preset key from TIMEDELTAS or 'over selected range'."
            )

        token_val = os.environ.get(self.token, "")
        query = query_builder(
            self.measurement_type,
            start_time,
            end_time,
            type=request_type,
            window_by=window_by,
        )

        client = InfluxDBClient3(
            host=self.host,
            database=self.database,
            org=self.org,
            token=token_val,
            flight_client_options=flight_client_options(tls_root_certs=_TLS_CERT),
        )
        try:
            df = client.query(query=query, mode="pandas", language="influxql")
        finally:
            client.close()

        return df

    # ------------------------------------------------------------------
    # Optional / legacy hooks
    # ------------------------------------------------------------------

    def fetch_live_data(self) -> pd.DataFrame:
        """Fetch the most recent data point (live mode).

        Override in subclasses if live-mode fetching is needed.
        The base implementation raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "fetch_live_data() is not implemented in BaseDataFetcher. "
            "Use fetch_averaged_data(recent_data_of='last 1 minute') instead."
        )

    def _get_variable_names(self) -> List[str]:
        return self.variables

    def dataframe_to_dict(self, df: pd.DataFrame) -> Dict:
        """Convert a single-row DataFrame to a dict (keyed by InfluxDB field names)."""
        return df.to_dict(orient="list")
