"""
offline_fetcher.py — wraps fetch_offline_data for the /data/offline endpoint.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from furnace_data.influx.offline import fetch_offline_data
from furnace_data.config import load_config

log = logging.getLogger(__name__)

config = load_config("setting_ds_dv.yml")

# Maps user-facing report type → InfluxDB measurement name
OFFLINE_REPORT_MAP = {
    "HM_SLAG":          config["offline_measurements"]["HM & Slag"],
    "CHARGE":           config["offline_measurements"]["Charge"],
    "RM_COMPOSITION":   config["offline_measurements"]["Bunker Report"],
    "DPR":              config["offline_measurements"]["DPR"],
}

OFFLINE_BUCKET: str = config["influx_offline"]["database"]

PRESET_TIMEDELTAS = {
    "last 1 day":     timedelta(days=1),
    "last 3 days":    timedelta(days=3),
    "last 1 week":    timedelta(weeks=1),
    "last 2 weeks":   timedelta(weeks=2),
    "last 1 month":   timedelta(days=30),
    "last 3 months":  timedelta(days=90),
    "last 6 months":  timedelta(days=180),
    "last 1 year":    timedelta(days=365),
}


def fetch_offline(
    report_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    preset: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch offline data for a given report type.

    Args:
        report_type: "HM_SLAG" | "CHARGE" | "RM_COMPOSITION" | "DPR"
        start_time:  UTC-aware datetime (ignored when preset is given)
        end_time:    UTC-aware datetime (ignored when preset is given)
        preset:      e.g. "last 1 month" (takes priority over start/end)

    Returns:
        DataFrame indexed by time (UTC).
    """
    measurement = OFFLINE_REPORT_MAP.get(report_type)
    if not measurement:
        raise ValueError(
            f"Unknown report_type '{report_type}'. Valid: {list(OFFLINE_REPORT_MAP)}"
        )

    now = datetime.now(timezone.utc)

    if preset:
        delta = PRESET_TIMEDELTAS.get(preset.lower())
        if delta is None:
            raise ValueError(f"Unknown preset '{preset}'. Valid: {list(PRESET_TIMEDELTAS)}")
        start_time = now - delta
        end_time = now
    elif start_time is None or end_time is None:
        raise ValueError("Provide either 'preset' or both 'start_time' and 'end_time'.")

    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    log.info("Fetching offline %s (%s) from %s to %s", report_type, measurement, start_time, end_time)

    df = fetch_offline_data(
        measurement=measurement,
        time_range=(start_time, end_time),
        database=OFFLINE_BUCKET,
    )

    return df if df is not None else pd.DataFrame()
