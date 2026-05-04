"""Neon/PostgreSQL offline fetch wrapper for the /data/offline endpoint."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from furnace_data.neon_db.offline import (
    NEON_OFFLINE_REPORT_MAP,
    NEON_OFFLINE_TABLES,
    fetch_offline_data,
)

log = logging.getLogger(__name__)

PRESET_TIMEDELTAS = {
    "last 1 day": timedelta(days=1),
    "last 3 days": timedelta(days=3),
    "last 1 week": timedelta(weeks=1),
    "last 2 weeks": timedelta(weeks=2),
    "last 1 month": timedelta(days=30),
    "last 3 months": timedelta(days=90),
    "last 6 months": timedelta(days=180),
    "last 1 year": timedelta(days=365),
}


def fetch_neon_offline(
    report_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    preset: Optional[str] = None,
    table_name: Optional[str] = None,
    query_type: str = "ts",
    window: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch offline data from Neon for a report type or explicit table."""
    table = table_name or NEON_OFFLINE_REPORT_MAP.get(report_type)
    if not table or table not in NEON_OFFLINE_TABLES:
        raise ValueError(
            f"Unknown Neon offline table/report '{table_name or report_type}'. "
            f"Valid tables: {sorted(NEON_OFFLINE_TABLES)}"
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

    log.info(
        "Fetching Neon offline %s (%s) from %s to %s as %s",
        report_type,
        table,
        start_time,
        end_time,
        query_type,
    )
    return fetch_offline_data(
        table_name=table,
        time_range=(start_time, end_time),
        query_type=query_type,
        window=window,
    )
