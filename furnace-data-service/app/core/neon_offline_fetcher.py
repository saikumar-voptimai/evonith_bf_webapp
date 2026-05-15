"""Neon/PostgreSQL offline fetch wrapper for /data/offline endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from furnace_data.neon_db.offline import (
    NEON_OFFLINE_REPORT_MAP,
    fetch_offline_data,
    fetch_offline_report,
    resolve_neon_table_name,
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


def _resolve_time_range(
    *,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    preset: Optional[str],
) -> tuple[datetime, datetime]:
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
    return start_time, end_time


def fetch_neon_offline(
    report_type: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    preset: Optional[str] = None,
    table_name: Optional[str] = None,
    query_type: str = "ts",
    window: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch offline data from Neon by logical report or explicit table."""
    start_time, end_time = _resolve_time_range(
        start_time=start_time,
        end_time=end_time,
        preset=preset,
    )

    if table_name:
        try:
            resolved_table = resolve_neon_table_name(table_name)
        except ValueError as exc:
            raise ValueError(
                str(exc)
            ) from exc
        log.info(
            "Fetching Neon offline table %s from %s to %s as %s",
            resolved_table,
            start_time,
            end_time,
            query_type,
        )
        return fetch_offline_data(
            table_name=resolved_table,
            time_range=(start_time, end_time),
            query_type=query_type,
            window=window,
        )

    if report_type not in NEON_OFFLINE_REPORT_MAP:
        raise ValueError(
            f"Unknown Neon offline report '{report_type}'. "
            f"Valid reports: {sorted(NEON_OFFLINE_REPORT_MAP)}"
        )

    log.info(
        "Fetching Neon offline report %s from %s to %s as %s",
        report_type,
        start_time,
        end_time,
        query_type,
    )
    return fetch_offline_report(
        report_type=report_type,
        time_range=(start_time, end_time),
        query_type=query_type,
        window=window,
    )
