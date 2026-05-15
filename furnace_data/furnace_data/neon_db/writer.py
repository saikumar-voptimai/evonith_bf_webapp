"""Write helpers for the offline_feed.historical_static_ml_dataset table.

Only two public functions — append rows and delete rows.  Nothing else.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import inspect as sa_inspect, text

from furnace_data.relational.engine import build_relational_engine

_SCHEMA = "offline_feed"
_TABLE  = "historical_static_ml_dataset"
_LOCAL_TZ = ZoneInfo("Asia/Kolkata")
_UTC      = timezone.utc

log = logging.getLogger(__name__)


def _ist_date_to_utc(d: date) -> datetime:
    """Convert an IST calendar date (start of day) to a UTC-aware datetime."""
    return datetime.combine(d, time.min, tzinfo=_LOCAL_TZ).astimezone(_UTC)


def write_to_static_table(
    df: pd.DataFrame,
    database_url: str | None = None,
    chunksize: int = 500,
) -> int:
    """Append rows to ``offline_feed.historical_static_ml_dataset``.

    The DataFrame must have:
    - An IST tz-naive DatetimeIndex (will be converted to UTC before write).
    - Column names matching ML-dataset snake_case names (rename_dict keys).

    Columns not present in the DB table are silently dropped.

    Args:
        df:            DataFrame to write (ML column names, IST-naive index).
        database_url:  Optional explicit PostgreSQL URL. Falls back to
                       ``DATABASE_URL`` env var.
        chunksize:     Rows per INSERT batch.

    Returns:
        Number of rows written.
    """
    if df is None or df.empty:
        return 0

    engine = build_relational_engine(database_url)
    try:
        available_cols = {
            c["name"]
            for c in sa_inspect(engine).get_columns(_TABLE, schema=_SCHEMA)
        }

        df_write = df.copy()

        # Convert IST-naive index to UTC-aware
        if df_write.index.tz is None:
            df_write.index = df_write.index.tz_localize(_LOCAL_TZ).tz_convert(_UTC)
        else:
            df_write.index = df_write.index.tz_convert(_UTC)
        df_write.index.name = "date_time"

        # Drop any columns not in the target table schema
        keep = [c for c in df_write.columns if c in available_cols]
        df_write = df_write[keep]

        if df_write.empty or not keep:
            log.warning("No matching columns found in %s.%s — nothing written.", _SCHEMA, _TABLE)
            return 0

        df_write.to_sql(
            _TABLE,
            engine,
            schema=_SCHEMA,
            if_exists="append",
            index=True,
            index_label="date_time",
            method="multi",
            chunksize=chunksize,
        )
        log.info("Wrote %d rows to %s.%s.", len(df_write), _SCHEMA, _TABLE)
        return len(df_write)
    finally:
        engine.dispose()


def delete_from_static_table(
    from_date: date,
    database_url: str | None = None,
) -> int:
    """Delete rows from ``offline_feed.historical_static_ml_dataset`` where
    ``date_time >= from_date`` (start of that IST calendar day in UTC).

    Args:
        from_date:     Inclusive start date (IST) of rows to delete.
        database_url:  Optional explicit PostgreSQL URL.

    Returns:
        Number of rows deleted.
    """
    cutoff_utc = _ist_date_to_utc(from_date)

    engine = build_relational_engine(database_url)
    try:
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f"DELETE FROM {_SCHEMA}.{_TABLE}"
                    " WHERE date_time >= :cutoff"
                ),
                {"cutoff": cutoff_utc},
            )
            deleted = result.rowcount
        log.info(
            "Deleted %d rows from %s.%s (date_time >= %s UTC).",
            deleted, _SCHEMA, _TABLE, cutoff_utc.isoformat(),
        )
        return deleted
    finally:
        engine.dispose()
