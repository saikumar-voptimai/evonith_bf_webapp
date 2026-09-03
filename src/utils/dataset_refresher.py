"""Hourly background refresh for the published furnace-dataset fallback."""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

REFRESH_THRESHOLD_HOURS: int = 1
OFFLINE_LAG_DAYS: int = 2  # offline DB data has a ~24-48 h ingestion lag

_lock = threading.Lock()
_refreshing: bool = False
_dataset_version: int = 0


def maybe_refresh(config: dict, rm_choice: str = "Full") -> bool:
    """Refresh the local full-dataset cache in the background when stale."""
    global _refreshing

    if _refreshing:
        return True

    from data.ml.static_csv import get_static_dataset_path
    from data.ml.static_dataset_manager import StaticDatasetManager

    static_path = get_static_dataset_path(config.get("DATA"))
    remote_url = str(config.get("DATA_URL", "") or "").strip() or None
    meta = StaticDatasetManager(static_path, remote_url=remote_url).get_meta()

    if not _is_stale(meta):
        return False

    if not _lock.acquire(blocking=False):
        return _refreshing

    _refreshing = True
    last = meta.last_updated if meta else "never"
    log.info(
        "Stale static dataset cache detected (last_updated=%s); refreshing",
        last,
    )
    threading.Thread(
        target=_do_refresh,
        args=(static_path, rm_choice, remote_url),
        name="dataset-refresher",
        daemon=True,
    ).start()
    return True


def get_version() -> int:
    """Return the current local cache version counter."""
    return _dataset_version


def _is_stale(meta) -> bool:
    if meta is None or not meta.last_updated:
        return True
    try:
        last = datetime.fromisoformat(meta.last_updated)
        if (datetime.now() - last).total_seconds() / 3600 >= REFRESH_THRESHOLD_HOURS:
            return True
    except ValueError:
        return True
    # Date-based staleness: only fire if the last refresh was at least half the
    # threshold ago.  This prevents an immediate re-trigger loop where the refresh
    # just completed but the data still ends before today (normal offline ingestion lag).
    try:
        age_hours = (datetime.now() - last).total_seconds() / 3600
        if age_hours >= REFRESH_THRESHOLD_HOURS / 2 and meta.confirmed_end:
            end = date.fromisoformat(meta.confirmed_end)
            if end < date.today() - timedelta(days=OFFLINE_LAG_DAYS):
                return True
    except (ValueError, AttributeError):
        pass
    return False


def _do_refresh(
    static_path: Path, rm_choice: str, remote_url: str | None = None
) -> None:
    global _refreshing, _dataset_version
    try:
        from data.ml.static_dataset_manager import StaticDatasetManager

        manager = StaticDatasetManager(static_path, remote_url=remote_url)
        df = manager.update_static(rm_choice=rm_choice)
        if df.empty:
            log.warning("Static dataset refresh returned no rows; skipping save")
            return

        saved_path = manager.save(df)
        _dataset_version += 1
        log.info(
            "Static dataset cache refreshed: %d rows saved to %s",
            len(df),
            saved_path.name,
        )
    except Exception as exc:
        log.error("Static dataset refresh failed: %s", exc, exc_info=True)
    finally:
        _refreshing = False
        _lock.release()
