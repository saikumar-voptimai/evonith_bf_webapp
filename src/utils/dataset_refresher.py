"""Background stale-while-revalidate refresh for the static furnace dataset.

The strategy is simple:

1. On page load call :func:`maybe_refresh`.  It reads ``cache_meta.json``
   (< 1 ms) and, if the dataset is stale, spawns a daemon thread.
2. The thread fetches the incremental delta from InfluxDB, saves a new
   versioned CSV, then increments :data:`_dataset_version`.
3. Pages detect the version bump on the next re-run and clear any
   session-state objects that hold a stale copy of the data (e.g.
   ``dfprocessor``, ``fm_ml_df_cache``).

Usage::

    # top of any page that reads furnace_dataset.csv
    from utils.dataset_refresher import maybe_refresh, get_version
    _updating = maybe_refresh(config)
    if _updating:
        st.sidebar.caption("⏳ Refreshing dataset…")

    # reset session-cached objects when a new version lands
    if st.session_state.get("_ds_version") != get_version():
        st.session_state.pop("dfprocessor", None)   # Recommendations
        st.session_state.pop("fm_ml_df_cache", None) # FurnaceMind
        st.session_state["_ds_version"] = get_version()
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ── tuneable constant ──────────────────────────────────────────────────────
# Trigger a refresh when the last successful update was more than this many
# hours ago.  Offline data (RM composition, HM/slag) matures with a ~3-day
# lag, so 6 h is generous but ensures the CSV never drifts by more than a
# working shift relative to what is actually available.
REFRESH_THRESHOLD_HOURS: int = 6

# ── module-level singleton state ───────────────────────────────────────────
# One instance per Streamlit worker process; not shared across processes but
# that is fine — worst case two processes update in parallel and one is a no-op
# because the data is already fresh when it runs.
_lock = threading.Lock()
_refreshing: bool = False
_dataset_version: int = 0  # bumped after each successful update


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maybe_refresh(config: dict, rm_choice: str = "RM Charge") -> bool:
    """Check staleness and fire a background incremental update if needed.

    Args:
        config:    The webapp's ``setting_ds_dv.yml`` config dict.  Must
                   contain a ``"DATA"`` key pointing to the CSV path.
        rm_choice: ``"RM Charge"`` (default) or ``"RM DPR"``.

    Returns:
        ``True`` while a refresh is in progress, ``False`` otherwise.
    """
    global _refreshing

    if _refreshing:
        return True

    from data.ml.static_csv import get_static_dataset_path
    from data.ml.static_dataset_manager import StaticDatasetManager

    static_path = get_static_dataset_path(config.get("DATA"))
    meta = StaticDatasetManager(static_path).get_meta()

    if not _is_stale(meta):
        return False

    # Non-blocking acquire — if another thread already owns the lock we just
    # skip; _refreshing is True in that case and we return it above.
    if not _lock.acquire(blocking=False):
        return _refreshing

    _refreshing = True
    last = meta.last_updated if meta else "never"
    log.info(
        "Stale dataset detected (last_updated=%s) — starting background refresh",
        last,
    )

    threading.Thread(
        target=_do_refresh,
        args=(static_path, rm_choice),
        name="dataset-refresher",
        daemon=True,
    ).start()
    return True


def get_version() -> int:
    """Return the current dataset version counter.

    Pages store this in ``st.session_state["_ds_version"]`` and compare on
    each re-run to detect when new data has landed.
    """
    return _dataset_version


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _is_stale(meta) -> bool:
    """Return ``True`` when the dataset needs a refresh."""
    if meta is None or not meta.last_updated:
        return True
    try:
        last = datetime.fromisoformat(meta.last_updated)
    except ValueError:
        return True
    age_hours = (datetime.now() - last).total_seconds() / 3600
    return age_hours >= REFRESH_THRESHOLD_HOURS


def _do_refresh(static_path: Path, rm_choice: str) -> None:
    """Perform the incremental InfluxDB fetch, persist the new CSV, and
    bump the version counter so pages know to reload their cached objects."""
    global _refreshing, _dataset_version
    try:
        from data.ml.static_dataset_manager import StaticDatasetManager

        mgr = StaticDatasetManager(static_path)
        log.info("Background dataset refresh: fetching incremental delta…")
        df = mgr.update_static(rm_choice)

        if df.empty:
            log.warning("Refresh returned an empty DataFrame — skipping save.")
            return

        mgr.save(df)
        log.info(
            "Dataset refresh complete — %d rows saved to %s",
            len(df),
            static_path,
        )

        # Invalidate the @st.cache_data layer so the next page re-run reads
        # the freshly written CSV from disk instead of the in-memory cache.
        try:
            from data.ml.static_csv import load_static_dataset
            load_static_dataset.clear()
        except Exception:
            pass  # not running inside Streamlit (e.g., unit tests)

        # Bump version — pages detect this on the next re-run and clear any
        # session-state objects that hold a stale copy of the dataframe.
        _dataset_version += 1
        log.info("Dataset version bumped to %d", _dataset_version)

    except Exception as exc:
        log.error("Dataset refresh failed: %s", exc, exc_info=True)
    finally:
        _refreshing = False
        _lock.release()
