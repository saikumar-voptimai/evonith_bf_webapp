"""Lag-aware incremental CSV update manager for the static furnace dataset.

Design
------
``confirmed_end = raw_end - offline_lag_days``

Rows up to *confirmed_end* are frozen - offline data (RM composition, HM/Slag)
is stable by then.  Rows from *confirmed_end* onward are re-fetched on every
run to capture delayed manual-entry data (typically 2-3 days late).

Provides
--------
CacheMeta            Dataclass for cache-metadata JSON sidecar.
StaticDatasetManager Manager that incrementally updates and rotates versioned CSVs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from furnace_data.dataset.fetcher import DatasetFetcher
from furnace_data.dataset.cleaning import DataCleaner, build_default_config

log = logging.getLogger(__name__)

_META_FILE = "cache_meta.json"
_CSV_PREFIX = "furnace_dataset_"
_MAX_VERSIONS = 3  # module-level default; overridden per-instance


# ---------------------------------------------------------------------------
# Cache metadata
# ---------------------------------------------------------------------------


@dataclass
class CacheMeta:
    """JSON-serialisable sidecar for the static dataset cache.

    Attributes:
        version:          Schema version integer.
        rm_choice:        ``"charge"`` or ``"dpr"``.
        data_start:       ISO date string of the first row in the CSV.
        confirmed_end:    ISO date string; rows up to here are frozen.
        raw_end:          ISO date string of the last date actually fetched.
        last_updated:     ISO datetime string of the last update run.
        offline_lag_days: Lag used to compute *confirmed_end*.
        rows:             Row count of the saved CSV.
        columns:          Column count of the saved CSV.
        csv_file:         Filename (not full path) of the current active CSV.
    """

    version: int = 1
    rm_choice: str = "charge"
    data_start: str = ""
    confirmed_end: str = ""
    raw_end: str = ""
    last_updated: str = ""
    offline_lag_days: int = 3
    rows: int = 0
    columns: int = 0
    csv_file: str = ""

    @property
    def confirmed_end_date(self) -> Optional[date]:
        """Return *confirmed_end* as a :class:`datetime.date`, or ``None``."""
        return date.fromisoformat(self.confirmed_end) if self.confirmed_end else None

    @property
    def raw_end_date(self) -> Optional[date]:
        """Return *raw_end* as a :class:`datetime.date`, or ``None``."""
        return date.fromisoformat(self.raw_end) if self.raw_end else None

    @property
    def data_start_date(self) -> Optional[date]:
        """Return *data_start* as a :class:`datetime.date`, or ``None``."""
        return date.fromisoformat(self.data_start) if self.data_start else None


def _load_meta(static_dir: Path) -> Optional[CacheMeta]:
    meta_path = static_dir / _META_FILE
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            d = json.load(f)
        return CacheMeta(**{k: v for k, v in d.items() if k in CacheMeta.__dataclass_fields__})
    except Exception as e:
        log.warning("Could not read cache_meta.json: %s", e)
        return None


def _save_meta(static_dir: Path, meta: CacheMeta) -> None:
    meta_path = static_dir / _META_FILE
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _versioned_filename() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_CSV_PREFIX}{ts}.csv"


def _rotate_csvs(static_dir: Path, keep: int) -> None:
    """Delete the oldest versioned CSV files, retaining only *keep* most recent."""
    csvs = sorted(
        static_dir.glob(f"{_CSV_PREFIX}*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    for old in csvs[:-keep]:
        try:
            old.unlink()
            log.info("Rotated out old CSV: %s", old.name)
        except Exception as e:
            log.warning("Could not delete %s: %s", old.name, e)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = df.columns.str.strip()
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    return df.set_index(time_col).sort_index()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class StaticDatasetManager:
    """Manages the static furnace dataset with lag-aware incremental updates.

    On each :meth:`update_static` call:

    1. Load the existing versioned CSV + ``cache_meta.json`` (or a legacy
       bootstrap CSV on first run).
    2. Compute ``confirmed_end = raw_end - offline_lag_days``.
    3. Keep the frozen rows (index ≤ *confirmed_end*) unchanged.
    4. Re-fetch *confirmed_end* -> today to pick up delayed offline data.
    5. Merge (new rows win on overlap), optionally clean with
       :class:`~furnace_data.dataset.cleaning.DataCleaner`.
    6. Call :meth:`save` to write a new versioned CSV and rotate old ones.
    7. Write the updated ``cache_meta.json``.

    Args:
        static_dir:       Directory where versioned CSVs and ``cache_meta.json``
                          are stored.  Created if it does not exist.
        offline_lag_days: Days of lag before offline data is considered stable.
        max_versions:     Maximum number of versioned CSV files to retain.
        legacy_csv_path:  Optional path to a pre-existing CSV to bootstrap from
                          when no versioned CSV exists yet.
    """

    _DEFAULT_START = date(2023, 1, 1)

    def __init__(
        self,
        static_dir: str | Path,
        offline_lag_days: int = 3,
        max_versions: int = _MAX_VERSIONS,
        legacy_csv_path: Optional[str | Path] = None,
    ) -> None:
        self.static_dir = Path(static_dir)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.offline_lag_days = offline_lag_days
        self.max_versions = max_versions
        self.legacy_csv_path = Path(legacy_csv_path) if legacy_csv_path else None
        self.fetcher = DatasetFetcher()
        self.cleaner = DataCleaner(build_default_config())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_static(
        self,
        rm_choice: str,
        reprocess_from: Optional[date] = None,
        apply_cleaning: bool = True,
    ) -> pd.DataFrame:
        """Perform a smart incremental update of the static dataset.

        Args:
            rm_choice:       ``"RM Charge"`` or ``"RM DPR"``.
            reprocess_from:  If set, discard all data from this date onward and
                             re-fetch.  Useful to force a manual backfill.
            apply_cleaning:  Whether to run :class:`DataCleaner` on the new slice.

        Returns:
            The final merged :class:`pandas.DataFrame` (not yet saved - call
            :meth:`save` separately if you want to persist it).
        """
        meta = _load_meta(self.static_dir)
        existing = self._load_existing(meta)

        today = date.today()
        rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

        # ---- Manual reprocess override ----
        if reprocess_from:
            log.info("Manual reprocess from %s", reprocess_from)
            if not existing.empty:
                existing = existing.loc[existing.index.date < reprocess_from]
            fetch_start = reprocess_from
            fetch_end = today
            frozen = existing

        # ---- Normal smart incremental ----
        else:
            fetch_start, fetch_end, frozen = self._compute_fetch_window(
                existing, meta, today
            )
            if fetch_start is None:
                log.info("Nothing to fetch - dataset is already up to date.")
                return existing

        log.info("Fetching %s -> %s (rm=%s)", fetch_start, fetch_end, rm_mode)

        new_df = self.fetcher.get_dataset(
            start_date=fetch_start,
            end_date=fetch_end,
            rm_choice=rm_choice,
            cache_override=True,
        )

        if new_df.empty:
            log.warning("Fetch returned empty DataFrame.")
            return existing

        if apply_cleaning:
            log.info("Cleaning fetched slice (%d rows)...", len(new_df))
            new_df = self.cleaner.clean(new_df)

        # Merge: new rows win on overlap
        final_df = new_df.combine_first(frozen).sort_index().dropna(how="all")
        log.info("Final dataset: %d rows, %d cols", len(final_df), len(final_df.columns))

        return final_df

    def save(self, df: pd.DataFrame, rm_choice: str) -> Path:
        """Save *df* as a new versioned CSV and update the metadata sidecar.

        Rotates old CSV files so that at most :attr:`max_versions` files are kept.

        Args:
            df:        The DataFrame to persist.
            rm_choice: ``"RM Charge"`` or ``"RM DPR"`` (stored in metadata).

        Returns:
            :class:`pathlib.Path` of the newly written CSV file.
        """
        filename = _versioned_filename()
        csv_path = self.static_dir / filename

        df_out = df.copy()
        df_out.index.name = None
        df_out.to_csv(csv_path, index=True)
        log.info("Saved %s (%d rows, %d cols)", filename, len(df), len(df.columns))

        _rotate_csvs(self.static_dir, self.max_versions)

        today = date.today()
        raw_end = today
        confirmed_end = today - timedelta(days=self.offline_lag_days)
        data_start = df.index.min().date() if not df.empty else today

        meta = CacheMeta(
            rm_choice="charge" if rm_choice == "RM Charge" else "dpr",
            data_start=data_start.isoformat(),
            confirmed_end=confirmed_end.isoformat(),
            raw_end=raw_end.isoformat(),
            last_updated=datetime.now().isoformat(timespec="seconds"),
            offline_lag_days=self.offline_lag_days,
            rows=len(df),
            columns=len(df.columns),
            csv_file=filename,
        )
        _save_meta(self.static_dir, meta)

        return csv_path

    def get_meta(self) -> Optional[CacheMeta]:
        """Return the current :class:`CacheMeta`, or ``None`` if not found."""
        return _load_meta(self.static_dir)

    def current_csv_path(self) -> Optional[Path]:
        """Return the path of the active CSV file, or ``None`` if it doesn't exist."""
        meta = _load_meta(self.static_dir)
        if not meta or not meta.csv_file:
            return None
        p = self.static_dir / meta.csv_file
        return p if p.exists() else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_existing(self, meta: Optional[CacheMeta]) -> pd.DataFrame:
        """Load the active versioned CSV, falling back to any CSV or legacy path."""
        if meta and meta.csv_file:
            active = self.static_dir / meta.csv_file
            if active.exists():
                log.info("Loading active cache: %s", active.name)
                return _read_csv(active)

        # Any versioned CSV in the folder (fallback if meta is stale)
        csvs = sorted(
            self.static_dir.glob(f"{_CSV_PREFIX}*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if csvs:
            log.info("Loading most recent CSV (no meta pointer): %s", csvs[-1].name)
            return _read_csv(csvs[-1])

        # Legacy bootstrap
        if self.legacy_csv_path and self.legacy_csv_path.exists():
            log.info("Bootstrapping from legacy CSV: %s", self.legacy_csv_path)
            return _read_csv(self.legacy_csv_path)

        log.info("No existing dataset found - will do a full fetch.")
        return pd.DataFrame()

    def _compute_fetch_window(
        self,
        existing: pd.DataFrame,
        meta: Optional[CacheMeta],
        today: date,
    ) -> tuple[Optional[date], date, pd.DataFrame]:
        """Compute the fetch window and frozen slice.

        Returns:
            A 3-tuple ``(fetch_start, fetch_end, frozen_df)``.
            *fetch_start* is ``None`` when the dataset is already up to date.
        """
        if existing.empty:
            return self._DEFAULT_START, today, pd.DataFrame()

        raw_end = meta.raw_end_date if meta else existing.index.max().date()
        confirmed_end = raw_end - timedelta(days=self.offline_lag_days)

        if confirmed_end >= today - timedelta(days=self.offline_lag_days) and raw_end >= today:
            return None, today, existing

        frozen = existing.loc[existing.index.date <= confirmed_end]
        fetch_start = confirmed_end  # inclusive re-fetch to capture delayed offline data
        fetch_end = today

        log.info(
            "Cache: raw_end=%s, confirmed_end=%s -> re-fetch %s -> %s",
            raw_end,
            confirmed_end,
            fetch_start,
            fetch_end,
        )

        return fetch_start, fetch_end, frozen
