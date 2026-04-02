"""
StaticDatasetManager — lag-aware caching, CSV rotation, legacy bootstrap.

Key design:
  confirmed_end = raw_end - offline_lag_days
  - Rows up to confirmed_end are frozen (offline data is stable by then).
  - Rows from confirmed_end onward are re-fetched every run to pick up delayed
    offline data (RM composition, HM/slag etc. typically arrive 2–3 days late).
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.dataset_fetcher import MlDatasetFetcher
from app.core.data_cleaning import DataCleaner, build_default_config

log = logging.getLogger(__name__)

_META_FILE = "cache_meta.json"
_CSV_PREFIX = "ml_dataset_"
_MAX_VERSIONS = 3   # module-level default; overridden by settings at call time


# ---------------------------------------------------------------------------
# Cache metadata
# ---------------------------------------------------------------------------

@dataclass
class CacheMeta:
    version: int = 1
    rm_choice: str = "charge"
    data_start: str = ""          # ISO date string
    confirmed_end: str = ""       # rows up to here are frozen
    raw_end: str = ""             # last date actually fetched (may be partial)
    last_updated: str = ""        # ISO datetime string
    offline_lag_days: int = 3
    rows: int = 0
    columns: int = 0
    csv_file: str = ""            # filename (not full path) of current active CSV

    @property
    def confirmed_end_date(self) -> Optional[date]:
        return date.fromisoformat(self.confirmed_end) if self.confirmed_end else None

    @property
    def raw_end_date(self) -> Optional[date]:
        return date.fromisoformat(self.raw_end) if self.raw_end else None

    @property
    def data_start_date(self) -> Optional[date]:
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
    """Delete oldest CSV files, keeping only `keep` most recent."""
    csvs = sorted(static_dir.glob(f"{_CSV_PREFIX}*.csv"), key=lambda p: p.stat().st_mtime)
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
    """
    Manages the static ML dataset with lag-aware incremental updates.

    On each update_static() call:
      1. Load existing CSV + cache_meta.json (or legacy CSV if first run)
      2. confirmed_end = raw_end - offline_lag_days
      3. Keep frozen rows (up to confirmed_end)
      4. Re-fetch confirmed_end → today  (picks up delayed offline data)
      5. Merge, clean, save new versioned CSV
      6. Rotate old CSVs (keep max_versions)
      7. Write updated cache_meta.json
    """

    # Default start when no CSV exists at all
    _DEFAULT_START = date(2023, 1, 1)

    def __init__(
        self,
        static_dir: str | Path,
        offline_lag_days: int = 3,
        max_versions: int = _MAX_VERSIONS,
        legacy_csv_path: Optional[str | Path] = None,
    ):
        self.static_dir = Path(static_dir)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.offline_lag_days = offline_lag_days
        self.max_versions = max_versions
        self.legacy_csv_path = Path(legacy_csv_path) if legacy_csv_path else None
        self.fetcher = MlDatasetFetcher()
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
        """
        Perform a smart incremental update.

        Args:
            rm_choice:       "RM Charge" | "RM DPR"
            reprocess_from:  If set, discard all data from this date onward and
                             re-fetch. Useful to manually force a backfill.
            apply_cleaning:  Whether to run DataCleaner on the fetched slice.

        Returns:
            The final merged DataFrame (not yet saved — call save() separately).
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
                log.info("Nothing to fetch — dataset is already up to date.")
                return existing

        log.info("Fetching %s → %s (rm=%s)", fetch_start, fetch_end, rm_mode)

        new_df = self.fetcher.get_ml_dataset(
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

        # Merge: new wins on overlap
        final_df = new_df.combine_first(frozen).sort_index().dropna(how="all")
        log.info("Final dataset: %d rows, %d cols", len(final_df), len(final_df.columns))

        return final_df

    def save(self, df: pd.DataFrame, rm_choice: str) -> Path:
        """
        Save df as a new versioned CSV, rotate old files, update cache_meta.json.
        Returns the path of the saved CSV.
        """
        filename = _versioned_filename()
        csv_path = self.static_dir / filename

        df_out = df.copy()
        df_out.index.name = None
        df_out.to_csv(csv_path, index=True)
        log.info("Saved %s (%d rows, %d cols)", filename, len(df), len(df.columns))

        _rotate_csvs(self.static_dir, self.max_versions)

        # Update metadata
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
        return _load_meta(self.static_dir)

    def current_csv_path(self) -> Optional[Path]:
        meta = _load_meta(self.static_dir)
        if not meta or not meta.csv_file:
            return None
        p = self.static_dir / meta.csv_file
        return p if p.exists() else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_existing(self, meta: Optional[CacheMeta]) -> pd.DataFrame:
        """
        Load the active CSV. Falls back to legacy_csv_path on first run.
        """
        # Active versioned CSV (pointed to by meta)
        if meta and meta.csv_file:
            active = self.static_dir / meta.csv_file
            if active.exists():
                log.info("Loading active cache: %s", active.name)
                return _read_csv(active)

        # Any versioned CSV in the folder (fallback if meta is stale)
        csvs = sorted(self.static_dir.glob(f"{_CSV_PREFIX}*.csv"),
                      key=lambda p: p.stat().st_mtime)
        if csvs:
            log.info("Loading most recent CSV (no meta pointer): %s", csvs[-1].name)
            return _read_csv(csvs[-1])

        # Legacy bootstrap
        if self.legacy_csv_path and self.legacy_csv_path.exists():
            log.info("Bootstrapping from legacy CSV: %s", self.legacy_csv_path)
            return _read_csv(self.legacy_csv_path)

        log.info("No existing dataset found — will do a full fetch.")
        return pd.DataFrame()

    def _compute_fetch_window(
        self,
        existing: pd.DataFrame,
        meta: Optional[CacheMeta],
        today: date,
    ) -> tuple[Optional[date], date, pd.DataFrame]:
        """
        Returns (fetch_start, fetch_end, frozen_df).

        frozen_df: the part of existing we keep as-is (up to confirmed_end).
        fetch_start: start of the uncertain window to re-fetch.
                     None if already up to date.
        """
        if existing.empty:
            return self._DEFAULT_START, today, pd.DataFrame()

        raw_end = meta.raw_end_date if meta else existing.index.max().date()
        confirmed_end = raw_end - timedelta(days=self.offline_lag_days)

        # Already fetched through today's confirmed window → nothing to do
        if confirmed_end >= today - timedelta(days=self.offline_lag_days) and raw_end >= today:
            return None, today, existing

        # Freeze rows up to confirmed_end; re-fetch from confirmed_end onward
        frozen = existing.loc[existing.index.date <= confirmed_end]
        fetch_start = confirmed_end       # inclusive re-fetch to capture delayed offline data
        fetch_end = today

        log.info(
            "Cache: raw_end=%s, confirmed_end=%s → re-fetch %s → %s",
            raw_end, confirmed_end, fetch_start, fetch_end,
        )

        return fetch_start, fetch_end, frozen
