"""Range-aware cached dataset fetcher.

Provides
--------
RangeCache      Thread-safe holder for the last fetched date range + DataFrame.
DatasetFetcher  Orchestrates mixed-source fetch (pre/post-cutoff) with range cache.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from threading import Lock

import pandas as pd

from furnace_data.config import load_config
from furnace_data.dataset.service import DatasetService

log = logging.getLogger(__name__)

_config = load_config("setting_ds_dv.yml")

RENAME_DICT: dict = _config.get("rename_dict", {})
RENAME_SET: set = set(RENAME_DICT.values())
KEEP_COLS: list = _config.get("keep_cols", [])


class RangeCache:
    """Thread-safe holder for the last fetched dataset and its date range.

    Attributes:
        start:   Start date of the cached slice (or ``None``).
        end:     End date of the cached slice (or ``None``).
        rm_mode: RM mode used for the cached fetch (``"charge"`` or ``"dpr"``).
        df:      The cached :class:`pandas.DataFrame` (or ``None``).
    """

    def __init__(self) -> None:
        self.start: date | None = None
        self.end: date | None = None
        self.rm_mode: str | None = None
        self.source: str | None = None
        self.df: pd.DataFrame | None = None
        self._lock = Lock()

    def reset(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.start = self.end = self.rm_mode = self.source = self.df = None

    def snapshot(
        self,
    ) -> tuple[date | None, date | None, str | None, str | None, pd.DataFrame | None]:
        """Return a consistent (start, end, rm_mode, df) snapshot."""
        with self._lock:
            return self.start, self.end, self.rm_mode, self.source, self.df

    def update(
        self,
        start: date,
        end: date,
        rm_mode: str,
        source: str,
        df: pd.DataFrame,
    ) -> None:
        """Atomically update the cache."""
        with self._lock:
            self.start = start
            self.end = end
            self.rm_mode = rm_mode
            self.source = source
            self.df = df


class DatasetFetcher:
    """Fetches furnace datasets with range-aware caching.

    Orchestrates between the legacy (pre-cutoff) and new (post-cutoff)
    data sources exposed by :class:`~furnace_data.dataset.service.DatasetService`,
    caching the last fetched range to avoid redundant InfluxDB queries.

    Args:
        service: A :class:`~furnace_data.dataset.service.DatasetService` instance.
                 Defaults to a new instance constructed with default settings.
    """

    def __init__(self, service: DatasetService | None = None) -> None:
        self.service = service or DatasetService()
        self.cache = RangeCache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_df(df: pd.DataFrame, *, keep_unmapped: bool = False) -> pd.DataFrame:
        """Rename columns and keep only the renamed subset."""
        if df.empty:
            return df
        df = df.rename(columns=RENAME_DICT)
        if keep_unmapped:
            return df
        return df.loc[:, df.columns.intersection(RENAME_SET)]

    @staticmethod
    def _align_distribution(
        df_dist: pd.DataFrame,
        df_rm: pd.DataFrame,
        df_hm: pd.DataFrame,
    ) -> pd.DataFrame:
        """Reindex burden distribution to the RM / HM combined index."""
        if df_dist.empty:
            return df_dist
        idx = df_rm.index.union(df_hm.index)
        return df_dist.reindex(idx).sort_index().ffill()

    @staticmethod
    def _normalise_source(source: str) -> str:
        source = (source or "influx").strip().lower()
        if source in {"neon", "neondb", "neon-db"}:
            return "neon_db"
        if source not in {"influx", "neon_db"}:
            raise ValueError("source must be 'influx' or 'neon_db'.")
        return source

    def _fetch_full_range(
        self,
        start: date,
        end: date,
        rm_mode: str,
        source: str,
    ) -> pd.DataFrame:
        if source == "neon_db":
            return self._fetch_new_range(start, end, rm_mode, source)

        cutoff = self.service.cutoff_date

        if end <= cutoff:
            return self._clean_df(
                self.service.fetch_step1(start, end, allowed_columns=RENAME_DICT)
            )

        if start > cutoff:
            return self._fetch_new_range(start, end, rm_mode)

        # Mixed range: legacy slice + new slice
        df_old = self._clean_df(
            self.service.fetch_step1(start, cutoff, allowed_columns=RENAME_DICT)
        )
        new_start = cutoff + timedelta(days=1)
        df_new = self._fetch_new_range(new_start, end, rm_mode, source)

        return pd.concat([df_old, df_new]).sort_index()

    def _fetch_new_range(
        self,
        start: date,
        end: date,
        rm_mode: str,
        source: str,
    ) -> pd.DataFrame:
        df_rm = self.service.fetch_step2(
            start,
            end,
            rm_mode,
            allowed_columns=RENAME_DICT,
            source=source,
        )
        df_hm = self.service.fetch_hotmetal_hourly(
            start,
            end,
            keep_columns=KEEP_COLS,
            source=source,
        )
        df_dist = self.service.fetch_distribution_data(start, end)

        df_dist = self._align_distribution(df_dist, df_rm, df_hm)

        df = df_rm.join([df_hm, df_dist], how="outer").sort_index()
        return self._clean_df(df, keep_unmapped=source == "neon_db")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(
        self,
        start_date: date,
        end_date: date,
        rm_choice: str,
        cache_override: bool = False,
        source: str = "influx",
    ) -> pd.DataFrame:
        """Fetch the furnace dataset for a date range with range-aware caching.

        The cache stores the widest range fetched so far. Subsequent calls that
        fall inside the cached range are served instantly; calls that extend
        beyond either end trigger incremental fetches for only the missing slices.

        Args:
            start_date:     Inclusive start date.
            end_date:       Inclusive end date.
            rm_choice:      ``"RM Charge"`` or ``"RM DPR"``.
            cache_override: If ``True``, clear the cache before fetching.
            source:         ``"influx"`` for rollback or ``"neon_db"``.

        Returns:
            Time-indexed :class:`pandas.DataFrame` sliced to [start_date, end_date]
            with the index named ``"time"``.
        """
        rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"
        source = self._normalise_source(source)

        if cache_override:
            self.cache.reset()

        cache_start, cache_end, cache_rm, cache_source, cache_df = self.cache.snapshot()

        # Fast cache hit
        if (
            cache_df is not None
            and cache_rm == rm_mode
            and cache_source == source
            and cache_start <= start_date
            and cache_end >= end_date
        ):
            out = cache_df.loc[start_date:end_date].copy()
            out.index.name = "time"
            return out

        # Partial / full fetch
        parts = []
        fetch_start, fetch_end = start_date, end_date

        if cache_df is not None and cache_rm == rm_mode and cache_source == source:
            if start_date < cache_start:
                parts.append(
                    self._fetch_full_range(
                        start_date,
                        cache_start - timedelta(days=1),
                        rm_mode,
                        source,
                    )
                )
                fetch_start = start_date
            else:
                fetch_start = cache_start

            parts.append(cache_df)

            if end_date > cache_end:
                parts.append(
                    self._fetch_full_range(
                        cache_end + timedelta(days=1),
                        end_date,
                        rm_mode,
                        source,
                    )
                )
                fetch_end = end_date
            else:
                fetch_end = cache_end

            df_full = pd.concat(parts).sort_index()
        else:
            df_full = self._fetch_full_range(start_date, end_date, rm_mode, source)

        self.cache.update(fetch_start, fetch_end, rm_mode, source, df_full)

        out = df_full.loc[start_date:end_date].copy()
        out.index.name = "time"
        return out

    # Keep the old name as an alias so any callers that used MlDatasetFetcher.get_ml_dataset()
    # continue to work after renaming.
    get_ml_dataset = get_dataset
