"""Range-aware cached dataset fetcher.

Provides
--------
RangeCache      Thread-safe holder for the last fetched date range + DataFrame.
DatasetFetcher  Orchestrates database-backed fetches with range cache.
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
        self.df: pd.DataFrame | None = None
        self._lock = Lock()

    def reset(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.start = self.end = self.rm_mode = self.df = None

    def snapshot(
        self,
    ) -> tuple[date | None, date | None, str | None, pd.DataFrame | None]:
        """Return a consistent (start, end, rm_mode, df) snapshot."""
        with self._lock:
            return self.start, self.end, self.rm_mode, self.df

    def update(
        self,
        start: date,
        end: date,
        rm_mode: str,
        df: pd.DataFrame,
    ) -> None:
        """Atomically update the cache."""
        with self._lock:
            self.start = start
            self.end = end
            self.rm_mode = rm_mode
            self.df = df


class DatasetFetcher:
    """Fetches furnace datasets with range-aware caching.

    Orchestrates historical static and current database-backed data exposed by
    :class:`~furnace_data.dataset.service.DatasetService`, caching the last
    fetched range to avoid redundant PostgreSQL queries.

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
    def _slice_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Slice a time-indexed dataset by inclusive calendar dates."""
        if df.empty:
            out = df.copy()
        else:
            source = df
            if not isinstance(source.index, pd.DatetimeIndex):
                coerced_index = pd.to_datetime(source.index, errors="coerce")
                if not pd.isna(coerced_index).any():
                    source = source.copy()
                    source.index = pd.DatetimeIndex(coerced_index)

        if not df.empty and isinstance(source.index, pd.DatetimeIndex):
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
            out = source.loc[(source.index >= start_ts) & (source.index < end_ts)].copy()
        elif not df.empty:
            out = df.loc[start:end].copy()
        out.index.name = "time"
        return out

    def _fetch_full_range(
        self,
        start: date,
        end: date,
        rm_mode: str,
    ) -> pd.DataFrame:
        cutoff = self.service.cutoff_date

        if end <= cutoff:
            return self._clean_df(
                self.service.fetch_step1(start, end, allowed_columns=RENAME_DICT)
            )

        if start > cutoff:
            return self._fetch_new_range(start, end, rm_mode)

        # Mixed range: historical static slice + current generated slice
        df_old = self._clean_df(
            self.service.fetch_step1(start, cutoff, allowed_columns=RENAME_DICT)
        )
        new_start = cutoff + timedelta(days=1)
        df_new = self._fetch_new_range(new_start, end, rm_mode)

        return pd.concat([df_old, df_new]).sort_index()

    def _build_post_cutoff_df(
        self,
        start: date,
        end: date,
        rm_mode: str,
        *,
        raise_online_errors: bool = False,
    ) -> pd.DataFrame:
        """Join Steps 2-5 into a raw DataFrame with ML column names (no rename)."""
        df_rm = self.service.fetch_step2(
            start,
            end,
            rm_mode,
            allowed_columns=RENAME_DICT,
        )
        df_hm = self.service.fetch_hotmetal_hourly(
            start,
            end,
            keep_columns=KEEP_COLS,
        )
        df_dist = self.service.fetch_distribution_data(start, end)
        df_online = self.service.fetch_online_process_params(
            start, end, raise_on_error=raise_online_errors
        )
        df_temp = self.service.fetch_online_temperature_params(
            start, end, raise_on_error=raise_online_errors
        )
        df_heatload = self.service.fetch_online_heatload_params(
            start, end, raise_on_error=raise_online_errors
        )
        df_misc = self.service.fetch_online_misc_params(
            start, end, raise_on_error=raise_online_errors
        )

        df_dist = self._align_distribution(df_dist, df_rm, df_hm)

        frames = [df_hm, df_dist]
        if not df_online.empty:
            frames.append(df_online)
        if not df_temp.empty:
            frames.append(df_temp)
        if not df_heatload.empty:
            frames.append(df_heatload)
        if not df_misc.empty:
            frames.append(df_misc)
        return df_rm.join(frames, how="outer").sort_index()

    def _fetch_new_range(
        self,
        start: date,
        end: date,
        rm_mode: str,
    ) -> pd.DataFrame:
        return self._clean_df(
            self._build_post_cutoff_df(start, end, rm_mode),
            keep_unmapped=True,
        )

    def extend_and_persist(
        self,
        start_date: date,
        end_date: date,
        rm_choice: str,
        override_from_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch post-cutoff data (Steps 2-5), write to ``historical_static_ml_dataset``.

        Args:
            start_date:          Minimum start date for the fetch (ignored when
                                 ``override_from_date`` is set).
            end_date:            Inclusive end date to generate data to.
            rm_choice:           ``"RM Charge"`` or ``"RM DPR"``.
            override_from_date:  If given, existing DB rows from this date
                                 onwards are deleted before re-generating.

        Returns:
            Raw (pre-cleaning) joined DataFrame written to the DB, or an
            empty DataFrame if there is nothing new to generate.
        """
        from furnace_data.offline_writer import (
            delete_from_static_table,
            write_to_static_table,
        )

        rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

        if override_from_date is not None:
            deleted = delete_from_static_table(override_from_date)
            log.info("Override: deleted %d rows from %s onwards.", deleted, override_from_date)
            fetch_start = override_from_date
        else:
            fetch_start = max(
                self.service.cutoff_date + timedelta(days=1),
                start_date,
            )

        if fetch_start > end_date:
            log.info("Nothing to generate: fetch_start %s > end_date %s.", fetch_start, end_date)
            return pd.DataFrame()

        log.info("Building post-cutoff data %s -> %s (mode=%s).", fetch_start, end_date, rm_mode)
        raw_df = self._build_post_cutoff_df(fetch_start, end_date, rm_mode)

        if raw_df.empty:
            log.warning("Steps 2-5 returned no data for %s -> %s.", fetch_start, end_date)
            return raw_df

        rows_written = write_to_static_table(raw_df)
        log.info(
            "Persisted %d rows to historical_static_ml_dataset (%s -> %s).",
            rows_written, fetch_start, end_date,
        )
        self.cache.reset()
        return raw_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataset(
        self,
        start_date: date,
        end_date: date,
        rm_choice: str,
        cache_override: bool = False,
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
        Returns:
            Time-indexed :class:`pandas.DataFrame` sliced to [start_date, end_date]
            with the index named ``"time"``.
        """
        rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

        if cache_override:
            self.cache.reset()

        cache_start, cache_end, cache_rm, cache_df = self.cache.snapshot()

        # Fast cache hit
        if (
            cache_df is not None
            and cache_rm == rm_mode
            and cache_start <= start_date
            and cache_end >= end_date
        ):
            return self._slice_date_range(cache_df, start_date, end_date)

        # Partial / full fetch
        parts = []
        fetch_start, fetch_end = start_date, end_date

        if cache_df is not None and cache_rm == rm_mode:
            if start_date < cache_start:
                parts.append(
                    self._fetch_full_range(
                        start_date,
                        cache_start - timedelta(days=1),
                        rm_mode,
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
                    )
                )
                fetch_end = end_date
            else:
                fetch_end = cache_end

            df_full = pd.concat(parts).sort_index()
        else:
            df_full = self._fetch_full_range(start_date, end_date, rm_mode)

        self.cache.update(fetch_start, fetch_end, rm_mode, df_full)

        return self._slice_date_range(df_full, start_date, end_date)

    def build_local_delta(
        self,
        start: date,
        end: date,
        rm_mode: str = "charge",
        *,
        raise_on_error: bool = False,
    ) -> pd.DataFrame:
        """Return renamed (not yet fully cleaned) post-cutoff data for local caching.

        Calls Steps 2-5 and applies the column rename, but does NOT run the full
        DataCleaner pipeline and does NOT write to the offline DB static table.
        Returns an empty DataFrame on failure unless ``raise_on_error`` is true.
        """
        try:
            raw = self._build_post_cutoff_df(
                start,
                end,
                rm_mode,
                raise_online_errors=raise_on_error,
            )
            if raw.empty:
                return raw
            return self._clean_df(raw, keep_unmapped=True)
        except Exception:
            log.warning(
                "build_local_delta failed for %s -> %s (rm_mode=%s)",
                start, end, rm_mode,
                exc_info=True,
            )
            if raise_on_error:
                raise
            return pd.DataFrame()

    # Deprecated compatibility alias retained for older integrations; new callers
    # must use get_dataset().
    get_ml_dataset = get_dataset
