"""
StaticDatasetManager — ported from src/ml_pipeline/static_dataset_manager.py.

Manages reading, updating, and saving a static ML dataset CSV.
Replaces st.info() calls with logging.
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.core.dataset_fetcher import MlDatasetFetcher
from app.core.data_cleaning import DataCleaner, build_default_config

log = logging.getLogger(__name__)


class StaticDatasetManager:
    """
    1. Reads existing static dataset from CSV.
    2. Determines fetch range based on existing data.
    3. Fetches new ML data using MlDatasetFetcher.
    4. Optionally cleans the fetched data using DataCleaner.
    5. Merges new data with existing data (new data takes precedence).
    6. Saves the updated dataset back to CSV.
    """

    def __init__(self, static_path: str | Path):
        self.static_path = Path(static_path)
        self.fetcher = MlDatasetFetcher()
        self.cleaner = DataCleaner(build_default_config())

    def _read_existing_static(self) -> pd.DataFrame:
        if not self.static_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(self.static_path, parse_dates=[0])
        df.columns = df.columns.str.strip()

        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
        df = df.set_index(time_col)

        return df.sort_index()

    # Default start date when no existing CSV — matches cutoff in config
    _DEFAULT_START = date(2025, 1, 1)

    def _get_fetch_range(self, existing_df: pd.DataFrame) -> tuple[date, date]:
        if existing_df.empty:
            return self._DEFAULT_START, date.today()

        last_date = existing_df.index.max().date()
        return last_date, date.today() + timedelta(days=1)

    def update_static(
        self,
        rm_choice: str,
        start_date: date | None = None,
        apply_cleaning: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch new data, optionally clean, merge with existing, return final df.
        Does NOT save automatically — call save() separately.
        """
        existing = self._read_existing_static()

        # Case 1: Reprocess from a specific date
        if start_date:
            log.info("Reprocessing historical data from %s", start_date)
            if not existing.empty:
                existing = existing.loc[existing.index.date < start_date]
            fetch_start = start_date
            fetch_end = date.today() + timedelta(days=1)

        # Case 2: Normal incremental
        else:
            fetch_start, fetch_end = self._get_fetch_range(existing)

            if fetch_start is None:
                log.info("Initial full fetch.")
            else:
                log.info("Fetching incrementally from %s -> %s", fetch_start, fetch_end)

            if fetch_start == date.today():
                log.info("No new data to fetch.")
                return existing

        ml_df = self.fetcher.get_ml_dataset(
            start_date=fetch_start,
            end_date=fetch_end,
            rm_choice=rm_choice,
            cache_override=True,
        )

        if ml_df.empty:
            log.info("No data fetched.")
            return existing

        if apply_cleaning:
            ml_df = self.cleaner.clean(ml_df)

        final_df = ml_df.combine_first(existing)
        final_df = final_df.sort_index()
        final_df = final_df.dropna(how="all")

        return final_df

    def save(self, df: pd.DataFrame) -> None:
        self.static_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.copy()
        df_out.index.name = None
        df_out.to_csv(self.static_path, index=True)
        log.info("Saved static dataset to %s (%d rows, %d cols)", self.static_path, len(df), len(df.columns))
