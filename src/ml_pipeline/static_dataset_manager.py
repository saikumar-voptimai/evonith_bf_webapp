import streamlit as st
from datetime import date, timedelta
import pandas as pd
from pathlib import Path

from ml_pipeline.main import MlDatasetFetcher
from ml_pipeline.data_cleaning import DataCleaner, build_default_config


class StaticDatasetManager:
    """
    Orchestrates incremental Static (filtered ML dataset) updates.
    """

    def __init__(self, static_path: str):
        self.static_path = Path(static_path)
        self.fetcher = MlDatasetFetcher()
        self.cleaner = DataCleaner(build_default_config())

    # -----------------------------
    # Step 1: Read existing Static dataset safely
    # -----------------------------
    def _read_existing_static(self) -> pd.DataFrame:
        if not self.static_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(self.static_path, parse_dates=[0])
        df.columns = df.columns.str.strip()

        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
        df = df.set_index(time_col)

        return df.sort_index()


    # -----------------------------
    # Step 2: Decide fetch range
    # -----------------------------
    def _get_fetch_range(self, existing_df: pd.DataFrame):
        if existing_df.empty:
            return None, date.today()

        last_date = existing_df.index.max().date()
        return last_date, date.today()+timedelta(days=1)

    # -----------------------------
    # Step 3: Fetch → Filter → Merge
    # -----------------------------
    def update_static(self, rm_choice: str) -> pd.DataFrame:
        existing = self._read_existing_static()
        start_ts, end_ts = self._get_fetch_range(existing)

        if start_ts is None:
            st.info("Initial full fetch.")
        else:
            st.info(f"Fetching from {start_ts} → {end_ts}")
        if start_ts == date.today():
            st.info("No new data to fetch.")
            return existing
        else:
            ml_df = self.fetcher.get_ml_dataset(
                start_date=start_ts,
                end_date=end_ts,
                rm_choice=rm_choice,
                cache_override=True,
            )

            if ml_df.empty:
                st.info("No new data fetched.")
                return existing

            filtered_df = self.cleaner.clean(ml_df)

            final_df = filtered_df.combine_first(existing)
            final_df = final_df.sort_index()
            final_df = final_df.dropna(how="all")

            return final_df






    # -----------------------------
    # Step 4: Persist (no index header)
    # -----------------------------
    def save(self, df: pd.DataFrame):
        # Remove index name so no column header is written
        df = df.copy()
        df.index.name = None

        df.to_csv(self.static_path, index=True)
