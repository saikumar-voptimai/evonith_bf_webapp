# src/ml_pipeline/main.py
from datetime import date, timedelta
from threading import Lock
import pandas as pd

from config.config_loader import load_config
from ml_pipeline.ml_dataset_service import MlDatasetService


# ---------------- CONFIG ----------------
config = load_config("setting_ds_dv.yml")

RENAME_DICT = config.get("rename_dict", {})
RENAME_SET = set(RENAME_DICT.values())
KEEP_COLS = config.get("keep_cols", [])


# ---------------- RANGE CACHE ----------------
class RangeCache:
    """
    Caches the last fetched dataset along with its date range and RM mode.
    """
    def __init__(self):
        self.start: date | None = None
        self.end: date | None = None
        self.rm_mode: str | None = None
        self.df: pd.DataFrame | None = None
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.start = self.end = self.rm_mode = self.df = None

    def snapshot(self):
        with self._lock:
            return self.start, self.end, self.rm_mode, self.df

    def update(self, start, end, rm_mode, df):
        with self._lock:
            self.start = start
            self.end = end
            self.rm_mode = rm_mode
            self.df = df


# ---------------- DATASET FETCHER ----------------
class MlDatasetFetcher:
    """
    Fetches ML datasets with range-aware caching.   
    """
    def __init__(self):
        self.service = MlDatasetService()
        self.cache = RangeCache()

    # ---------- HELPERS ----------
    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns and KEEP ONLY renamed columns.
        Never add new columns.
        """
        if df.empty:
            return df
        df = df.rename(columns=RENAME_DICT)
        return df.loc[:, df.columns.intersection(RENAME_SET)]

    @staticmethod
    def _align_distribution(df_dist, df_rm, df_hm):
        """
        Align burden distribution index to RM / HM index.
        """
        if df_dist.empty:
            return df_dist

        idx = df_rm.index.union(df_hm.index)
        return df_dist.reindex(idx).sort_index().ffill()

    # ---------- CORE FETCH ----------
    def _fetch_full_range(self, start: date, end: date, rm_mode: str) -> pd.DataFrame:
        """
        Fetch full dataset for given range by combining old and new data fetches.
        """
        cutoff = self.service.cutoff_date

        # ---------- OLD ONLY ----------
        if end <= cutoff:
            return self._clean_df(
                self.service.fetch_step1(start, end, allowed_columns=RENAME_DICT)
            )

        # ---------- NEW ONLY ----------
        if start > cutoff:
            return self._fetch_new_range(start, end, rm_mode)

        # ---------- MIXED ----------
        df_old = self._clean_df(
            self.service.fetch_step1(start, cutoff, allowed_columns=RENAME_DICT)
        )

        new_start = cutoff + timedelta(days=1)
        df_new = self._fetch_new_range(new_start, end, rm_mode)

        return pd.concat([df_old, df_new]).sort_index()

    def _fetch_new_range(self, start: date, end: date, rm_mode: str) -> pd.DataFrame:
        """
        Fetch RM + Hot Metal + Distribution and merge.
        """
        df_rm = self.service.fetch_step2(
            start, end, rm_mode, allowed_columns=RENAME_DICT
        )
        df_hm = self.service.fetch_hotmetal_hourly(
            start, end, keep_columns=KEEP_COLS
        )
        df_dist = self.service.fetch_distribution_data(start, end)

        df_dist = self._align_distribution(df_dist, df_rm, df_hm)

        df = df_rm.join([df_hm, df_dist], how="outer").sort_index()
        return self._clean_df(df)

    # ---------- PUBLIC API ----------
    def get_ml_dataset(
        self,
        start_date: date,
        end_date: date,
        rm_choice: str,  # "RM Charge" | "RM DPR"
        cache_override: bool = False,
    ) -> pd.DataFrame:
        """
        Range-aware cached dataset fetch.
        """
        rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

        if cache_override:
            self.cache.reset()

        cache_start, cache_end, cache_rm, cache_df = self.cache.snapshot()

        # ---------- FAST CACHE HIT ----------
        if (
            cache_df is not None
            and cache_rm == rm_mode
            and cache_start <= start_date
            and cache_end >= end_date
        ):
            out = cache_df.loc[start_date:end_date].copy()
            out.index.name = "time"
            return out

        # ---------- PARTIAL / FULL FETCH ----------
        parts = []
        fetch_start, fetch_end = start_date, end_date

        if cache_df is not None and cache_rm == rm_mode:
            if start_date < cache_start:
                parts.append(
                    self._fetch_full_range(
                        start_date, cache_start - timedelta(days=1), rm_mode
                    )
                )
                fetch_start = start_date
            else:
                fetch_start = cache_start

            parts.append(cache_df)

            if end_date > cache_end:
                parts.append(
                    self._fetch_full_range(
                        cache_end + timedelta(days=1), end_date, rm_mode
                    )
                )
                fetch_end = end_date
            else:
                fetch_end = cache_end

            df_full = pd.concat(parts).sort_index()
        else:
            df_full = self._fetch_full_range(start_date, end_date, rm_mode)

        # ---------- UPDATE CACHE ----------
        self.cache.update(fetch_start, fetch_end, rm_mode, df_full)

        out = df_full.loc[start_date:end_date].copy()
        out.index.name = "time"
        return out
