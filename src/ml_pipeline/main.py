# # src/ml_pipeline/main.py
# from datetime import date, timedelta
# import pandas as pd
# from threading import Lock

# from config.config_loader import load_config
# from ml_pipeline.ml_dataset_service import MlDatasetService

# config = load_config("setting_ds_dv.yml")

# # ---------------- CONFIG ----------------
# rename_dict = config.get("rename_dict", {})
# rename_set = set(rename_dict.values())
# keep_cols = config.get("keep_cols", [])

# service = MlDatasetService()

# # ---------------- RANGE CACHE ----------------
# _RANGE_CACHE = {
#     "start": None,
#     "end": None,
#     "rm_mode": None,
#     "df": None,
# }
# _CACHE_LOCK = Lock()

# # ---------------- HELPERS ----------------
# def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Rename columns and KEEP ONLY renamed columns.
#     Never add new columns.
#     """
#     if df.empty:
#         return df

#     df = df.rename(columns=rename_dict)
#     return df.loc[:, df.columns.intersection(rename_set)]


# def _align_distribution(df_dist, df1, df2):
#     if df_dist.empty:
#         return df_dist

#     if df1.index.equals(df2.index):
#         return df_dist.reindex(df1.index).ffill()

#     idx = df1.index.union(df2.index)
#     return df_dist.reindex(idx).sort_index().ffill()


# # ---------------- CORE FETCH (NO CACHE) ----------------
# def _fetch_full_range(
#     start_date: date,
#     end_date: date,
#     rm_mode: str,
# ) -> pd.DataFrame:
#     cutoff = service.cutoff_date

#     # -------- CASE 1: OLD ONLY --------
#     if end_date <= cutoff:
#         df = service.fetch_step1(start_date, end_date, allowed_columns=rename_dict)
#         return _clean_df(df)

#     # -------- CASE 2: NEW ONLY --------
#     if start_date > cutoff:
#         df_rm = service.fetch_step2(start_date, end_date, rm_mode, allowed_columns=rename_dict)
#         df_hm = service.fetch_hotmetal_hourly(start_date, end_date, keep_columns=keep_cols)
#         df_dist = service.fetch_distribution_data(start_date, end_date)

#         df_dist = _align_distribution(df_dist, df_rm, df_hm)
#         df = df_rm.join([df_hm, df_dist], how="outer").sort_index()
#         return _clean_df(df)

#     # -------- CASE 3: MIXED --------
#     df_old = _clean_df(
#         service.fetch_step1(start_date, cutoff, allowed_columns=rename_dict)
#     )

#     new_start = cutoff + timedelta(days=1)

#     df_rm = service.fetch_step2(new_start, end_date, rm_mode, allowed_columns=rename_dict)
#     df_hm = service.fetch_hotmetal_hourly(new_start, end_date, keep_columns=keep_cols)
#     df_dist = service.fetch_distribution_data(new_start, end_date)

#     df_dist = _align_distribution(df_dist, df_rm, df_hm)

#     df_new = _clean_df(
#         df_rm.join([df_hm, df_dist], how="outer").sort_index()
#     )

#     return pd.concat([df_old, df_new]).sort_index()


# # ---------------- PUBLIC API ----------------
# def get_ml_dataset(
#     start_date: date,
#     end_date: date,
#     rm_choice: str,        # "RM Charge" | "RM DPR"
#     cache_override: bool = False,
# ) -> pd.DataFrame:
#     """
#     Optimized range-aware cached dataset fetch.
#     """
#     rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

#     # -------- STEP 1: FAST CACHE CHECK --------
#     with _CACHE_LOCK:
#         if cache_override:
#             _RANGE_CACHE.update(
#                 {"start": None, "end": None, "rm_mode": None, "df": None}
#             )

#         cache = _RANGE_CACHE.copy()

#     if (
#         cache["df"] is not None
#         and cache["rm_mode"] == rm_mode
#         and cache["start"] <= start_date
#         and cache["end"] >= end_date
#     ):
#         df = cache["df"].loc[start_date:end_date].copy()
#         df.index.name = "time"
#         return df

#     # -------- STEP 2: FETCH OUTSIDE LOCK --------
#     parts = []
#     fetch_start, fetch_end = start_date, end_date

#     if cache["df"] is not None and cache["rm_mode"] == rm_mode:
#         if start_date < cache["start"]:
#             parts.append(
#                 _fetch_full_range(
#                     start_date,
#                     cache["start"] - timedelta(days=1),
#                     rm_mode,
#                 )
#             )
#             fetch_start = start_date
#         else:
#             fetch_start = cache["start"]

#         parts.append(cache["df"])

#         if end_date > cache["end"]:
#             parts.append(
#                 _fetch_full_range(
#                     cache["end"] + timedelta(days=1),
#                     end_date,
#                     rm_mode,
#                 )
#             )
#             fetch_end = end_date
#         else:
#             fetch_end = cache["end"]

#         df_full = pd.concat(parts).sort_index()

#     else:
#         df_full = _fetch_full_range(start_date, end_date, rm_mode)
#         fetch_start = start_date
#         fetch_end = end_date

#     # -------- STEP 3: UPDATE CACHE --------
#     with _CACHE_LOCK:
#         _RANGE_CACHE.update(
#             {
#                 "start": fetch_start,
#                 "end": fetch_end,
#                 "rm_mode": rm_mode,
#                 "df": df_full,
#             }
#         )

#     df = df_full.loc[start_date:end_date].copy()
#     df.index.name = "time"
#     return df


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


# ---------------- CACHE OBJECT ----------------
class RangeCache:
    def __init__(self):
        self.start = None
        self.end = None
        self.rm_mode = None
        self.df = None
        self._lock = Lock()

    def reset(self):
        with self._lock:
            self.start = None
            self.end = None
            self.rm_mode = None
            self.df = None

    def snapshot(self):
        with self._lock:
            return {
                "start": self.start,
                "end": self.end,
                "rm_mode": self.rm_mode,
                "df": self.df,
            }

    def update(self, start, end, rm_mode, df):
        with self._lock:
            self.start = start
            self.end = end
            self.rm_mode = rm_mode
            self.df = df


# ---------------- DATASET FETCHER ----------------
class MlDatasetFetcher:
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
    def _align_distribution(df_dist, df1, df2):
        if df_dist.empty:
            return df_dist

        if df1.index.equals(df2.index):
            return df_dist.reindex(df1.index).ffill()

        idx = df1.index.union(df2.index)
        return df_dist.reindex(idx).sort_index().ffill()

    # ---------- CORE FETCH ----------
    def _fetch_full_range(
        self,
        start_date: date,
        end_date: date,
        rm_mode: str,
    ) -> pd.DataFrame:
        cutoff = self.service.cutoff_date

        # ----- OLD ONLY -----
        if end_date <= cutoff:
            df = self.service.fetch_step1(
                start_date, end_date, allowed_columns=RENAME_DICT
            )
            return self._clean_df(df)

        # ----- NEW ONLY -----
        if start_date > cutoff:
            df_rm = self.service.fetch_step2(
                start_date, end_date, rm_mode, allowed_columns=RENAME_DICT
            )
            df_hm = self.service.fetch_hotmetal_hourly(
                start_date, end_date, keep_columns=KEEP_COLS
            )
            df_dist = self.service.fetch_distribution_data(start_date, end_date)

            df_dist = self._align_distribution(df_dist, df_rm, df_hm)
            df = df_rm.join([df_hm, df_dist], how="outer").sort_index()
            return self._clean_df(df)

        # ----- MIXED -----
        df_old = self._clean_df(
            self.service.fetch_step1(
                start_date, cutoff, allowed_columns=RENAME_DICT
            )
        )

        new_start = cutoff + timedelta(days=1)

        df_rm = self.service.fetch_step2(
            new_start, end_date, rm_mode, allowed_columns=RENAME_DICT
        )
        df_hm = self.service.fetch_hotmetal_hourly(
            new_start, end_date, keep_columns=KEEP_COLS
        )
        df_dist = self.service.fetch_distribution_data(new_start, end_date)

        df_dist = self._align_distribution(df_dist, df_rm, df_hm)

        df_new = self._clean_df(
            df_rm.join([df_hm, df_dist], how="outer").sort_index()
        )

        return pd.concat([df_old, df_new]).sort_index()

    # ---------- PUBLIC API ----------
    def get_ml_dataset(
        self,
        start_date: date,
        end_date: date,
        rm_choice: str,   # "RM Charge" | "RM DPR"
        cache_override: bool = False,
    ) -> pd.DataFrame:
        """
        Optimized range-aware cached dataset fetch.
        """
        rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

        if cache_override:
            self.cache.reset()

        cache = self.cache.snapshot()

        # ----- FAST CACHE HIT -----
        if (
            cache["df"] is not None
            and cache["rm_mode"] == rm_mode
            and cache["start"] <= start_date
            and cache["end"] >= end_date
        ):
            df = cache["df"].loc[start_date:end_date].copy()
            df.index.name = "time"
            return df

        # ----- PARTIAL FETCH -----
        parts = []
        fetch_start, fetch_end = start_date, end_date

        if cache["df"] is not None and cache["rm_mode"] == rm_mode:
            if start_date < cache["start"]:
                parts.append(
                    self._fetch_full_range(
                        start_date,
                        cache["start"] - timedelta(days=1),
                        rm_mode,
                    )
                )
                fetch_start = start_date
            else:
                fetch_start = cache["start"]

            parts.append(cache["df"])

            if end_date > cache["end"]:
                parts.append(
                    self._fetch_full_range(
                        cache["end"] + timedelta(days=1),
                        end_date,
                        rm_mode,
                    )
                )
                fetch_end = end_date
            else:
                fetch_end = cache["end"]

            df_full = pd.concat(parts).sort_index()

        else:
            df_full = self._fetch_full_range(start_date, end_date, rm_mode)

        # ----- UPDATE CACHE -----
        self.cache.update(fetch_start, fetch_end, rm_mode, df_full)

        df = df_full.loc[start_date:end_date].copy()
        df.index.name = "time"
        return df
