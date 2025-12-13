from datetime import date, timedelta
import pandas as pd
from functools import lru_cache

from config.config_loader import load_config
from ml_pipeline.ml_dataset_service import MlDatasetService

config = load_config("setting_ds_dv.yml")

# ---------------- CONFIG ----------------
rename_dict = config.get("rename_dict", {})
rename_set = set(rename_dict.values())
keep_cols = config.get("keep_cols", [])

service = MlDatasetService()


# ---------------- HELPERS ----------------
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns and KEEP ONLY renamed columns.
    Never add new columns.
    """
    if df.empty:
        return df

    df = df.rename(columns=rename_dict)
    return df[[c for c in df.columns if c in rename_set]]


def _align_distribution(df_dist, df1, df2):
    if df_dist.empty:
        return df_dist

    idx = df1.index.union(df2.index)
    return df_dist.reindex(idx).sort_index().ffill()


# ---------------- CACHE LAYER ----------------
@lru_cache(maxsize=32)
def _cached_fetch(
    start_date: date,
    end_date: date,
    rm_mode: str,
):
    """
    Internal cached fetch.
    Cache key = (start_date, end_date, rm_mode)
    """
    cutoff = service.cutoff_date

    # ---------------- CASE 1: OLD ONLY ----------------
    if end_date <= cutoff:
        df = service.fetch_step1(start_date, end_date, allowed_columns=rename_dict)
        return _clean_df(df)

    # ---------------- CASE 2: NEW ONLY ----------------
    if start_date > cutoff:
        df_rm = service.fetch_step2(start_date, end_date, rm_mode, allowed_columns=rename_dict)
        df_hm = service.fetch_hotmetal_hourly(start_date, end_date, keep_columns=keep_cols)
        df_dist = service.fetch_distribution_data(start_date, end_date)

        df_dist = _align_distribution(df_dist, df_rm, df_hm)

        df = df_rm.join([df_hm, df_dist], how="outer").sort_index()
        return _clean_df(df)

    # ---------------- CASE 3: MIXED ----------------
    # OLD PART
    df_old = service.fetch_step1(start_date, cutoff, allowed_columns=rename_dict)
    df_old = _clean_df(df_old)

    # NEW PART
    new_start = cutoff + timedelta(days=1)

    df_rm = service.fetch_step2(new_start, end_date, rm_mode, allowed_columns=rename_dict)
    df_hm = service.fetch_hotmetal_hourly(new_start, end_date, keep_columns=keep_cols)
    df_dist = service.fetch_distribution_data(new_start, end_date)

    df_dist = _align_distribution(df_dist, df_rm, df_hm)

    df_new = df_rm.join([df_hm, df_dist], how="outer").sort_index()
    df_new = _clean_df(df_new)

    return pd.concat([df_old, df_new]).sort_index()


# ---------------- PUBLIC API ----------------
def get_ml_dataset(
    start_date: date,
    end_date: date,
    rm_choice: str,        # "RM Charge" | "RM DPR"
    cache_override: bool = False,
) -> pd.DataFrame:
    """
    Main entry point for UI.
    """
    rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

    if cache_override:
        _cached_fetch.cache_clear()

    df = _cached_fetch(start_date, end_date, rm_mode)
    df.index.name = "time"
    return df
