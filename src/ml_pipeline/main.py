# src/ml_pipeline/main.py
from datetime import date, timedelta
import pandas as pd
from threading import Lock

from config.config_loader import load_config
from ml_pipeline.ml_dataset_service import MlDatasetService

config = load_config("setting_ds_dv.yml")

# ---------------- CONFIG ----------------
rename_dict = config.get("rename_dict", {})
rename_set = set(rename_dict.values())
keep_cols = config.get("keep_cols", [])

service = MlDatasetService()

# ---------------- RANGE CACHE ----------------
_RANGE_CACHE = {
    "start": None,
    "end": None,
    "rm_mode": None,
    "df": None,
}
_CACHE_LOCK = Lock()

# ---------------- HELPERS ----------------
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns and KEEP ONLY renamed columns.
    Never add new columns.
    """
    if df.empty:
        return df

    df = df.rename(columns=rename_dict)
    return df.loc[:, df.columns.intersection(rename_set)]


def _align_distribution(df_dist, df1, df2):
    if df_dist.empty:
        return df_dist

    if df1.index.equals(df2.index):
        return df_dist.reindex(df1.index).ffill()

    idx = df1.index.union(df2.index)
    return df_dist.reindex(idx).sort_index().ffill()


# ---------------- CORE FETCH (NO CACHE) ----------------
def _fetch_full_range(
    start_date: date,
    end_date: date,
    rm_mode: str,
) -> pd.DataFrame:
    cutoff = service.cutoff_date

    # -------- CASE 1: OLD ONLY --------
    if end_date <= cutoff:
        df = service.fetch_step1(start_date, end_date, allowed_columns=rename_dict)
        return _clean_df(df)

    # -------- CASE 2: NEW ONLY --------
    if start_date > cutoff:
        df_rm = service.fetch_step2(start_date, end_date, rm_mode, allowed_columns=rename_dict)
        df_hm = service.fetch_hotmetal_hourly(start_date, end_date, keep_columns=keep_cols)
        df_dist = service.fetch_distribution_data(start_date, end_date)

        df_dist = _align_distribution(df_dist, df_rm, df_hm)
        df = df_rm.join([df_hm, df_dist], how="outer").sort_index()
        return _clean_df(df)

    # -------- CASE 3: MIXED --------
    df_old = _clean_df(
        service.fetch_step1(start_date, cutoff, allowed_columns=rename_dict)
    )

    new_start = cutoff + timedelta(days=1)

    df_rm = service.fetch_step2(new_start, end_date, rm_mode, allowed_columns=rename_dict)
    df_hm = service.fetch_hotmetal_hourly(new_start, end_date, keep_columns=keep_cols)
    df_dist = service.fetch_distribution_data(new_start, end_date)

    df_dist = _align_distribution(df_dist, df_rm, df_hm)

    df_new = _clean_df(
        df_rm.join([df_hm, df_dist], how="outer").sort_index()
    )

    return pd.concat([df_old, df_new]).sort_index()


# ---------------- PUBLIC API ----------------
def get_ml_dataset(
    start_date: date,
    end_date: date,
    rm_choice: str,        # "RM Charge" | "RM DPR"
    cache_override: bool = False,
) -> pd.DataFrame:
    """
    Optimized range-aware cached dataset fetch.
    """
    rm_mode = "charge" if rm_choice == "RM Charge" else "dpr"

    # -------- STEP 1: FAST CACHE CHECK --------
    with _CACHE_LOCK:
        if cache_override:
            _RANGE_CACHE.update(
                {"start": None, "end": None, "rm_mode": None, "df": None}
            )

        cache = _RANGE_CACHE.copy()

    if (
        cache["df"] is not None
        and cache["rm_mode"] == rm_mode
        and cache["start"] <= start_date
        and cache["end"] >= end_date
    ):
        df = cache["df"].loc[start_date:end_date].copy()
        df.index.name = "time"
        return df

    # -------- STEP 2: FETCH OUTSIDE LOCK --------
    parts = []
    fetch_start, fetch_end = start_date, end_date

    if cache["df"] is not None and cache["rm_mode"] == rm_mode:
        if start_date < cache["start"]:
            parts.append(
                _fetch_full_range(
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
                _fetch_full_range(
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
        df_full = _fetch_full_range(start_date, end_date, rm_mode)
        fetch_start = start_date
        fetch_end = end_date

    # -------- STEP 3: UPDATE CACHE --------
    with _CACHE_LOCK:
        _RANGE_CACHE.update(
            {
                "start": fetch_start,
                "end": fetch_end,
                "rm_mode": rm_mode,
                "df": df_full,
            }
        )

    df = df_full.loc[start_date:end_date].copy()
    df.index.name = "time"
    return df
