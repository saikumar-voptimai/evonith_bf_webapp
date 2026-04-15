"""Day-windowed data fetchers for the Material Balance page.

Primary data source
-------------------
Static furnace dataset CSV at ``src/assets/data/furnace_dataset.csv``.
This CSV is produced by the ML-Dataset service and contains hourly rows
with UTC-naive timestamps (00–23 each day) for all composition, blast,
chemistry, and tonnage columns.

Why the CSV and not InfluxDB?
The offline InfluxDB bucket (``bf2_evonith_offline_utc``) holds
shift-granularity composition data and is used only as a secondary /
future option. The ML-dataset CSV is already cleaned, aligned, and
available for any historically-analysed date.

IST window handling
-------------------
One IST calendar date corresponds to UTC ``[start_utc, end_utc)`` as
returned by :func:`get_day_window_utc`.  The CSV timestamps are parsed
as UTC-aware and filtered with that window.

Mass vs composition columns
---------------------------
``*_CALC_MT`` columns contain hourly **tonnes charged / tapped** and
must be **summed** over the 24 h window to get the day total.  All other
numeric columns (composition %, blast parameters, chemistry readings)
are **averaged**.

Lag-shift capability
--------------------
``rm_lag_hours`` (default 0) shifts the raw-material and composition
input window backward.  Example: ``rm_lag_hours=96`` says "the RM
charged 4 days ago is what produced today's hot metal."

``blast_lag_hours`` (default 0) does the same for blast / process-param
columns.  Positive = look further back in time.

InfluxDB helpers (``fetch_rm_for_day``, ``fetch_hm_slag_for_day``, etc.)
are preserved for the live-data pathway and DPR mapping feature but are
not called by the page by default.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Set, Tuple

import pandas as pd
import pytz
import streamlit as st

from furnace_data.influx.base import BaseDataFetcher
from data.ml.static_csv import get_static_dataset_path
from data.retrieval import clean_rm_data, fetch_offline_data

log = logging.getLogger("root")

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# InfluxDB measurement constants (secondary / live-data path)
# ---------------------------------------------------------------------------
RM_MEASUREMENT = "rm_updated_data"
HM_SLAG_MEASUREMENT = "hotmetal_slag_updated_data"
DPR_MEASUREMENT = "dpr_data"
OFFLINE_DB = "bf2_evonith_offline_utc"

# Process-param field names used by the gas-phase math (schema names, not
# CSV column names — the rename mapping below covers the translation).
PROCESS_FIELDS = [
    "hot_blast_vol_nm3h",
    "hot_blast_temp",
    "hot_blast_press",
    "oxygen_enrichment_pct",
    "steam_injection",
    "co_pct",
    "co2_pct",
    "h2_pct",
    "top_press_avg",
    "top_temp_avg",
    "body_etaco",
    "production_per_hour",  # used by resolve_hm_slag_masses fallback
]

# ---------------------------------------------------------------------------
# Static CSV path
# ---------------------------------------------------------------------------
_CSV_PATH = get_static_dataset_path()

# ---------------------------------------------------------------------------
# Column mapping: CSV column name → schema-compatible name expected by
# constants.py, compute.py, and the InfluxDB process_params field list.
# ---------------------------------------------------------------------------

# Hourly tonnage columns that must be SUMMED (not averaged) over a day window.
MASS_CSV_COLS: Set[str] = {
    "COKE_CALC_MT",
    "NUTCOKE_CALC_MT",
    "PCI_2_CALC_MT",
    "ORE_CALC_MT",
    "SINTER_CALC_MT",
    "TOTAL_PELLET_CALC_MT",
    "FLUX_CALC_MT",
}

CSV_TO_SCHEMA: Dict[str, str] = {
    # ── Coke ──────────────────────────────────────────────────────────
    "COKE_FC%": "coke_fc_pct",
    "COKE_ASH%": "coke_ash_pct",
    "COKE_MOIST%": "coke_moist_pct",
    "COKE_VM%": "coke_vm_pct",
    "COKE_IM%": "coke_im_pct",
    "COKE_CALC_MT": "coke_mt",
    # ── Nut Coke ──────────────────────────────────────────────────────
    "NUTCOKE_FC%": "nutcoke_fc_pct",
    "NUTCOKE_ASH%": "nutcoke_ash_pct",
    "NUTCOKE_MOIST%": "nutcoke_moist_pct",
    "NUTCOKE_VM%": "nutcoke_vm_pct",
    "NUTCOKE_IM%": "nutcoke_im_pct",
    "NUTCOKE_CALC_MT": "nutcoke_prime_mt",
    # ── PCI ───────────────────────────────────────────────────────────
    "PCI_2_FC%": "pci2_fc_pct",
    "PCI_2_ASH%": "pci2_ash_pct",
    "PCI_2_CALC_MT": "pci2_mt",
    # ── Ore ───────────────────────────────────────────────────────────
    "ORE_FE(T)%": "ore_fe_total_pct",
    "ORE_P%": "ore_p_pct",
    "ORE_SIO2%": "ore_sio2_pct",
    "ORE_CAO%": "ore_cao_pct",
    "ORE_MGO%": "ore_mgo_pct",
    "ORE_AL2O3%": "ore_al2o3_pct",
    "ORE_MNO%": "ore_mno_pct",
    "ORE_TIO2%": "ore_tio2_pct",
    "ORE_K2O%": "ore_k2o_pct",
    "ORE_NA2O%": "ore_na2o_pct",
    "ORE_TM%": "ore_tm_pct",
    "ORE_LOI%": "ore_loi_pct",
    "ORE_CALC_MT": "ore_mt",
    # ── Sinter ────────────────────────────────────────────────────────
    "SINTER_SP_02_FE(T)%": "sinter_fe_total_pct",
    "SINTER_SP_02_P%": "sinter_p_pct",
    "SINTER_SP_02_SIO2%": "sinter_sio2_pct",
    "SINTER_SP_02_CAO%": "sinter_cao_pct",
    "SINTER_SP_02_MGO%": "sinter_mgo_pct",
    "SINTER_SP_02_AL2O3%": "sinter_al2o3_pct",
    "SINTER_SP_02_FEO%": "sinter_feo_pct",
    "SINTER_SP_02_MNO%": "sinter_mno_pct",
    "SINTER_SP_02_TIO2%": "sinter_tio2_pct",
    "SINTER_SP_02_K2O%": "sinter_k2o_pct",
    "SINTER_SP_02_NA2O%": "sinter_na2o_pct",
    "SINTER_CALC_MT": "sinter_mt",
    # ── Pellet ────────────────────────────────────────────────────────
    "LLOYDS_PELLET_PCT_FE2O3": "lloyds_pellet_pct_fe2o3",
    "LLOYDS_PELLET_PCT_SIO2": "lloyds_pellet_pct_sio2",
    "LLOYDS_PELLET_PCT_CAO": "lloyds_pellet_pct_cao",
    "LLOYDS_PELLET_PCT_MGO": "lloyds_pellet_pct_mgo",
    "LLOYDS_PELLET_PCT_AL2O3": "lloyds_pellet_pct_al2o3",
    "LLOYDS_PELLET_PCT_MNO": "lloyds_pellet_pct_mno",
    "LLOYDS_PELLET_PCT_K2O": "lloyds_pellet_pct_k2o",
    "LLOYDS_PELLET_PCT_NA2O": "lloyds_pellet_pct_na2o",
    "LLOYDS_PELLET_PCT_TIO2": "lloyds_pellet_pct_tio2",
    "LLOYDS_PELLET_PCT_P": "lloyds_pellet_pct_p",
    "LLOYDS_PELLET_PCT_TM": "lloyds_pellet_pct_tm",
    "LLOYDS_PELLET_PCT_LOI": "lloyds_pellet_pct_loi",
    "TOTAL_PELLET_CALC_MT": "lloyds_pellet_mt",
    # ── Flux ──────────────────────────────────────────────────────────
    "FLUX_SIO2%": "flux_sio2_pct",
    "FLUX_FE2O3%": "flux_fe2o3_pct",
    "FLUX_AL2O3%": "flux_al2o3_pct",
    "FLUX_MGO%": "flux_mgo_pct",
    "FLUX_CAO%": "flux_cao_pct",
    "FLUX_TM%": "flux_tm_pct",
    "FLUX_LOI%": "flux_loi_pct",
    "FLUX_CALC_MT": "flux_mt",
    # ── Hot Metal chemistry ───────────────────────────────────────────
    "CHEM_PCT_FE": "chem_pct_fe",
    "CHEM_PCT_C": "chem_pct_c",
    "CHEM_PCT_SI": "chem_pct_si",
    "CHEM_PCT_MN": "chem_pct_mn",
    "CHEM_PCT_P": "chem_pct_p",
    "CHEM_PCT_S": "chem_pct_s",
    "CHEM_PCT_TI": "chem_pct_ti",
    "CHEM_PCT_CR": "chem_pct_cr",
    # ── Slag chemistry ────────────────────────────────────────────────
    "SLAG_PCT_SIO2": "slag_pct_sio2",
    "SLAG_PCT_CAO": "slag_pct_cao",
    "SLAG_PCT_MGO": "slag_pct_mgo",
    "SLAG_PCT_AL2O3": "slag_pct_al2o3",
    "SLAG_PCT_FEO": "slag_pct_feo",
    "SLAG_PCT_MNO": "slag_pct_mno",
    "SLAG_PCT_K2O": "slag_pct_k2o",
    "SLAG_PCT_NA2O": "slag_pct_na2o",
    "SLAG_PCT_TIO2": "slag_pct_tio2",
    "SLAG_PCT_S": "slag_pct_s",
    # ── Blast / process parameters ────────────────────────────────────
    "HOT BLAST VOLUMENM3/HR.": "hot_blast_vol_nm3h",
    "HOT BLAST PRESSUREBAR": "hot_blast_press",
    "HOT BLAST TEMP.OC": "hot_blast_temp",
    "O2 ENRICHMENT %": "oxygen_enrichment_pct",
    "STEAMKGS/HR.": "steam_injection",
    "FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%": "co_pct",
    "FURNACE TOP GAS ANALYSISCO2%": "co2_pct",
    "FURNACE TOP GAS ANALYSISH2%": "h2_pct",
    "TOPPRESSUREBAR": "top_press_avg",
    "PRODUCTIONTONNESPERHR": "production_per_hour",
    "FURNACETOPGASANALYSISCO2ETACO": "body_etaco",
    "OXYGENFLOWNM3/HR.": "oxygen_flow_nm3h",
}

# Set of schema names that represent summed mass columns (for reference by
# other modules wanting to differentiate tonnages from compositions).
SCHEMA_MASS_COLS: Set[str] = {CSV_TO_SCHEMA[c] for c in MASS_CSV_COLS}


# ---------------------------------------------------------------------------
# IST ↔ UTC conversion helper
# ---------------------------------------------------------------------------


def get_day_window_utc(day: date) -> Tuple[datetime, datetime]:
    """Return ``(start_utc, end_utc)`` for one IST calendar day.

    The window is closed-open ``[00:00 IST, 24:00 IST)`` converted to
    UTC.  Example: 2026-04-07 IST → ``(2026-04-06 18:30Z, 2026-04-07 18:30Z)``.

    Args:
        day (date): IST calendar date.

    Returns:
        tuple: ``(start_utc, end_utc)`` as timezone-aware datetimes.
    """
    start_local = IST.localize(datetime(day.year, day.month, day.day, 0, 0, 0))
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Static CSV loader (globally cached — reloaded only on full server restart)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_static_csv() -> pd.DataFrame:
    """Load, timestamp-parse, and rename the full ML dataset CSV.

    No TTL — the CSV does not change between page reruns. The in-memory
    DataFrame uses schema-compatible column names so all downstream
    compute functions work without column translation.

    Returns:
        pd.DataFrame: Full renamed dataset with a tz-naive ``timestamp``
        column (treated as IST local time, matching the original data
        recording timezone).  Returns an empty DataFrame if the file is
        missing.
    """
    if not _CSV_PATH.is_file():
        log.error("Static CSV not found at %s", _CSV_PATH)
        return pd.DataFrame()

    df = pd.read_csv(_CSV_PATH, low_memory=False)
    # The first column is the unlabelled timestamp index from pandas.
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    # Timestamps in the CSV are IST-naive (no timezone suffix).
    # Keep them naive — filtering is done by date comparison.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.rename(columns=CSV_TO_SCHEMA, inplace=True)
    return df


def _window_from_csv(
    df: pd.DataFrame,
    day: date,
    lag_hours: int = 0,
) -> pd.DataFrame:
    """Slice rows for one calendar day (with optional lag) from the CSV.

    Because CSV timestamps are IST-naive, we filter by the **date
    component** of the timestamp rather than a UTC window.  A lag
    expressed in hours is converted to whole days (rounded down) for
    the date comparison; use multiples of 24 h for exact day shifts.

    Args:
        df (pd.DataFrame): Full renamed dataset from :func:`_load_static_csv`.
        day (date): IST calendar date (unshifted target).
        lag_hours (int): Hours to shift the window backward (positive =
            older data).  E.g. ``lag_hours=96`` looks 4 days earlier.
            Sub-day lags (e.g. 4 h) are rounded to the nearest day for
            daily analysis.

    Returns:
        pd.DataFrame: Rows whose timestamp falls on the target date.
    """
    target_date = day - timedelta(hours=lag_hours)
    mask = df["timestamp"].dt.date == target_date
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Public static-CSV fetchers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_static_rm_for_day(day: date, lag_hours: int = 0) -> pd.DataFrame:
    """Return a 1-row DataFrame with RM composition + daily tonnages from CSV.

    **Mass columns** (``*_mt``) are **summed** over the day window because
    each CSV row holds the hourly charged tonnes.  **Composition columns**
    are **averaged** (wt% values).

    Args:
        day (date): IST calendar date.
        lag_hours (int): Hours to shift the RM window backward (default 0).
            Positive values look further into the past; e.g. ``96`` uses
            data from 4 days prior.

    Returns:
        pd.DataFrame: 1-row DataFrame with schema-compatible column names.
        ``df.attrs["n_rows"]`` stores the hourly-row count (≤24).
        Returns an empty DataFrame when no data is available.
    """
    df = _load_static_csv()
    if df.empty:
        return pd.DataFrame()

    window = _window_from_csv(df, day, lag_hours)
    if window.empty:
        return pd.DataFrame()

    result: Dict[str, float] = {}
    for col in window.columns:
        if col == "timestamp":
            continue
        series = pd.to_numeric(window[col], errors="coerce")
        if col in SCHEMA_MASS_COLS:
            v = series.sum(skipna=True)
        else:
            v = series.mean(skipna=True)
        result[col] = float(v) if pd.notna(v) else float("nan")

    out = pd.DataFrame([result])
    out.attrs["n_rows"] = len(window)
    out.attrs["lag_hours"] = lag_hours
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_static_hm_slag_for_day(day: date) -> pd.DataFrame:
    """Return 1-row HM + slag chemistry averaged over the selected day.

    Output chemistry (``chem_pct_*``, ``slag_pct_*``) is always taken
    from the selected day with no lag — it represents what was actually
    tapped on that day.

    Args:
        day (date): IST calendar date.

    Returns:
        pd.DataFrame: 1-row DataFrame with schema-compatible column names.
        Returns empty on no data.
    """
    df = _load_static_csv()
    if df.empty:
        return pd.DataFrame()

    window = _window_from_csv(df, day)
    if window.empty:
        return pd.DataFrame()

    avg = window.mean(numeric_only=True).to_frame().T
    avg.attrs["n_rows"] = len(window)
    return avg


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_static_online_for_day(day: date, lag_hours: int = 0) -> Dict[str, float]:
    """Return blast / process-param averages from the CSV as a flat dict.

    Args:
        day (date): IST calendar date.
        lag_hours (int): Hours to shift the blast window backward (default 0).

    Returns:
        dict: ``{schema_field_name: mean_value}`` for every field in
        :data:`PROCESS_FIELDS`.  Returns ``{}`` on failure or empty data.
    """
    df = _load_static_csv()
    if df.empty:
        return {}

    window = _window_from_csv(df, day, lag_hours)
    if window.empty:
        return {}

    out: Dict[str, float] = {}
    for field in PROCESS_FIELDS:
        if field in window.columns:
            v = pd.to_numeric(window[field], errors="coerce").mean(skipna=True)
            out[field] = float(v) if pd.notna(v) else 0.0
        else:
            out[field] = 0.0
    return out


def get_csv_date_range() -> Tuple[date | None, date | None]:
    """Return the earliest and latest IST dates available in the static CSV.

    Returns:
        tuple: ``(min_date, max_date)`` as :class:`datetime.date` objects,
        or ``(None, None)`` if the CSV is empty or unreadable.
    """
    df = _load_static_csv()
    if df.empty or "timestamp" not in df.columns:
        return None, None
    ts = df["timestamp"].dropna()
    if ts.empty:
        return None, None
    # Timestamps are IST-naive — dt.date gives the IST calendar date directly.
    dates = ts.dt.date
    return dates.min(), dates.max()


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


def clear_day_caches(day: date) -> None:
    """Invalidate every per-day cached fetch for a given IST date.

    Called by the Refresh button on the page to force re-reading both the
    static CSV window and the (optional) InfluxDB paths.

    Args:
        day (date): IST calendar date whose caches should be cleared.
    """
    for fn in (
        fetch_static_rm_for_day,
        fetch_static_hm_slag_for_day,
        fetch_static_online_for_day,
        fetch_rm_for_day,
        fetch_hm_slag_for_day,
        fetch_dpr_for_day,
        fetch_online_aggregates_for_day,
    ):
        try:
            fn.clear()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# InfluxDB fetchers — secondary / live-data path (kept for future use)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rm_for_day(day: date) -> pd.DataFrame:
    """Fetch ``rm_updated_data`` for one IST day from InfluxDB (offline bucket).

    Legacy path — the page uses :func:`fetch_static_rm_for_day` by default.

    Args:
        day (date): IST calendar date.

    Returns:
        pd.DataFrame: 1-row shift-averaged DataFrame, or empty on failure.
    """
    start_utc, end_utc = get_day_window_utc(day)
    try:
        df = fetch_offline_data(RM_MEASUREMENT, (start_utc, end_utc), OFFLINE_DB)
    except Exception as exc:
        log.warning("fetch_rm_for_day(%s) failed: %s", day, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = clean_rm_data(df)
    avg = df.mean(numeric_only=True).to_frame().T
    avg.attrs["n_shifts"] = int(len(df))
    return avg


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hm_slag_for_day(day: date) -> pd.DataFrame:
    """Fetch hot-metal & slag chemistry from InfluxDB (offline bucket).

    Args:
        day (date): IST calendar date.

    Returns:
        pd.DataFrame: 1-row day-averaged DataFrame, or empty on failure.
    """
    start_utc, end_utc = get_day_window_utc(day)
    try:
        df = fetch_offline_data(HM_SLAG_MEASUREMENT, (start_utc, end_utc), OFFLINE_DB)
    except Exception as exc:
        log.warning("fetch_hm_slag_for_day(%s) failed: %s", day, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    avg = df.mean(numeric_only=True).to_frame().T
    avg.attrs["n_rows"] = int(len(df))
    return avg


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dpr_for_day(day: date) -> pd.DataFrame:
    """Fetch daily production report row(s) from InfluxDB (offline bucket).

    Args:
        day (date): IST calendar date.

    Returns:
        pd.DataFrame: Raw DPR row(s), or empty on failure.
    """
    start_utc, end_utc = get_day_window_utc(day)
    try:
        df = fetch_offline_data(DPR_MEASUREMENT, (start_utc, end_utc), OFFLINE_DB)
    except Exception as exc:
        log.warning("fetch_dpr_for_day(%s) failed: %s", day, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_online_aggregates_for_day(day: date) -> Dict[str, float]:
    """Fetch ``process_params`` day-averages from InfluxDB (live bucket).

    Args:
        day (date): IST calendar date.

    Returns:
        dict: ``{field_name: mean_value}`` for every field in
        :data:`PROCESS_FIELDS`.  Returns ``{}`` on failure.
    """
    start_utc, end_utc = get_day_window_utc(day)
    try:
        fetcher = BaseDataFetcher(
            variable_tag="process_params",
            database="bf2_evonith_raw",
            token="INFLUX_ONLINE_TOKEN",
        )
        df = fetcher.fetch_averaged_data(
            recent_data_of="over selected range",
            start_time=start_utc,
            end_time=end_utc,
            request_type="windowed-average",
            window_by="1h",
        )
    except Exception as exc:
        log.warning("fetch_online_aggregates_for_day(%s) failed: %s", day, exc)
        return {}

    if df is None or len(df) == 0:
        return {}

    out: Dict[str, float] = {}
    for field in PROCESS_FIELDS:
        if field in df.columns:
            try:
                v = pd.to_numeric(df[field], errors="coerce").mean()
                out[field] = float(v) if pd.notna(v) else 0.0
            except Exception:  # noqa: BLE001
                out[field] = 0.0
        else:
            out[field] = 0.0
    return out
