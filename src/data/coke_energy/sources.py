"""Data for the Coke Energy Model page, in the formats the web model was built on.

The web model (``utils.coke_energy``) was developed on four exported files. These loaders produce the same frames from
the app's own sources, so ``build_shift_inputs`` runs unchanged:

  hourly dataset   furnace_dataset.csv (the StaticDatasetManager file BMO uses): charges and raw-material composition,
                   ``time`` in IST. The web export was this same file (furnace_dataset_08_10_26.csv).
  10-min online    InfluxDB process_params / heatload_delta_t / delta_t / cooling_water, ``ts`` = IST bin START. The
                   web export was all_measurements_10min_*.csv.
  HM / slag taps   Neon offline_feed.hot_metal_slag_analysis, ``t`` = IST record time (hm_slag_*.csv).
  DPR              Neon offline_feed.dpr_data, one row per IST date: hot metal, dust-catcher (flue) dust, GCP dust
                   (the web read them from the DPR workbook).

10-min data for a whole year is the slow part (about 3 s per month), so it is cached on disk one calendar month per file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PLANT_TZ = "Asia/Kolkata"
WINDOW = pd.Timedelta(minutes=10)
REPO_ROOT = Path(__file__).resolve().parents[3]

# DPR columns -> the names the web DPR reader produced (dpr.py). Rates are per tonne of DPR hot metal.
_DPR_TONNES = {
    "total_hot_metal_mt": "hm_t",
    "flue_dust_mt": "dc_dust_t",
    "gcp_dust_mt": "gcp_dust_t",
    "slag_generation_mt": "slag_t",
    "pci_mt": "pci_t",
}


def repo_path(path_str: str | Path) -> Path:
    """Resolve a configured path against the repository root (the app may run from the root or from src/)."""

    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _ist_to_utc(ts: pd.Timestamp) -> datetime:
    return pd.Timestamp(ts).tz_localize(PLANT_TZ).tz_convert("UTC").to_pydatetime()


def _to_ist_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(PLANT_TZ).tz_localize(None)


# --------------------------------------------------------------------------------------------- hourly dataset
def load_hourly_dataset(csv_path: str | Path) -> pd.DataFrame:
    """The hourly furnace dataset with a ``time`` column (IST, tz-naive), sorted."""

    df = pd.read_csv(repo_path(csv_path), low_memory=False)
    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if getattr(df["time"].dt, "tz", None) is not None:
        df["time"] = df["time"].dt.tz_convert(PLANT_TZ).dt.tz_localize(None)
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def static_dataset_manager(data_cfg: Mapping[str, Any]):
    """The shared StaticDatasetManager for the hourly dataset (same file and URL as BMO)."""

    from data.ml.static_dataset_manager import StaticDatasetManager

    return StaticDatasetManager(
        repo_path(data_cfg.get("static_dataset_path", "src/assets/data/furnace_dataset.csv")),
        remote_url=str(data_cfg.get("static_dataset_url", "") or "").strip() or None,
    )


def static_dataset_status(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Freshness of the local hourly dataset: when it was fetched and where its data ends."""

    manager = static_dataset_manager(data_cfg)
    meta = manager.get_meta()
    path = manager.current_csv_path()
    last_updated = None
    if meta is not None and meta.last_updated:
        try:
            last_updated = datetime.fromisoformat(str(meta.last_updated))
        except ValueError:
            last_updated = None
    max_age = int(data_cfg.get("static_refresh_max_age_minutes", 60) or 60)
    now = datetime.now(last_updated.tzinfo) if last_updated and last_updated.tzinfo else datetime.now()
    stale = last_updated is None or now - last_updated >= timedelta(minutes=max_age)
    if meta is None or str(meta.source_url or "").strip() != (manager.remote_url or ""):
        stale = True
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return {
        "path": path,
        "exists": path.exists(),
        "mtime_ns": int(mtime_ns),
        "last_updated": last_updated,
        "data_end": manager.get_csv_end_timestamp() if path.exists() else None,
        "stale": stale,
        "max_age_minutes": max_age,
    }


def refresh_static_dataset(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fetch the published hourly dataset (or rebuild it from the database when no URL is set) and save it."""

    manager = static_dataset_manager(data_cfg)
    try:
        df = manager.update_static("Full")
        saved = manager.save(df)
        return {"refreshed": True, "path": saved}
    except Exception as exc:  # noqa: BLE001 - the page falls back to the local copy
        log.warning("hourly dataset refresh failed: %s", exc)
        return {"refreshed": False, "error": str(exc)}


# --------------------------------------------------------------------------------------------- 10-min online data
def _fetch_10min_window(start: pd.Timestamp, end: pd.Timestamp, fields: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """[start, end) in IST, 10-min means, ``ts`` = bin start."""

    from furnace_data.influx.online import fetch_online_df

    # fetch_online_df labels bins at their END and overwrites the last label with the request end, so ask for one
    # extra bin and drop it.
    raw = fetch_online_df(
        list(fields),
        time_range="last 1 week",
        request_type="windowed-average",
        window_by="10 minutes",
        start_time_override=_ist_to_utc(start),
        end_time_override=_ist_to_utc(end + WINDOW),
        column_naming="field",
        fields_by_measurement={k: list(v) for k, v in fields.items()},
    )
    columns = [f for fs in fields.values() for f in fs]
    if raw is None or raw.empty or not isinstance(raw.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["ts", *columns])
    raw = raw.iloc[:-1]
    ts = _to_ist_naive(raw.index) - WINDOW
    out = raw.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    out.insert(0, "ts", ts)
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    out = out[(out.ts >= start) & (out.ts < end)]
    return out[["ts", *columns]].reset_index(drop=True)


def fetch_10min(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    fields: Mapping[str, Sequence[str]],
    cache_dir: str | Path,
    refresh_minutes: float = 30.0,
    now: pd.Timestamp | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """10-min online data for [start, end) (IST, tz-naive), cached one calendar month per parquet file.

    A month that ended before its cache file was written is final and is never fetched again. The running month (and a
    month whose file was written before the month ended) is re-fetched when the file is older than ``refresh_minutes``.
    """

    now = pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None) if now is None else pd.Timestamp(now)
    now_floor = now.floor("10min")
    end = min(pd.Timestamp(end), now_floor)
    cache = repo_path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    columns = [f for fs in fields.values() for f in fs]
    frames: list[pd.DataFrame] = []
    for m0 in pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), end, freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        path = cache / f"{m0:%Y-%m}.parquet"
        written = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").tz_convert(PLANT_TZ).tz_localize(None) if path.exists() else None
        final = written is not None and written >= m1 + pd.Timedelta(hours=1)
        fresh = written is not None and now - written < pd.Timedelta(minutes=refresh_minutes)
        frame = None
        if final or fresh:
            try:
                frame = pd.read_parquet(path)
                if set(columns) - set(frame.columns):
                    frame = None          # the field contract changed: fetch again
            except Exception:  # noqa: BLE001 - a corrupt cache file is simply re-fetched
                frame = None
        if frame is None:
            if progress:
                progress(f"Fetching 10-min online data for {m0:%b %Y}")
            frame = _fetch_10min_window(m0, min(m1, now_floor), fields)
            if not frame.empty:
                frame.to_parquet(path, index=False)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["ts", *columns])
    m10 = pd.concat(frames, ignore_index=True)
    m10 = m10[(m10.ts >= pd.Timestamp(start)) & (m10.ts < end)]
    return m10.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------------------------- Neon
def fetch_hm_slag(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Tap-wise hot metal and slag analyses with ``t`` = IST record time (the web CSV's 'time (IST)')."""

    from furnace_data.offline import fetch_offline_data

    raw = fetch_offline_data(
        "hot_metal_slag_analysis",
        time_range=(_ist_to_utc(start), _ist_to_utc(end)),
        query_type="raw",
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["t"])
    cols = [c for c in raw.columns if c.startswith(("chem_pct", "slag_")) or c == "hmt_gt_1480c"]
    out = raw[cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    out.insert(0, "t", _to_ist_naive(raw.index))
    return out.sort_values("t").reset_index(drop=True)


def fetch_dpr_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One row per IST date with the columns the web DPR reader produced (hm_t, dc_dust_t, gcp_dust_t, rates)."""

    from furnace_data.offline import fetch_offline_data

    raw = fetch_offline_data(
        "dpr_data",
        time_range=(_ist_to_utc(start), _ist_to_utc(end)),
        query_type="raw",
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["hm_t", "dc_dust_t", "gcp_dust_t"])
    raw = raw.copy()
    raw["_date"] = _to_ist_naive(raw.index).normalize()
    if "created_at" in raw.columns:
        raw = raw.sort_values("created_at")
    raw = raw.drop_duplicates("_date", keep="last").set_index("_date").sort_index()
    num = raw.apply(pd.to_numeric, errors="coerce")

    def total(prefix: str, count: int) -> pd.Series:
        cols = [f"{prefix}_{i}_mt" for i in range(1, count + 1) if f"{prefix}_{i}_mt" in num.columns]
        return num[cols].fillna(0.0).sum(axis=1) if cols else pd.Series(np.nan, index=num.index)

    d = pd.DataFrame({target: num[source] if source in num.columns else np.nan for source, target in _DPR_TONNES.items()},
                     index=num.index)
    d["coke_t"] = total("coke", 2)
    d["nut_t"] = total("nut_coke", 2)
    d = d[d.hm_t > 0]
    d["coke_rate_dpr"] = d.coke_t / d.hm_t * 1000
    d["nut_rate_dpr"] = d.nut_t / d.hm_t * 1000
    d["pci_rate_dpr"] = d.pci_t / d.hm_t * 1000
    d["slag_rate_dpr"] = d.slag_t / d.hm_t * 1000
    d["etaco_dpr"] = np.nan        # not in dpr_data; informational only in the web model
    d["delay_min"] = np.nan
    d.index.name = "date"
    return d


def extend_dpr_provisional(dpr: pd.DataFrame, through: pd.Timestamp, *, lookback_days: int = 7,
                           max_days: int = 3) -> tuple[pd.DataFrame, set]:
    """Carry the DPR dust rates forward over the dates the DPR has not reached yet.

    The DPR is entered a day or two after the date, and the web model's shift filter needs that day's dust-catcher and
    GCP dust. Dates after the last DPR row, up to ``through`` and at most ``max_days`` of them, get the median dust per
    tonne of the last ``lookback_days`` DPR rows. Dust is a small term (about 30 kg/tHM, a few kg of carbon), so the
    shifts stay usable; the returned dates let the page flag them as provisional.
    """

    if dpr.empty:
        return dpr, set()
    last = pd.Timestamp(dpr.index.max())
    dates = pd.date_range(last + pd.Timedelta(days=1), pd.Timestamp(through).normalize(), freq="D")[:max_days]
    if len(dates) == 0:
        return dpr, set()
    recent = dpr[dpr.index > last - pd.Timedelta(days=lookback_days)]
    hm = float(recent.hm_t.median())
    fill = pd.DataFrame(index=pd.DatetimeIndex(dates, name=dpr.index.name))
    fill["hm_t"] = hm
    fill["dc_dust_t"] = float((recent.dc_dust_t / recent.hm_t).median()) * hm
    fill["gcp_dust_t"] = float((recent.gcp_dust_t / recent.hm_t).median()) * hm
    for c in dpr.columns:
        if c not in fill.columns:
            fill[c] = np.nan
    return pd.concat([dpr, fill[dpr.columns]]), {d.date() for d in dates}


# --------------------------------------------------------------------------------------------- freshness
@dataclass(frozen=True)
class SourceFreshness:
    """Latest timestamp each source holds (IST), for the page's status line."""

    hourly_end: pd.Timestamp | None
    online_end: pd.Timestamp | None
    hm_slag_end: pd.Timestamp | None
    dpr_end: pd.Timestamp | None


def freshness(fd: pd.DataFrame, m10: pd.DataFrame, hm: pd.DataFrame, dpr: pd.DataFrame) -> SourceFreshness:
    def last(series: pd.Series | pd.Index) -> pd.Timestamp | None:
        return None if len(series) == 0 else pd.Timestamp(pd.Series(series).max())

    return SourceFreshness(
        hourly_end=last(fd["time"]) if "time" in fd else None,
        online_end=last(m10["ts"]) if "ts" in m10 else None,
        hm_slag_end=last(hm["t"]) if "t" in hm else None,
        dpr_end=last(dpr.index),
    )
