"""TRACE: burnt coke from the gas-side oxygen balance, for the API and the worker (SV-1.10, #290).

TRACE is page 10's burnt coke. Raw (gas-balance coke, GBC): the oxygen balance solved for the dry coke rate at the
metered blast air with the analyser's eta CO and eta H2 (``utils.coke_energy.burnt.k_burnt``). Calibrated: raw plus the
14-day correction, the median of (charged - raw burnt) over the stable shifts dated in [d - 14 d, d), at least
``min_shifts`` of them (``utils.coke_energy.pipeline.burnt_coke``). Every quantity here is read from the Coke Energy
pipeline result (``services.coke_energy.page_data``) or composed by the pipeline's own modules; nothing is recomputed:

    per shift    the pipeline's shift table (K_burnt_raw, burnt_offset, K_burnt, K_charged): page 10's values
    per day      production-weighted mean of the plant day's three shifts (``accuracy.window_means``, w = 3; a day
                 counts only with all three shifts, the web model's rule)
    per hour     the twin's 8-h windows ending at each full hour (``twin.snapshot``: k_trace, k_burnt_raw_kg_thm,
                 burnt_offset_kg_thm), equal to the shift values on the shift windows (tests/test_twin_window.py)
    correction   the latest shift's offset, its 14-day window and the stable shifts the median runs over
    feeds        the oxygen balance's inputs of the latest shift and the age of every source as on page 10
                 (``freshness.assess``), aged again at each request from the last sample times stored with the view
    agreement    burnt against charged coke per averaging window (``transit.summary``: Lin's CCC, Bland-Altman bias
                 and limits of agreement) with the per-shift and per-week pairs (``transit.window_pairs``)

A cold build reads 15 months of 10-min data and balances every shift, so a request never builds: the worker job
``twin.trace`` (every hour at :20) builds the view (``refresh``) into ``src/storage/coke_energy/trace_view.json`` and
the API reads that file (``load``). The view is plain JSON: coke rates in kg/tHM dry, masses in t, times UTC (ISO 8601).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from services import coke_energy as ce
from services import twin_views
from utils.coke_energy import transit as tr
from utils.coke_energy.accuracy import window_means
from utils.coke_energy.freshness import assess, in_use_between, item_label, parse_limits
from utils.coke_energy.pipeline import PipelineSettings
from utils.memo import memo

PLANT_TZ = ce.PLANT_TZ
VERSION = 1
JOB_KIND = "twin.trace"
STORE_PATH = Path(__file__).resolve().parents[1] / "storage" / "coke_energy" / "trace_view.json"
#: The view is rebuilt every hour; older than this it is stale (one missed build and the time of a cold build).
STALE_AFTER_S = 2 * 3600.0
SHIFT = pd.Timedelta(hours=8)
#: Pipeline shift-table column -> view field.
FIELDS = {"K_burnt": "k_burnt", "K_burnt_raw": "k_burnt_raw", "burnt_offset": "burnt_offset", "K_charged": "k_charged"}
#: Snapshot field or quantity -> view field (the twin's window values of the same quantities).
HOUR_FIELDS = {"k_trace": "k_burnt", "k_burnt_raw_kg_thm": "k_burnt_raw", "burnt_offset_kg_thm": "burnt_offset"}
#: Agreement bases of the page -> label of the window in ``transit.WINDOWS``.
BASES = {"shift": "Shift (8 h)", "week": "1 week"}
#: Inputs of the oxygen balance shown for the latest shift: (shift-table column, key, label, unit, digits, source).
#: ``source`` is the freshness source the value rests on (``utils.coke_energy.freshness.SOURCES``) or a fixed origin.
INPUTS: tuple[tuple[str, str, str, str, int, str], ...] = (
    ("blast_air_nm3thm", "blast_air", "Blast air, metered (hot blast − O₂)", "Nm³/tHM", 0, "online_10min"),
    ("O2_nm3thm", "oxygen", "Oxygen enrichment", "Nm³/tHM", 1, "online_10min"),
    ("eta_meas", "eta_co", "ηCO, analyser", "%", 2, "online_10min"),
    ("eta_H2", "eta_h2", "ηH₂, analyser", "%", 2, "online_10min"),
    ("HBT", "hot_blast", "Hot blast temperature", "°C", 0, "online_10min"),
    ("T_top", "top_gas", "Top gas temperature", "°C", 0, "online_10min"),
    ("humidity_g", "blast_moisture", "Blast moisture", "g/Nm³", 1, "open_meteo"),
    ("prod_th", "production", "Production", "t/h", 1, "hourly_dataset"),
    ("hm_t", "hot_metal", "Hot metal in the shift", "t", 0, "hourly_dataset"),
    ("K_charged", "coke_charged", "Coke charged (dry)", "kg/tHM", 1, "hourly_dataset"),
    ("pci_kgthm_dry", "pci", "PCI (dry)", "kg/tHM", 1, "hourly_dataset"),
    ("nut_kgthm_dry", "nut_coke", "Nut coke (dry)", "kg/tHM", 1, "hourly_dataset"),
    ("HMT", "hmt", "Hot metal temperature", "°C", 0, "hm_slag"),
    ("hm_si", "hm_si", "Hot metal Si", "%", 2, "hm_slag"),
    ("slag_al2o3_pct", "slag_al2o3", "Slag Al₂O₃", "%", 1, "hm_slag"),
    ("slag_superheat_c", "slag_superheat", "Slag above hot metal (plant rule)", "°C", 0, "plant_rule"),
)


# --------------------------------------------------------------------------------------------- values
def number(value: Any) -> float | None:
    """A finite float, else None (NaN and missing values become null in the view)."""
    return twin_views.number(value)


def utc_text(ts: Any) -> str | None:
    """An IST wall-clock time (tz-naive) or an aware time as ISO 8601 UTC; None for a missing time."""
    if ts is None:
        return None
    stamp = pd.Timestamp(ts)
    if pd.isna(stamp):
        return None
    stamp = stamp.tz_localize(PLANT_TZ) if stamp.tzinfo is None else stamp
    return stamp.tz_convert("UTC").isoformat()


def _ist(text: str | None) -> pd.Timestamp | None:
    """ISO 8601 UTC from the view -> IST wall-clock time (tz-naive), the pipeline's convention."""
    if text is None:
        return None
    return pd.Timestamp(text).tz_convert(PLANT_TZ).tz_localize(None)


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None)


# --------------------------------------------------------------------------------------------- per shift and day
def shift_rows(T: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per balanced shift, oldest first: the shift's window and the pipeline's coke values (page 10's)."""
    if T is None or T.empty:
        return []
    rows = []
    for skey, r in T.sort_values("t_mid").iterrows():
        start = pd.Timestamp(r["t_mid"]) - SHIFT / 2
        rows.append(
            {
                "skey": str(skey),
                "shift": str(r["shift"]),
                "start": utc_text(start),
                "end": utc_text(start + SHIFT),
                "stable": bool(r.get("stable", False)),
                "provisional": bool(r.get("provisional", False)),
                **{field: number(r.get(col)) for col, field in FIELDS.items()},
                "hm_t": number(r.get("hm_t")),
                "prod_th": number(r.get("prod_th")),
            }
        )
    return rows


def day_rows(T: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per plant day (A 06:00 to C 06:00 next day), oldest first: the production-weighted mean of each coke
    value over the day's three shifts (``accuracy.window_means`` with w = 3; null when a shift lacks the value)."""
    if T is None or T.empty:
        return []
    cols = list(FIELDS)
    means = {c: window_means(T, 3, [c]) for c in cols}
    index = means[cols[0]].index
    rows = []
    for t_mid in index:
        day = (pd.Timestamp(t_mid) - pd.Timedelta(hours=18)).normalize()  # the day's A starts 06:00, its mid is 18:00
        start = day + pd.Timedelta(hours=6)
        rows.append(
            {
                "day": f"{day:%Y-%m-%d}",
                "start": utc_text(start),
                "end": utc_text(start + 3 * SHIFT),
                "n_shifts": int(means["K_burnt_raw"].loc[t_mid, "n"]),
                **{FIELDS[c]: number(means[c].loc[t_mid, c]) for c in cols},
            }
        )
    return rows


def latest_key(T: pd.DataFrame) -> str | None:
    """The latest balanced shift with a burnt coke (raw)."""
    if T is None or T.empty or "K_burnt_raw" not in T:
        return None
    have = T[T["K_burnt_raw"].notna()].sort_values("t_mid")
    return None if have.empty else str(have.index[-1])


def oxygen_inputs(row: pd.Series) -> list[dict[str, Any]]:
    """The oxygen balance's inputs of one shift with their units and the source each rests on."""
    return [
        {"key": key, "label": label, "value": number(row.get(col)), "unit": unit, "digits": digits, "source": source}
        for col, key, label, unit, digits, source in INPUTS
    ]


# --------------------------------------------------------------------------------------------- correction
def correction(T: pd.DataFrame, settings: PipelineSettings) -> dict[str, Any] | None:
    """The 14-day correction of the latest shift that has one: the offset (pipeline ``burnt_offset``), its window of
    dates [d - window_days, d) and the stable shifts in it with their (charged - raw burnt), as the pipeline pools them
    (``pipeline.burnt_coke``: stable shifts by date, at least ``min_shifts`` rows)."""
    if T is None or T.empty or "burnt_offset" not in T:
        return None
    have = T[T["burnt_offset"].notna()].sort_values("t_mid")
    if have.empty:
        return None
    skey = str(have.index[-1])
    day = pd.Timestamp(have.loc[skey, "sdate"]).normalize()
    first = day - pd.Timedelta(days=settings.window_days)
    sdate = pd.to_datetime(T["sdate"])
    pool = T[T["stable"].eq(True) & (sdate >= first) & (sdate < day)].sort_values("t_mid")
    return {
        "skey": skey,
        "offset": number(have.loc[skey, "burnt_offset"]),
        "window_start": f"{first:%Y-%m-%d}",
        "window_end": f"{day:%Y-%m-%d}",
        "window_days": int(settings.window_days),
        "min_shifts": int(settings.min_shifts),
        "n_shifts": int(len(pool)),
        "pool": [
            {
                "skey": str(k),
                "start": utc_text(pd.Timestamp(r["t_mid"]) - SHIFT / 2),
                "k_charged": number(r["K_charged"]),
                "k_burnt_raw": number(r["K_burnt_raw"]),
                "gap": number(r["K_charged"] - r["K_burnt_raw"]),
            }
            for k, r in pool.iterrows()
        ],
    }


# --------------------------------------------------------------------------------------------- feeds and freshness
def reduce_samples(samples: pd.DataFrame, reference: pd.Timestamp) -> pd.DataFrame:
    """The sample times ``freshness.assess`` needs for any reference at or after ``reference``: per source and material
    the last time at or before ``reference`` and every later one (daily records are stamped at the end of the day)."""
    if samples is None or samples.empty:
        return pd.DataFrame(columns=["source", "material", "t"])
    s = samples.dropna(subset=["t"]).copy()
    s["t"] = pd.to_datetime(s["t"])
    before = s[s["t"] <= reference].groupby(["source", "material"], as_index=False)["t"].max()
    after = s[s["t"] > reference][["source", "material", "t"]]
    out = pd.concat([before, after], ignore_index=True).drop_duplicates()
    return out.sort_values(["source", "material", "t"]).reset_index(drop=True)


def in_use_now(charged: pd.DataFrame, reference: pd.Timestamp) -> set[str]:
    """Materials charged in the 24 h before the last charge report (page 10's status line)."""
    last = charged.index.max() + pd.Timedelta(hours=1) if charged is not None and not charged.empty else reference
    return in_use_between(charged if charged is not None else pd.DataFrame(), last - pd.Timedelta(hours=24), last)


def feed_rows(assessed: pd.DataFrame, names: Mapping[str, str]) -> list[dict[str, Any]]:
    """``freshness.assess`` rows as plain values (ages and limits in s, times UTC)."""
    rows = []
    for r in assessed.to_dict("records"):
        age = r["age"]
        rows.append(
            {
                "source": r["source"],
                "material": r["material"],
                "item": item_label(r, names),
                "last": utc_text(r["last"]),
                "age_s": None if age is None or pd.isna(age) else float(age / pd.Timedelta(seconds=1)),
                "limit_s": float(r["limit"] / pd.Timedelta(seconds=1)),
                "stale": bool(r["stale"]),
                "used_by": r["used_by"],
                "feeds_models": bool(r["feeds_models"]),
            }
        )
    return rows


def feeds_at(stored: Mapping[str, Any], cfg: Mapping[str, Any], reference: pd.Timestamp) -> list[dict[str, Any]]:
    """Page 10's freshness table at ``reference`` (IST, at or after the view's build) from the stored sample times."""
    samples = pd.DataFrame(stored.get("samples") or [], columns=["source", "material", "t"])
    samples["t"] = pd.to_datetime(samples["t"], utc=True).dt.tz_convert(PLANT_TZ).dt.tz_localize(None)
    samples["material"] = samples["material"].fillna("")
    limits = parse_limits(cfg.get("freshness_limits"))
    assessed = assess(samples, reference, limits, set(stored.get("in_use") or []))
    return feed_rows(assessed, stored.get("names") or {})


def feeds_now(view: Mapping[str, Any], now: datetime | None = None, cfg: Mapping[str, Any] | None = None) -> list[dict]:
    """Page 10's freshness table as of ``now`` (UTC; default the current time) for a stored view."""
    at = pd.Timestamp(now or datetime.now(UTC))
    at = at.tz_localize("UTC") if at.tzinfo is None else at
    reference = at.tz_convert(PLANT_TZ).tz_localize(None)
    return feeds_at(view["feeds"], ce.config() if cfg is None else cfg, reference)


def stored_feeds(data: ce.PageData, reference: pd.Timestamp) -> dict[str, Any]:
    """What ``feeds_at`` needs, from the page data at ``reference`` (the build time)."""
    reduced = reduce_samples(data.samples, reference)
    return {
        "reference": utc_text(reference),
        "samples": [
            {"source": r["source"], "material": r["material"] or "", "t": utc_text(r["t"])}
            for r in reduced.to_dict("records")
        ],
        "in_use": sorted(in_use_now(data.charged, reference)),
        "names": {str(k): str(v) for k, v in (data.names or {}).items()},
    }


def shift_feeds(data: ce.PageData, cfg: Mapping[str, Any], skey: str) -> list[dict[str, Any]]:
    """Freshness of every source at the end of shift ``skey`` (what that shift's balance could use; page 10's notice
    under the tiles)."""
    T = data.result.shifts
    start = pd.Timestamp(T.loc[skey, "t_mid"]) - SHIFT / 2
    end = start + SHIFT
    limits = parse_limits(cfg.get("freshness_limits"))
    assessed = assess(data.samples, end, limits, in_use_between(data.charged, start, end))
    return feed_rows(assessed, data.names or {})


# --------------------------------------------------------------------------------------------- agreement
def _stats(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "window": str(row["window"]),
        "shifts": int(row["shifts"]),
        "hours": float(row["hours"]),
        "n": int(row["n"]),
        "ccc": number(row["ccc"]),
        "r": number(row["r"]),
        "cb": number(row["cb"]),
        "bias": number(row["bias"]),
        "sd": number(row["sd"]),
        "loa_lo": number(row["loa_lo"]),
        "loa_hi": number(row["loa_hi"]),
        "mean_charged": number(row["mean_x"]),
        "mean_burnt": number(row["mean_y"]),
        "slope": number(row["slope"]),
    }


def agreement_points(T: pd.DataFrame, shifts: int) -> list[dict[str, Any]]:
    """The pairs ``transit.summary`` compares for windows of ``shifts`` shifts, with their Bland-Altman coordinates:
    the mean of burnt and charged and the difference burnt - charged (kg/tHM)."""
    W = tr.window_pairs(T, shifts)
    return [
        {
            "t": utc_text(t),
            "n": int(r["n"]),
            "charged": number(r["K_charged"]),
            "burnt": number(r["K_burnt"]),
            "mean": number((r["K_burnt"] + r["K_charged"]) / 2),
            "diff": number(r["K_burnt"] - r["K_charged"]),
        }
        for t, r in W.iterrows()
    ]


def agreement(T: pd.DataFrame) -> dict[str, Any]:
    """Burnt (calibrated) against charged coke: ``transit.summary`` (window table, variance split, lag, closure on the
    raw burnt coke) and the per-shift and per-week pairs."""
    S = tr.summary(T)
    table = [_stats(r) for r in S["table"].to_dict("records")]
    split = S["split"]
    closure = S["closure"]
    return {
        "table": table,
        "split": {k: number(split.get(k)) for k in ("transit_sd", "random_sd", "floor_sd")},
        "peak_lag_h": number(S["peak_lag_h"]),
        "closure": None
        if closure is None
        else {
            "n": int(closure["n"]),
            "ratio": number(closure["ratio"]),
            "burnt_t": number(closure["burnt_t"]),
            "charged_t": number(closure["charged_t"]),
        },
        "points": {basis: agreement_points(T, tr.WINDOWS[label]) for basis, label in BASES.items()},
    }


def basis_stats(view_agreement: Mapping[str, Any], basis: str) -> dict[str, Any] | None:
    """The window-table row of an agreement basis (``shift`` or ``week``)."""
    label = BASES[basis]
    return next((r for r in view_agreement.get("table", []) if r["window"] == label), None)


# --------------------------------------------------------------------------------------------- the view
def build_view(data: ce.PageData, cfg: Mapping[str, Any], *, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """The TRACE view of one pipeline result (``now``: IST build time, tz-naive)."""
    now = now_ist() if now is None else pd.Timestamp(now)
    result = data.result
    T = result.shifts
    settings = result.settings
    fresh = data.fresh
    key = latest_key(T)
    latest = None
    if key is not None:
        row = T.loc[key]
        latest = {"skey": key, "inputs": oxygen_inputs(row), "feeds": shift_feeds(data, cfg, key)}
    return {
        "version": VERSION,
        "built_at": utc_text(now),
        "settings": {
            "window_days": int(settings.window_days),
            "min_shifts": int(settings.min_shifts),
            "history_start": str(settings.history_start),
        },
        "sources": {
            "hourly_end": utc_text(fresh.hourly_end),
            "online_end": utc_text(fresh.online_end),
            "hm_slag_end": utc_text(fresh.hm_slag_end),
            "dpr_end": utc_text(fresh.dpr_end),
        },
        "errors": len(result.errors),
        "shifts": shift_rows(T),
        "days": day_rows(T),
        "latest": latest,
        "correction": correction(T, settings),
        "feeds": stored_feeds(data, now),
        "agreement": agreement(T) if T is not None and not T.empty else None,
    }


def headline(view: Mapping[str, Any]) -> dict[str, Any]:
    """Job result: build time, shift count and the latest shift's burnt coke."""
    shifts = view.get("shifts") or []
    latest = next((r for r in reversed(shifts) if r["skey"] == (view.get("latest") or {}).get("skey")), None)
    return {
        "built_at": view.get("built_at"),
        "shifts": len(shifts),
        "days": len(view.get("days") or []),
        "errors": view.get("errors"),
        "latest_shift": None if latest is None else latest["skey"],
        "k_burnt": None if latest is None else latest["k_burnt"],
        "k_burnt_raw": None if latest is None else latest["k_burnt_raw"],
    }


def save(view: Mapping[str, Any], path: Path | None = None) -> Path:
    """Write the view atomically (a reader never sees a partial file)."""
    target = Path(path or STORE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".trace_view_", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(view, fh, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


@memo(max_entries=2)
def _read(path: str, mtime_ns: int) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path: Path | None = None) -> dict[str, Any] | None:
    """The stored view (re-read when the file changes), or None before the first build. Treat as read-only."""
    target = Path(path or STORE_PATH)
    try:
        mtime = target.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    view = _read(str(target), mtime)
    return view if view.get("version") == VERSION else None


def refresh(*, nonce: int = 0, path: Path | None = None, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the view from the memoised Coke Energy page data (the worker's, shared with ``twin.evaluate``) and store
    it."""
    cfg = ce.config() if cfg is None else cfg
    settings = PipelineSettings.from_config(cfg)
    data = ce.page_data(ce.data_keys(cfg, nonce), settings)
    view = build_view(data, cfg)
    save(view, path)
    return view


def age_s(view: Mapping[str, Any], now: datetime | None = None) -> float | None:
    built = view.get("built_at")
    if built is None:
        return None
    now = now or datetime.now(UTC)
    return (now - pd.Timestamp(built).to_pydatetime()).total_seconds()


# --------------------------------------------------------------------------------------------- series
def _between(rows: Iterable[Mapping[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Rows whose window ends in [start, end]."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return [dict(r) for r in rows if lo <= pd.Timestamp(r["end"]) <= hi]


def shift_series(view: Mapping[str, Any], start: datetime, end: datetime) -> list[dict[str, Any]]:
    return [
        {
            "start": r["start"],
            "end": r["end"],
            "label": r["skey"],
            "stable": r["stable"],
            "status": None,
            "n_shifts": 1,
            **{f: r[f] for f in FIELDS.values()},
        }
        for r in _between(view.get("shifts") or [], start, end)
    ]


def day_series(view: Mapping[str, Any], start: datetime, end: datetime) -> list[dict[str, Any]]:
    return [
        {
            "start": r["start"],
            "end": r["end"],
            "label": r["day"],
            "stable": None,
            "status": None,
            "n_shifts": r["n_shifts"],
            **{f: r[f] for f in FIELDS.values()},
        }
        for r in _between(view.get("days") or [], start, end)
    ]


def hour_series(snapshots: Sequence[Any]) -> list[dict[str, Any]]:
    """Hourly TRACE from the twin's stored windows (8 h ending at each full hour), oldest first."""
    rows = twin_views.series_rows(
        snapshots, resolution="hour", quantities=["k_burnt_raw_kg_thm", "burnt_offset_kg_thm"]
    )
    out = []
    for r in rows:
        end = r["window_end"]
        q = r["quantities"]
        out.append(
            {
                "start": (pd.Timestamp(end) - SHIFT).isoformat(),
                "end": pd.Timestamp(end).isoformat(),
                "label": r["shift_key"] or f"{pd.Timestamp(end).tz_convert(PLANT_TZ):%Y-%m-%d %H:%M}",
                "stable": None,
                "status": r["status"],
                "n_shifts": None,
                "k_burnt": r["k_trace"],
                "k_burnt_raw": q.get("k_burnt_raw_kg_thm"),
                "burnt_offset": q.get("burnt_offset_kg_thm"),
                "k_charged": r["k_charged"],
            }
        )
    return out


def latest_hour(snapshots: Sequence[Any]) -> dict[str, Any] | None:
    """The newest stored window with a burnt coke (raw), or None."""
    for row in reversed(hour_series(snapshots)):
        if row["k_burnt_raw"] is not None or row["k_burnt"] is not None:
            return row
    return None
