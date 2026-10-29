"""FORGE page data (SV-1.11, #291): the two-zone (Rist) energy balance at the 14-day medians against burnt coke.

FORGE is page 10's energy-balance model (``K_energy``: omega, Q_LZ, Q_UZ and r_H fixed at their medians over the stable
shifts dated in the 14 days before the shift); burnt coke is TRACE (``K_burnt``: the metered blast's oxygen balance
less the 14-day charged - burnt offset). Both come from the Coke Energy pipeline of ``services.coke_energy``, unchanged.

Two halves, so no request runs the pipeline or a balance:

* ``refresh`` (the ``twin.forge`` job, HH:20 IST): the memoised pipeline result, the Rist geometry of the stable shifts
  dated in the ``rist_days`` before the latest one (``coke_energy.build_rist_frames``, page 10's diagram) and the
  stored view (``build_view`` -> ``save``): every balanced shift with its coke rates and calibration, and the frames;
* the API reads the stored view (``load``) and answers from it: the shifts of a period with page 10's accuracy
  statistics (``prediction``: ``accuracy.period_rows`` and ``accuracy.accuracy``) and the statistics over the trailing
  ``stats_days`` of every day; the frames (``rist``); the calibration of a day (``calibration``: the pipeline's
  ``rolling_pool_median`` over the stored stable shifts, so it equals the shift table's ``omega_fixed``, ...).

Coke rates are kg/tHM dry; heat losses GJ/h; shaft efficiency and r_H dimensionless; shift times IST (tz-naive) in
the store and UTC in the API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services import coke_energy as ce
from utils.coke_energy.accuracy import accuracy, period_rows, scored_shifts
from utils.coke_energy.pipeline import PipelineSettings, rolling_pool_median
from utils.coke_energy.rist_diagram import RistShift, rist_dict
from utils.memo import memo

PLANT_TZ = ce.PLANT_TZ
SRC = Path(__file__).resolve().parents[1]
STORE_DIR = SRC / "storage" / "forge"
VIEW_FILE = "view.json"
SCHEMA = 1
MODEL = "FORGE: two-zone (Rist) balance with omega, Q_LZ, Q_UZ and r_H at their 14-day medians (Coke Energy page)"

# Stored per balanced shift: the shift table's columns (PipelineResult.shifts) and r_H of the shift's calibration.
NUMERIC = (
    "K_energy",
    "K_burnt",
    "K_obal",
    "K_charged",
    "omega_fixed",
    "Q_LZ_GJh_fixed",
    "Q_UZ_GJh_fixed",
    "r_H_fixed",
    "omega_shift",
    "Q_LZ_GJh_shift",
    "Q_UZ_GJh_shift",
    "r_H_shift",
)
COLUMNS = ("skey", "sdate", "shift", "t_mid", "stable", "provisional", *NUMERIC)
# The pipeline's calibration columns (``energy_balance_model``: medians of the shift calibrations Z) -> stored names.
CALIBRATION = {"omega": "omega_shift", "Q_LZ_GJh": "Q_LZ_GJh_shift", "Q_UZ_GJh": "Q_UZ_GJh_shift", "r_H": "r_H_shift"}
SOURCES = ("hourly_end", "online_end", "hm_slag_end", "dpr_end")


class ViewUnavailable(LookupError):
    """No stored view (the ``twin.forge`` job has not run yet) or one written by another schema."""


class PlantDataUnavailable(RuntimeError):
    """The Coke Energy pipeline could not run (hourly dataset, InfluxDB or Neon unavailable)."""


@dataclass(frozen=True)
class ForgeSettings:
    stats_days: int = 30  # trailing window of the statistics (page 10's default period)
    rist_days: int = 30  # stable shifts drawn: those dated in these days before the latest one

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> ForgeSettings:
        """``coke_energy.yml``: ``twin.forge`` (stats_days falls back to ``ui.timeline_default_days``)."""
        block = ((cfg.get("twin") or {}).get("forge")) or {}
        ui = cfg.get("ui") or {}
        return cls(
            stats_days=int(block.get("stats_days", ui.get("timeline_default_days", cls.stats_days))),
            rist_days=int(block.get("rist_days", cls.rist_days)),
        )


def number(value: Any) -> float | None:
    """A finite float, else None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _stamp(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()


def to_utc(ist: Any) -> datetime:
    """An IST wall-clock time (tz-naive) as an aware UTC datetime."""
    return pd.Timestamp(ist).tz_localize(PLANT_TZ).tz_convert("UTC").to_pydatetime()


# ---------------------------------------------------------------------------------------------- building (worker)
def shift_records(result: Any) -> list[dict[str, Any]]:
    """Every balanced shift of the pipeline result: date, shift, mid-time (IST), stability, the four coke rates, the
    fixed calibration it was predicted with and its own calibration (the pool of later shifts' medians)."""
    T = result.shifts
    if T is None or T.empty:
        return []
    Z = result.Z if result.Z is not None else pd.DataFrame()
    r_h = Z["r_H"] if "r_H" in Z else pd.Series(dtype=float)
    out = []
    for k, row in T.iterrows():
        rec: dict[str, Any] = {
            "skey": str(k),
            "sdate": pd.Timestamp(row["sdate"]).date().isoformat(),
            "shift": str(row["shift"]),
            "t_mid": pd.Timestamp(row["t_mid"]).isoformat(),
            "stable": bool(row["stable"]),
            "provisional": bool(row.get("provisional", False)),
        }
        rec.update({c: number(row.get(c)) for c in NUMERIC if c != "r_H_shift"})
        rec["r_H_shift"] = number(r_h.get(k))
        out.append(rec)
    return out


def build_view(
    data: ce.PageData,
    settings: PipelineSettings,
    forge_settings: ForgeSettings,
    *,
    built_at: datetime,
    job_id: str | None = None,
    frames_for: Callable[[tuple[str, ...]], Sequence[RistShift]] | None = None,
) -> dict[str, Any]:
    """The stored view of one pipeline result. ``frames_for`` draws the Rist frames of a tuple of shift keys (default:
    ``coke_energy.build_rist_frames`` on ``data``; the worker passes the memoised ``coke_energy.rist_frames``)."""
    result = data.result
    T = result.shifts
    keys: list[str] = []
    if T is not None and not T.empty:
        candidates = ce.rist_candidates(T)
        if not candidates.empty:
            last = candidates.sdate.max()
            keys = ce.rist_keys(T, (last - pd.Timedelta(days=forge_settings.rist_days)).date(), last.date())
    draw = frames_for or (lambda ks: ce.build_rist_frames(data, settings, ks))
    frames = [p for p in (frame_payload(f) for f in (draw(tuple(keys)) if keys else [])) if p is not None]
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "built_at": built_at.astimezone(UTC).isoformat(),
        "job_id": job_id,
        "settings": {
            "window_days": int(settings.window_days),
            "min_shifts": int(settings.min_shifts),
            "history_start": str(settings.history_start),
            "predict_from": str(settings.predict_from),
            "stats_days": forge_settings.stats_days,
            "rist_days": forge_settings.rist_days,
        },
        "sources": {k: _stamp(getattr(data.fresh, k, None)) for k in SOURCES},
        "provisional": [pd.Timestamp(d).date().isoformat() for d in data.provisional],
        "errors": len(result.errors),
        "shifts": shift_records(result),
        "rist_requested": len(keys),
        "rist": frames,
    }


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return number(value) if isinstance(value, float) else value


def frame_payload(rs: RistShift) -> dict[str, Any] | None:
    """``rist_dict`` of a frame; a frame whose shift line is not finite is left out, a model line, pivot or ideal
    line that is not finite is dropped from it."""
    d = _finite(rist_dict(rs))
    if d["actual"] is None or any(v is None for v in d["actual"].values()):
        return None
    if d["model"] is not None and any(v is None for v in d["model"].values()):
        d["model"] = None
    for key in ("p_actual", "p_model"):
        if d[key] is not None and any(v is None for v in d[key]):
            d[key] = None
    if d["ideal"] is not None and any(v is None for v in d["ideal"].values()):
        d["ideal"] = None
    return d


def folder(directory: Path | str | None = None) -> Path:
    """``directory``, else the store folder (``STORE_DIR``, read at the call: tests point it elsewhere)."""
    return Path(directory) if directory is not None else STORE_DIR


def save(view: Mapping[str, Any], directory: Path | str | None = None) -> Path:
    """Write the view (atomic replace)."""
    path = folder(directory) / VIEW_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(view, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def refresh(
    *,
    job_id: str | None = None,
    nonce: int = 0,
    cfg: Mapping[str, Any] | None = None,
    directory: Path | str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the view from the memoised Coke Energy pipeline and store it (the ``twin.forge`` job)."""
    cfg = ce.config() if cfg is None else cfg
    settings = PipelineSettings.from_config(cfg)
    try:
        keys = ce.data_keys(cfg, nonce)
        data = ce.page_data(keys, settings)
    except Exception as exc:  # noqa: BLE001 - any source failure means: no pipeline result this time
        raise PlantDataUnavailable(f"{type(exc).__name__}: {exc}") from exc
    view = build_view(
        data,
        settings,
        ForgeSettings.from_config(cfg),
        built_at=now or datetime.now(UTC),
        job_id=job_id,
        frames_for=lambda ks: ce.rist_frames(keys, settings, ks),
    )
    save(view, directory)
    return view


def headline(view: Mapping[str, Any]) -> dict[str, Any]:
    """Job result: counts and the latest shift."""
    shifts = view.get("shifts") or []
    return {
        "built_at": view.get("built_at"),
        "shifts": len(shifts),
        "stable": sum(1 for s in shifts if s.get("stable")),
        "latest_shift": shifts[-1]["skey"] if shifts else None,
        "rist_frames": len(view.get("rist") or []),
        "rist_requested": view.get("rist_requested", 0),
        "pipeline_errors": view.get("errors", 0),
    }


# ---------------------------------------------------------------------------------------------- reading (API)
def shift_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The stored shifts as the pipeline's shift table (index = shift key; ``sdate`` and ``t_mid`` IST datetimes)."""
    T = pd.DataFrame(list(records), columns=list(COLUMNS)).set_index("skey")
    T["sdate"] = pd.to_datetime(T["sdate"])
    T["t_mid"] = pd.to_datetime(T["t_mid"])
    T["stable"] = T["stable"].fillna(False).astype(bool)
    T["provisional"] = T["provisional"].fillna(False).astype(bool)
    for c in NUMERIC:
        T[c] = pd.to_numeric(T[c], errors="coerce").astype(float)
    return T


@dataclass(frozen=True)
class StoredView:
    """A stored view: its header, the shift table and the Rist frames. Treat as read-only (it is memoised)."""

    meta: dict[str, Any]
    shifts: pd.DataFrame
    frames: list[dict[str, Any]] = field(default_factory=list)
    #: (path, mtime_ns, size) of the file it was read from: the key of the memoised answers
    token: tuple[str, int, int] | None = None

    @property
    def settings(self) -> dict[str, Any]:
        return self.meta.get("settings") or {}

    @property
    def built_at(self) -> datetime:
        return datetime.fromisoformat(self.meta["built_at"])

    def info(self) -> dict[str, Any]:
        """Header of every API answer."""
        s = self.settings
        return {
            "built_at": self.built_at,
            "job_id": self.meta.get("job_id"),
            "model": self.meta.get("model", MODEL),
            "sources": {k: _utc_or_none(v) for k, v in (self.meta.get("sources") or {}).items()},
            "shifts": int(len(self.shifts)),
            "pipeline_errors": int(self.meta.get("errors", 0)),
            "window_days": int(s.get("window_days", 14)),
            "min_shifts": int(s.get("min_shifts", 10)),
            "stats_days": int(s.get("stats_days", ForgeSettings.stats_days)),
            "rist_days": int(s.get("rist_days", ForgeSettings.rist_days)),
        }


def _utc_or_none(value: str | None) -> datetime | None:
    return None if value is None else to_utc(value)


def parse(view: Mapping[str, Any], token: tuple[str, int, int] | None = None) -> StoredView:
    if view.get("schema") != SCHEMA:
        raise ViewUnavailable(f"The stored FORGE view has schema {view.get('schema')!r}, not {SCHEMA}: rebuild it.")
    meta = {k: v for k, v in view.items() if k not in ("shifts", "rist")}
    shifts = shift_frame(view.get("shifts") or [])
    return StoredView(meta=meta, shifts=shifts, frames=list(view.get("rist") or []), token=token)


@memo(max_entries=2)
def _read(path: str, mtime_ns: int, size: int) -> StoredView:
    return parse(json.loads(Path(path).read_text(encoding="utf-8")), (path, mtime_ns, size))


def load(directory: Path | str | None = None) -> StoredView:
    """The stored view, re-read when the file changes."""
    path = folder(directory) / VIEW_FILE
    try:
        st = path.stat()
        return _read(str(path), st.st_mtime_ns, st.st_size)
    except FileNotFoundError as exc:
        raise ViewUnavailable("The FORGE view has not been built yet: the twin.forge job writes it hourly.") from exc
    except ValueError as exc:  # unreadable JSON
        raise ViewUnavailable(f"The stored FORGE view cannot be read ({exc}): rebuild it.") from exc


def stats(model: pd.Series, target: pd.Series) -> dict[str, Any]:
    """``accuracy.accuracy`` of ``model - target`` as plain values (shares in %)."""
    a = accuracy(model, target)
    w5, w10 = number(a["within_5"]), number(a["within_10"])
    return {
        "n": int(a["n"]),
        "bias": number(a["bias"]),
        "sd": number(a["sd"]),
        "mae": number(a["mae"]),
        "r": number(a["r"]),
        "within_5_pct": None if w5 is None else 100 * w5,
        "within_10_pct": None if w10 is None else 100 * w10,
    }


def _utc_times(t_mid: pd.Series) -> list[datetime]:
    return [t.to_pydatetime() for t in t_mid.dt.tz_localize(PLANT_TZ).dt.tz_convert("UTC")]


def _numbers(column: pd.Series) -> list[float | None]:
    return [number(v) for v in column.to_numpy()]


def shift_views(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """The shifts as the API returns them (column by column: a row loop costs 1 ms per shift)."""
    error = rows.K_energy - rows.K_burnt
    cols = {
        "k_forge": rows.K_energy,
        "k_burnt": rows.K_burnt,
        "k_obal": rows.K_obal,
        "k_charged": rows.K_charged,
        "error": error,
        "omega_fixed": rows.omega_fixed,
        "omega_shift": rows.omega_shift,
    }
    values = {k: _numbers(v) for k, v in cols.items()}
    fixed = zip(rows.index, rows.sdate.dt.date, _utc_times(rows.t_mid), rows.stable, rows.provisional, strict=True)
    return [
        {"shift_key": str(k), "day": day, "t_mid": t, "stable": bool(st), "provisional": bool(pr)}
        | {name: values[name][i] for name in cols}
        for i, (k, day, t, st, pr) in enumerate(fixed)
    ]


def rolling(scored: pd.DataFrame, days: Sequence[date], window_days: int, with_unstable: bool) -> list[dict[str, Any]]:
    """For each day E: the statistics of page 10 with the period E - ``window_days`` to E."""
    out = []
    for day in days:
        rows = period_rows(scored, day - timedelta(days=window_days), day, with_unstable)
        if rows.empty:
            continue
        out.append({"day": day, "t": to_utc(rows.t_mid.max()), **stats(rows.K_energy, rows.K_burnt)})
    return out


def resolve_period(stored: StoredView, start: date | None, end: date | None) -> tuple[date | None, date | None]:
    """The period with page 10's defaults: ``end`` the latest day with FORGE and burnt coke, ``start`` ``stats_days``
    before ``end`` (not before the first such day)."""
    scored = scored_shifts(stored.shifts)
    if scored.empty:
        return start, end
    days = int(stored.settings.get("stats_days", ForgeSettings.stats_days))
    first = scored.sdate.min().date()
    end = end or scored.sdate.max().date()
    return start or max(first, end - timedelta(days=days)), end


def prediction(
    stored: StoredView, start: date | None = None, end: date | None = None, with_unstable: bool = False
) -> dict[str, Any]:
    """FORGE and burnt coke per shift for the period (default: the ``stats_days`` before the latest scored day, as page
    10), the period's statistics, those of the whole record (stable shifts) and the trailing statistics per day."""
    days = int(stored.settings.get("stats_days", ForgeSettings.stats_days))
    scored = scored_shifts(stored.shifts)
    base = {"with_unstable": with_unstable, "stats_days": days}
    if scored.empty:
        empty = stats(pd.Series(dtype=float), pd.Series(dtype=float))
        return base | dict(start=start, end=end, first_day=None, last_day=None, stats=empty, record=empty, shifts=[],
                           rolling=[])  # fmt: skip
    first, last = scored.sdate.min().date(), scored.sdate.max().date()
    start, end = resolve_period(stored, start, end)
    rows = period_rows(scored, start, end, with_unstable)
    record = period_rows(scored, first, last, False)
    day_list = sorted(set(rows.sdate.dt.date))
    return base | {
        "start": start,
        "end": end,
        "first_day": first,
        "last_day": last,
        "stats": stats(rows.K_energy, rows.K_burnt),
        "record": stats(record.K_energy, record.K_burnt),
        "shifts": shift_views(rows),
        "rolling": rolling(scored, day_list, days, with_unstable),
    }


@memo(max_entries=16)
def _cached_prediction(
    token: tuple[str, int, int], start: date | None, end: date | None, with_unstable: bool
) -> dict[str, Any]:
    return prediction(_read(*token), start, end, with_unstable)


def cached_prediction(
    stored: StoredView, start: date | None = None, end: date | None = None, with_unstable: bool = False
) -> dict[str, Any]:
    """``prediction`` memoised per stored file and arguments: every page load asks for the default period, and the
    trailing statistics of a 30-day period take some 0.3 s. Treat the result as read-only."""
    if stored.token is None:
        return prediction(stored, start, end, with_unstable)
    return _cached_prediction(stored.token, start, end, with_unstable)


def rist(stored: StoredView, start: date | None = None, end: date | None = None) -> dict[str, Any]:
    """The stored Rist frames dated ``start`` to ``end`` (IST days, both included; default all), oldest first, each
    with the shift's time and coke rates."""
    T = stored.shifts
    frames = []
    for f in stored.frames:
        k = f["skey"]
        if k not in T.index:
            continue
        row = T.loc[k]
        day = row.sdate.date()
        if (start is not None and day < start) or (end is not None and day > end):
            continue
        frames.append(
            {
                **f,
                "day": day,
                "t_mid": to_utc(row.t_mid),
                "k_burnt": number(row.K_burnt),
                "k_forge": number(row.K_energy),
                "k_obal": number(row.K_obal),
                "k_charged": number(row.K_charged),
            }
        )
    days = [f["day"] for f in frames]
    return {
        "first_day": min(days) if days else None,
        "last_day": max(days) if days else None,
        "requested": int(stored.meta.get("rist_requested", len(stored.frames))),
        "frames": frames,
    }


def calibration_pool(T: pd.DataFrame) -> pd.DataFrame:
    """The stable shifts with their own calibration, as ``energy_balance_model`` pools them (``date`` = shift date)."""
    pool = T.loc[T.stable, ["sdate", "t_mid", "K_burnt", "K_charged", *CALIBRATION.values()]]
    return pool.rename(columns={v: k for k, v in CALIBRATION.items()}).assign(date=pool["sdate"])


def default_calibration_day(T: pd.DataFrame) -> date | None:
    """The day after the latest balanced shift (``live_calibration``'s default): the newest 14 days."""
    return None if T.empty else (T.sdate.max() + pd.Timedelta(days=1)).date()


def calibration(stored: StoredView, day: date | None = None) -> dict[str, Any]:
    """The FORGE calibration of ``day``: the medians of omega, Q_LZ, Q_UZ and r_H over the stable shifts dated in the
    ``window_days`` before it (at least ``min_shifts`` of them, else none), and those shifts."""
    s = stored.settings
    window_days, min_shifts = int(s.get("window_days", 14)), int(s.get("min_shifts", 10))
    T = stored.shifts
    day = day or default_calibration_day(T) or datetime.now(UTC).date()
    target = pd.Timestamp(day)
    lo = target - pd.Timedelta(days=window_days)
    pool = calibration_pool(T)
    med = rolling_pool_median(pool, pd.Series([target], index=["day"]), list(CALIBRATION), window_days, min_shifts)
    m = med.iloc[0]
    used = pool[(pool.date >= lo) & (pool.date < target)].sort_values("t_mid")
    return {
        "day": day,
        "window_start": lo.date(),
        "window_end": (target - pd.Timedelta(days=1)).date(),
        "window_days": window_days,
        "min_shifts": min_shifts,
        "n_shifts": int(len(used)),
        "n_calibrated": int(used["omega"].notna().sum()),
        "complete": bool(len(used) >= min_shifts),
        "omega_cal": number(m["omega_roll"]),
        "q_lz_gj_h": number(m["Q_LZ_GJh_roll"]),
        "q_uz_gj_h": number(m["Q_UZ_GJh_roll"]),
        "r_h": number(m["r_H_roll"]),
        "shifts": [
            {
                "shift_key": str(k),
                "t_mid": to_utc(r.t_mid),
                "omega": number(r["omega"]),
                "q_lz_gj_h": number(r["Q_LZ_GJh"]),
                "q_uz_gj_h": number(r["Q_UZ_GJh"]),
                "r_h": number(r["r_H"]),
                "k_burnt": number(r["K_burnt"]),
                "k_charged": number(r["K_charged"]),
            }
            for k, r in used.iterrows()
        ],
    }
