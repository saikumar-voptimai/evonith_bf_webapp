"""Live twin evaluation for the API and the worker (SV-1.1, #281).

``evaluate(t)`` evaluates the window W = [t - 8 h, t) on data up to the full hour of ``t``: the Coke Energy sources of
``services.coke_energy`` (the same memoised loaders and pipeline run as TRACE, FORGE and page 10), the window balance,
the 14-day quantities of the completed shifts and STRATA 1.5 against the same 14 days (``utils.coke_energy.twin``).
``snapshot_values(ev, ...)`` gives the row of ``twin.snapshot`` for ``data.twin.SnapshotRepository.upsert``: window
times in UTC, coke rates in kg/tHM dry, omega dimensionless, other quantities with their units in the keys, data flags
and the notes of the evaluation, model versions; ``snapshot_event(values)`` the payload of the ``twin.snapshot``
event (``services.events.TwinSnapshotData``).

Hourly job (``twin.evaluate`` at HH:10, SV-1.7): the window ending HH:00 needs the hourly row HH-1, which the
published dataset first holds in its build of about HH:06 (``ensure_dataset`` fetches it when the local copy lacks
it); at HH:05 the window would have been evaluated on 7 of its 8 hours. ``due_windows`` adds the windows of the
backfill period that have no snapshot or were stored with their last row missing; ``evaluate_and_store`` evaluates
and upserts one window.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.coke_energy import sources
from services import coke_energy as ce
from services import source_health, transit
from services.events import TwinSnapshotData
from utils.coke_energy.pipeline import PipelineSettings
from utils.coke_energy.twin import Window, WindowEvaluation, evaluate_window

PLANT_TZ = ce.PLANT_TZ
MODEL = "coke_energy pipeline windows (SV-1.1)"
HOUR = pd.Timedelta(hours=1)
PENDING_FLAG = "last_row_pending"


@dataclass(frozen=True)
class TwinFrames:
    """The four source frames of one load and the DPR dates carried forward provisionally."""

    fd: pd.DataFrame
    m10: pd.DataFrame
    hm: pd.DataFrame
    dpr: pd.DataFrame
    provisional: tuple


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=PLANT_TZ).tz_localize(None)


def load_frames(keys: ce.DataKeys, loaders: ce.Loaders = ce.CACHED_LOADERS) -> TwinFrames:
    """The sources through ``loaders`` (memoised by default, shared with the Coke Energy service)."""
    health = source_health.HEALTH  # a loader that fails marks its source (SV-6.5) and raises SourceDown
    fd = health.observe("hourly", loaders.hourly, keys.hourly_path, keys.hourly_mtime_ns, data_end=_last_of("time"))
    m10 = health.observe(
        "influxdb",
        loaders.online,
        keys.history_start,
        keys.clock,
        keys.nonce,
        keys.fields_json,
        keys.cache_dir,
        keys.online_refresh_minutes,
        data_end=_last_of("ts"),
    )
    try:
        hm, dpr, provisional = loaders.neon(keys.history_start, keys.clock, keys.nonce)
    except Exception as exc:
        for key in ("hm_slag", "dpr"):  # one Neon fetch serves both
            health.record_failure(key, exc)
        raise source_health.SourceDown("hm_slag", health.reason("hm_slag") or str(exc)) from exc
    real_dpr = dpr[~dpr.index.isin(pd.to_datetime(list(provisional)))] if len(dpr) else dpr
    health.record_ok("hm_slag", data_end=_last(hm["t"]) if "t" in hm else None)
    health.record_ok("dpr", data_end=_last(real_dpr.index))
    return TwinFrames(fd, m10, hm, dpr, tuple(provisional))


def _last(values: Any) -> pd.Timestamp | None:
    series = pd.Series(values).dropna()
    return None if series.empty else pd.Timestamp(series.max())


def _last_of(column: str) -> Callable[[pd.DataFrame], pd.Timestamp | None]:
    return lambda frame: _last(frame[column]) if column in frame else None


def evaluate(
    t: Any = None,
    *,
    nonce: int = 0,
    cfg: Mapping[str, Any] | None = None,
    keys: ce.DataKeys | None = None,
    loaders: ce.Loaders = ce.CACHED_LOADERS,
    page_data_for: Callable[[ce.DataKeys, PipelineSettings], ce.PageData] | None = None,
    hours: int = 8,
    strata: bool = True,
) -> WindowEvaluation:
    """Evaluate the window of ``hours`` ending at the full hour of ``t`` (IST; default now).

    ``keys`` (default: the sources' current tokens, ``ce.data_keys``), ``loaders`` and ``page_data_for`` (default:
    the memoised ``ce.page_data``) are the Coke Energy service's; a caller holding its own passes them. ``strata=False``
    skips K_STRATA (network solves)."""
    cfg = ce.config() if cfg is None else cfg
    settings = PipelineSettings.from_config(cfg)
    keys = keys or ce.data_keys(cfg, nonce)
    frames = load_frames(keys, loaders)
    pool = (page_data_for or ce.page_data)(keys, settings).result
    window = Window.ending(now_ist() if t is None else t, hours)
    reference, _ = transit.settings(cfg)
    ev = evaluate_window(
        frames.fd,
        frames.m10,
        frames.hm,
        frames.dpr,
        window,
        pool,
        settings,
        frames.provisional,
        strata=strata,
        transit=reference,
    )
    ev.flags["hmt_forecast_gate"] = hmt_gate_status(cfg)
    ev.flags["sources"] = source_ends(frames)
    return ev


def source_ends(frames: TwinFrames) -> dict[str, str | None]:
    """Newest time each source holds, as UTC ISO strings (the Live Twin's data ages, SV-1.9): the 10-min InfluxDB
    data, the hourly dataset (label of its last hour), the last hot-metal cast, the last DPR date (provisional dates
    carried forward excluded) and the last day of the Open-Meteo record (00:00 IST)."""
    real_dpr = frames.dpr[~frames.dpr.index.isin(pd.to_datetime(list(frames.provisional)))]
    fresh = sources.freshness(frames.fd, frames.m10, frames.hm, real_dpr)
    ends = {
        "influxdb": fresh.online_end,
        "hourly": fresh.hourly_end,
        "hm_slag": fresh.hm_slag_end,
        "dpr": fresh.dpr_end,
        "open_meteo": open_meteo_end(),
    }
    return {k: None if v is None or pd.isna(v) else to_utc(ist_naive(v)).isoformat() for k, v in ends.items()}


def open_meteo_end() -> pd.Timestamp | None:
    """Last day of the Open-Meteo humidity record (shipped file, cache and earlier fetches; no fetch here)."""
    from utils.coke_energy import weather

    try:
        record = weather.open_meteo_rh(refresh=False).dropna()
    except Exception as exc:  # noqa: BLE001 - a missing record is a missing data age, not a failed evaluation
        source_health.HEALTH.record_failure("open_meteo", exc)
        return None
    end = None if record.empty else pd.Timestamp(record.index.max()).normalize()
    source_health.HEALTH.record_ok("open_meteo", data_end=end)
    return end


def hmt_gate_status(cfg: Mapping[str, Any]) -> str:
    """Gate status of the HMT forecast (``twin.hmt_forecast.gate.status``): the Live Twin shows the forecast only
    when it is ``passed`` (SV-1.4)."""
    gate = (((cfg or {}).get("twin") or {}).get("hmt_forecast") or {}).get("gate") or {}
    status = str(gate.get("status", "pending"))
    return status if status in ("pending", "passed", "failed") else "pending"


def snapshot_event(values: Mapping[str, Any]) -> TwinSnapshotData:
    """The ``twin.snapshot`` event payload of a snapshot row (``snapshot_values``), validated against its schema."""
    return TwinSnapshotData.model_validate(dict(values))


def to_utc(ts: pd.Timestamp) -> datetime:
    """An IST wall-clock time (tz-naive) as an aware UTC datetime, the twin store's convention."""
    return pd.Timestamp(ts).tz_localize(PLANT_TZ).tz_convert("UTC").to_pydatetime()


def snapshot_values(
    ev: WindowEvaluation,
    *,
    evaluated_at: datetime,
    job_id: str | None = None,
    versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``twin.snapshot`` row of an evaluation (STRATA columns empty when STRATA was not evaluated)."""
    return {
        "window_start": to_utc(ev.window.start),
        "window_end": to_utc(ev.window.end),
        "evaluated_at": evaluated_at,
        "job_id": job_id,
        "status": ev.status,
        "k_trace": ev.k_trace,
        "k_forge": ev.k_forge,
        "k_strata": ev.k_strata,
        "k_strata_sd": ev.k_strata_sd,
        "k_charged": ev.k_charged,
        "omega_imp": ev.omega_imp,
        "omega_strata": ev.omega_strata,
        "omega_strata_sd": ev.omega_strata_sd,
        "omega_cal": ev.omega_cal,
        "quantities": dict(ev.quantities),
        "flags": {**ev.flags, "notes": list(ev.notes)},
        "versions": {"model": MODEL, **dict(ev.versions), **dict(versions or {})},
    }


# --------------------------------------------------------------------------------------------- hourly job (SV-1.7)
def ist_naive(value: Any) -> pd.Timestamp:
    """An IST wall-clock time (tz-naive) from an aware time (any zone) or a naive IST one."""
    ts = pd.Timestamp(value)
    return ts.tz_convert(PLANT_TZ).tz_localize(None) if ts.tzinfo is not None else ts


def target_window_end(requested: Any = None, slot: Any = None, now: Any = None) -> pd.Timestamp:
    """End (IST, full hour) of the window a job evaluates: the requested one, else the hour of its schedule slot (the
    HH:10 slot evaluates the window ending HH:00), else the hour of now."""
    if requested is not None:
        return ist_naive(requested).floor("h")
    if slot is not None:
        return ist_naive(slot).floor("h")
    return ist_naive(now if now is not None else pd.Timestamp.now(tz=PLANT_TZ)).floor("h")


@dataclass
class DatasetState:
    """The local hourly dataset: its last row and the publisher's build time (both IST), and what a refresh gave."""

    data_end: pd.Timestamp | None
    built_at: pd.Timestamp | None
    refresh: dict[str, Any] | None = None

    def holds(self, row: pd.Timestamp) -> bool:
        return self.data_end is not None and self.data_end >= row

    def as_dict(self) -> dict[str, Any]:
        return dict(
            data_end=None if self.data_end is None else self.data_end.isoformat(),
            built_at=None if self.built_at is None else self.built_at.isoformat(),
            refresh=self.refresh,
        )


def dataset_state(cfg: Mapping[str, Any] | None = None) -> DatasetState:
    from utils.bmo.charged_coke.bmo import _built_at

    cfg = ce.config() if cfg is None else cfg
    status = sources.static_dataset_status(cfg.get("data_sources", {}) or {})
    end = status.get("data_end")
    built = _built_at(Path(status["path"]).parent / "cache_meta.json")
    return DatasetState(
        None if end is None or pd.isna(end) else ist_naive(end), None if built is None else ist_naive(built)
    )


def ensure_dataset(
    window_end: pd.Timestamp,
    *,
    cfg: Mapping[str, Any] | None = None,
    now: Any = None,
    refresh_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = sources.refresh_static_dataset,
    state_fn: Callable[[Mapping[str, Any] | None], DatasetState] = dataset_state,
) -> tuple[bool, DatasetState]:
    """Whether the local hourly dataset holds the window's last row (``window_end - 1 h``). Without it the published
    dataset is fetched first, unless the local copy is already this hour's build (the next build is about HH:06)."""
    cfg = ce.config() if cfg is None else cfg
    last_row = window_end - HOUR
    state = state_fn(cfg)
    if state.holds(last_row):
        return True, state
    hour = target_window_end(now=now)
    if state.built_at is None or state.built_at < hour:
        outcome = refresh_fn(cfg.get("data_sources", {}) or {})
        state = state_fn(cfg)
        state.refresh = {k: str(v) for k, v in dict(outcome or {}).items()}
        source_health.record_dataset_refresh(outcome, state.data_end)  # a failed refresh marks the source (SV-6.5)
    return state.holds(last_row), state


def due_windows(store: Any, window_end: pd.Timestamp, backfill_hours: int) -> list[pd.Timestamp]:
    """``window_end`` and, before it, the window ends of the last ``backfill_hours`` without a snapshot or stored
    with their last hourly row missing (IST, oldest first)."""
    ends = [window_end - HOUR * k for k in range(int(backfill_hours), 0, -1)]
    if not ends:
        return [window_end]
    rows = store.snapshots.series(to_utc(ends[0]), to_utc(window_end))
    stored = {ist_naive(_aware_utc(r.window_end)): r for r in rows}
    due = [e for e in ends if e not in stored or bool((stored[e].flags or {}).get(PENDING_FLAG))]
    return [*due, window_end]


def _aware_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


@dataclass
class StoredWindow:
    values: dict[str, Any]
    seconds: float
    notes: list[str] = field(default_factory=list)


def evaluate_and_store(
    store: Any,
    window_end: pd.Timestamp,
    *,
    job_id: str | None = None,
    last_row_pending: bool = False,
    strata: bool = True,
    evaluate_fn: Callable[..., WindowEvaluation] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> StoredWindow:
    """Evaluate the window ending ``window_end`` (IST) and upsert its snapshot. ``last_row_pending``: the dataset did
    not hold the window's last row by the last try; the snapshot is flagged and evaluated again by a later run."""
    t0 = time.monotonic()
    ev = (evaluate_fn or evaluate)(window_end, strata=strata)
    if last_row_pending:
        ev.flags[PENDING_FLAG] = True
        ev.notes.append(
            f"the hourly dataset did not hold the row of {window_end - HOUR:%H:%M} by the last try; evaluated on "
            f"{ev.flags.get('n_hours', 0)} hourly rows and evaluated again when the row is published"
        )
    seconds = round(time.monotonic() - t0, 2)
    values = snapshot_values(ev, evaluated_at=(clock or (lambda: datetime.now(UTC)))(), job_id=job_id)
    values["flags"]["evaluation_s"] = seconds
    if store is not None:
        store.snapshots.upsert(values)
    return StoredWindow(values, seconds, list(ev.notes))


def headline(values: Mapping[str, Any]) -> dict[str, Any]:
    """The job result's line of one window."""
    keys = ("status", "k_trace", "k_forge", "k_strata", "k_charged", "omega_imp", "omega_strata")
    out = {"window_end": ist_naive(values["window_end"]).isoformat()} | {k: values.get(k) for k in keys}
    out["seconds"] = (values.get("flags") or {}).get("evaluation_s")
    return out
