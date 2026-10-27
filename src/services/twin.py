"""Live twin evaluation for the API and the worker (SV-1.1, #281).

``evaluate(t)`` evaluates the window W = [t - 8 h, t) on data up to the full hour of ``t``: the Coke Energy sources of
``services.coke_energy`` (the same memoised loaders and pipeline run as TRACE, FORGE and page 10), the window balance,
the 14-day quantities of the completed shifts and STRATA 1.5 against the same 14 days (``utils.coke_energy.twin``).
``snapshot_values(ev, ...)`` gives the row of ``twin.snapshot`` for ``data.twin.SnapshotRepository.upsert``: window
times in UTC, coke rates in kg/tHM dry, omega dimensionless, other quantities with their units in the keys, data flags
and the notes of the evaluation, model versions; ``snapshot_event(values)`` the payload of the ``twin.snapshot``
event (``services.events.TwinSnapshotData``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from services import coke_energy as ce
from services import transit
from services.events import TwinSnapshotData
from utils.coke_energy.pipeline import PipelineSettings
from utils.coke_energy.twin import Window, WindowEvaluation, evaluate_window

PLANT_TZ = ce.PLANT_TZ
MODEL = "coke_energy pipeline windows (SV-1.1)"


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
    fd = loaders.hourly(keys.hourly_path, keys.hourly_mtime_ns)
    m10 = loaders.online(
        keys.history_start, keys.clock, keys.nonce, keys.fields_json, keys.cache_dir, keys.online_refresh_minutes
    )
    hm, dpr, provisional = loaders.neon(keys.history_start, keys.clock, keys.nonce)
    return TwinFrames(fd, m10, hm, dpr, tuple(provisional))


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
    return evaluate_window(
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
