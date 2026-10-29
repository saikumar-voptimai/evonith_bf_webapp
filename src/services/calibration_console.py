"""Calibration console of the admin space (SV-4.6, #309): the day's two-zone calibration, its stable-shift mask, the
calibration history, the STRATA-Flow Jacobian cache and the runs of ``calibration.nightly``.

The view reads what ``calibration.nightly`` stored (``services.day_calibration``: one JSON file per IST day under
``src/storage/burden_engine/calibration``); it runs neither the pipeline nor a balance. The calibration values are the
ones ``data.burden_engine.calibration.live_calibration`` returned for the window, unrounded (omega_cal, Q_LZ and Q_UZ in
GJ/h, r_H, the window's charged coke in kg/tHM dry); the mask is ``utils.burden_engine.calibration.shift_mask`` of the
same call: every shift of the window's days with ``used`` (averaged into the calibration), ``stable`` (passed the
stability filter but not averaged), ``unstable`` (failed a test of ``whatif.stable_tests``; the tests it failed) or
``missing`` (no balanced shift: data gap, or the C shift still running at 02:30). Calibrations made before SV-4.6 carry
no mask (None).

Jacobian cache: ``src/storage/strata_flow/jacobian_<key>.json`` (``utils.burden_engine.strata_network``, one file per
calibration state, network version and RI / RDI catalogue). The day's file is the one the nightly job's network
statistics name (``cache_file``, or the ``cache <file>`` source of the network); a file of another network version is
never read again (``stale``). Days are IST calendar days, times UTC. No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from services import day_calibration, schedule
from services.jobs import CATALOG
from services.twin_views import utc
from utils.burden_engine import strata_network
from utils.coke_energy.whatif import STABLE_BOUNDS

KIND = "calibration.nightly"
SHIFTS = ("A", "B", "C")
#: Tests of the stability filter: balance column -> (name in the API, label, unit).
TESTS: dict[str, tuple[str, str, str]] = {
    "coke_kgthm_dry": ("coke_kg_thm", "Charged coke", "kg/tHM"),
    "C_closure_pct": ("c_closure_pct", "Carbon closure", "%"),
    "blast_meter_vs_reconciled_pct": ("blast_meter_pct", "Blast meter against reconciled blast", "%"),
    "R_out_cooling_total": ("cooling_mj_thm", "Cooling losses", "MJ/tHM"),
    "Fe_closure_pct": ("fe_closure_pct", "Iron closure", "%"),
    "prod_th": ("production_t_h", "Production", "t/h"),
}
FILES_LISTED = 50
_CACHE_SOURCE = re.compile(r"^cache (jacobian_[0-9a-f]+\.json)$")


class DayNotFound(LookupError):
    """No calibration stored for the requested day."""


# ------------------------------------------------------------------------------------------------------ values


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _day(text: Any) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10]) if text else None
    except ValueError:
        return None


def stability_tests() -> list[dict[str, Any]]:
    """The tests of the stability filter with their bands (``whatif.STABLE_BOUNDS``), in the filter's order."""
    return [
        dict(name=TESTS[col][0], label=TESTS[col][1], unit=TESTS[col][2], lo=float(lo), hi=float(hi))
        for col, (lo, hi) in STABLE_BOUNDS.items()
    ]


def calibration_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``burden_optimiser.yml → calibration`` (window_days, min_shifts), read at the call when not given."""
    if config is None:
        from services import optimiser

        config = optimiser.page_config(optimiser.config_mtime("burden_optimiser.yml")).get("calibration", {})
    return dict(config or {})


# ------------------------------------------------------------------------------------------------------ store


def stored_days(directory: Path | str | None = None) -> list[date]:
    """IST days with a stored calibration, newest first."""
    days = (_day(p.stem) for p in day_calibration.folder(directory).glob("????-??-??.json"))
    return sorted((d for d in days if d is not None), reverse=True)


def cache_file_of(network: Mapping[str, Any] | None) -> str | None:
    """Jacobian cache file the nightly job's network statistics name (None when they name none)."""
    network = network or {}
    name = network.get("cache_file")
    if isinstance(name, str) and name:
        return name
    found = _CACHE_SOURCE.match(str(network.get("jacobian") or ""))
    return found.group(1) if found else None


def _cal_values(cal: Mapping[str, Any]) -> dict[str, float | None]:
    return dict(
        omega_cal=_num(cal.get("omega")),
        q_lz_gj_h=_num(cal.get("Q_LZ_GJh")),
        q_uz_gj_h=_num(cal.get("Q_UZ_GJh")),
        r_h=_num(cal.get("r_H")),
        coke_kg_thm=_num(cal.get("K")),
        coke_model_kg_thm=_num(cal.get("coke_model_cal")),
    )


def mask_grid(window_start: date | None, window_end: date | None, mask: list[Mapping[str, Any]] | None) -> list[dict]:
    """Every shift of the window's days, oldest first, with its state and its value per stability test."""
    if mask is None or window_start is None or window_end is None or window_end < window_start:
        return []
    by_key = {str(m.get("shift_key")): m for m in mask}
    rows: list[dict[str, Any]] = []
    day = window_start
    while day <= window_end:
        for letter in SHIFTS:
            key = f"{day.isoformat()}_{letter}"
            m = by_key.get(key)
            values = dict((m or {}).get("values") or {})
            if m is None:
                state = "missing"
            elif m.get("used"):
                state = "used"
            elif m.get("stable"):
                state = "stable"
            else:
                state = "unstable"
            rows.append(
                dict(
                    shift_key=key,
                    day=day,
                    shift=letter,
                    state=state,
                    failed=[TESTS[c][0] for c in (m or {}).get("failed") or [] if c in TESTS],
                    **{TESTS[c][0]: _num(values.get(c)) for c in TESTS},
                )
            )
        day += timedelta(days=1)
    return rows


def detail(doc: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The stored calibration of one day for the console."""
    info = doc.get("info") or {}
    settings = {**calibration_settings(config), **dict(doc.get("settings") or {})}
    window = list(info.get("window") or ["", ""]) + ["", ""]
    start, end = _day(window[0]), _day(window[1])
    window_days = settings.get("window_days")
    if window_days is None and start is not None and end is not None:
        window_days = (end - start).days + 1
    mask = info.get("mask") if "mask" in info else None
    network = dict(doc.get("network") or {})
    n_shifts = int(info.get("n_shifts") or 0)
    min_shifts = int(settings.get("min_shifts", 10))
    return dict(
        day=_day(doc.get("day")),
        made_at=utc(doc.get("made_at")),
        job_id=doc.get("job_id"),
        source=info.get("source", "live"),
        message=info.get("message", ""),
        window_start=start,
        window_end=end,
        window_days=int(window_days) if window_days is not None else None,
        min_shifts=min_shifts,
        n_shifts=n_shifts,
        complete=info.get("source", "live") == "live" and n_shifts >= min_shifts,
        **_cal_values(info.get("cal") or {}),
        shifts=[str(k) for k in info.get("shifts") or []],
        mask=mask_grid(start, end, mask) if mask is not None else None,
        network=dict(
            network=network.get("network"),
            solves=network.get("solves"),
            seconds=_num(network.get("seconds")),
            seconds_total=_num(network.get("seconds_total")),
            anchors=network.get("anchors"),
            jacobian=network.get("jacobian"),
            cache_file=cache_file_of(network),
        )
        if network
        else None,
    )


def history_row(doc: Mapping[str, Any], present: set[str]) -> dict[str, Any]:
    info = doc.get("info") or {}
    cache_file = cache_file_of(doc.get("network"))
    return dict(
        day=_day(doc.get("day")),
        made_at=utc(doc.get("made_at")),
        job_id=doc.get("job_id"),
        source=info.get("source", "live"),
        n_shifts=int(info.get("n_shifts") or 0),
        **_cal_values(info.get("cal") or {}),
        jacobian_file=cache_file,
        jacobian_cached=(cache_file in present) if cache_file else None,
    )


# ------------------------------------------------------------------------------------------------- jacobians


def cache_files(cache_dir: Path | str | None = None) -> list[tuple[Path, os.stat_result]]:
    """The Jacobian files of the cache folder with their stat, newest first."""
    folder = Path(cache_dir) if cache_dir is not None else strata_network.CACHE_DIR
    found = []
    for path in folder.glob("jacobian_*.json") if folder.is_dir() else []:
        try:
            found.append((path, path.stat()))
        except OSError:
            continue
    return sorted(found, key=lambda item: item[1].st_mtime, reverse=True)


_NETWORK_OF: dict[tuple[str, int, int], str | None] = {}


def network_of(path: Path, st: os.stat_result) -> str | None:
    """Network version a Jacobian file was built for (kept per file, size and modification time)."""
    key = (str(path), int(st.st_mtime_ns), int(st.st_size))
    if key not in _NETWORK_OF:
        try:
            net = json.loads(path.read_text(encoding="utf-8")).get("net")
        except (OSError, ValueError, AttributeError):
            net = None
        if len(_NETWORK_OF) > 4096:
            _NETWORK_OF.clear()
        _NETWORK_OF[key] = net if isinstance(net, str) else None
    return _NETWORK_OF[key]


def jacobian_cache(
    day_file: str | None,
    files: list[tuple[Path, os.stat_result]],
    *,
    shown: bool,
    days_by_file: Mapping[str, list[date]] | None = None,
    listed: int = FILES_LISTED,
) -> dict[str, Any]:
    """The Jacobian cache folder: its files newest first (the first ``listed``) with the network version each was
    built for, and whether the shown calibration's file is there (``cached`` / ``missing``; ``unknown`` when its run
    named none; ``none`` without a calibration)."""
    versions = [network_of(path, st) for path, st in files]
    listing = [
        dict(
            name=path.name,
            size_bytes=int(st.st_size),
            modified_at=datetime.fromtimestamp(st.st_mtime, UTC),
            network=net,
            current=net == strata_network.NET_VERSION,
            days=sorted((days_by_file or {}).get(path.name, []), reverse=True),
        )
        for (path, st), net in list(zip(files, versions, strict=True))[:listed]
    ]
    names = {path.name for path, _ in files}
    if not shown:
        status = "none"
    elif day_file is None:
        status = "unknown"
    else:
        status = "cached" if day_file in names else "missing"
    return dict(
        directory="src/storage/strata_flow",
        network_version=strata_network.NET_VERSION,
        status=status,
        day_file=day_file,
        n_files=len(files),
        total_bytes=int(sum(st.st_size for _, st in files)),
        stale_files=sum(net != strata_network.NET_VERSION for net in versions),
        files=listing,
    )


# ------------------------------------------------------------------------------------------------------ jobs


def nightly_job(now: datetime | None = None) -> dict[str, Any]:
    kind = CATALOG[KIND]
    now = now or datetime.now(UTC)
    return dict(
        kind=kind.name,
        title=kind.title,
        role=kind.role,
        schedule=schedule.describe(kind.schedule),
        next_slot=schedule.next_slot(kind.schedule, now) if kind.schedule else None,
        timeout_s=float(kind.timeout_s),
        max_tries=int(kind.max_tries),
    )


def run_rows(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Runs of ``calibration.nightly`` (Redis ``JobRecord`` or ``twin.jobs`` rows), as given (newest first)."""
    rows = []
    for r in records:
        result = getattr(r, "result", None) or {}
        rows.append(
            dict(
                id=r.id,
                status=r.status,
                submitted_by=r.submitted_by,
                scheduled=schedule.is_scheduler(r),
                submitted_at=utc(r.submitted_at),
                started_at=utc(r.started_at),
                finished_at=utc(r.finished_at),
                run_s=_num(r.run_s),
                tries=int(r.tries or 0),
                error=r.error,
                day=_day(result.get("day")),
            )
        )
    return rows


# ------------------------------------------------------------------------------------------------------ view


def view(
    day: date | None = None,
    *,
    directory: Path | str | None = None,
    cache_dir: Path | str | None = None,
    config: Mapping[str, Any] | None = None,
    today: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The console: the calibration of ``day`` (default: the newest stored), the stored days, the stability tests, the
    Jacobian cache and the nightly job. ``DayNotFound`` when ``day`` has no stored calibration."""
    today = today or day_calibration.today()
    config = calibration_settings(config)
    docs = [doc for d in stored_days(directory) if (doc := day_calibration.read(d, directory)) is not None]
    shown_day = day if day is not None else (_day(docs[0].get("day")) if docs else None)
    doc = next((d for d in docs if _day(d.get("day")) == shown_day), None) if shown_day else None
    if day is not None and doc is None:
        raise DayNotFound(f"calibration stored for {day.isoformat()} (the last {day_calibration.KEEP} days are kept)")
    days_by_file: dict[str, list[date]] = {}
    for d in docs:
        name, when = cache_file_of(d.get("network")), _day(d.get("day"))
        if name and when:
            days_by_file.setdefault(name, []).append(when)
    shown = detail(doc, config) if doc is not None else None
    files = cache_files(cache_dir)
    cache = jacobian_cache(
        shown["network"]["cache_file"] if shown and shown["network"] else None,
        files,
        shown=shown is not None,
        days_by_file=days_by_file,
    )
    present = {path.name for path, _ in files}
    return dict(
        today=today,
        day=shown_day if doc is not None else None,
        current=doc is not None and shown_day == today,
        calibration=shown,
        days=[history_row(d, present) for d in docs],
        tests=stability_tests(),
        jacobian_cache=cache,
        job=nightly_job(now),
    )


__all__ = [
    "KIND",
    "TESTS",
    "DayNotFound",
    "cache_file_of",
    "cache_files",
    "calibration_settings",
    "detail",
    "history_row",
    "jacobian_cache",
    "mask_grid",
    "network_of",
    "nightly_job",
    "run_rows",
    "stability_tests",
    "stored_days",
    "view",
]
