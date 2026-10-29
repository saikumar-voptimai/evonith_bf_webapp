"""Hot-metal Si and charged-coke forecasts issued by the worker and read by page 9 (SV-1.5, #285).

Until SV-1.5 both forecasts were issued only while page 9 ran (``service.issue`` and ``issue_latest`` in
``9_Blend_Optimizer.py``). The worker now issues them on a schedule, with the same functions and into the same
ledgers, whose format is unchanged:

    forecast.si             every 5 min (IST): the Si path (Now, +1, +2, +3 h) at the model's origin; matured
                            outcomes are matched to the raw samples on every call, as before
    forecast.charged_coke   HH:07: the 4-h charged-coke block for the latest complete row of the hourly dataset,
                            which is fetched again first when the local copy predates this hour's build (published at
                            about HH:06); outcomes are recorded when a new row is issued, as before

    src/storage/bmo_si_forecast/issued_forecasts.jsonl     Si paths and their matched samples
    src/storage/bmo_charged_coke/issued_forecasts.jsonl    charged-coke blocks and their realised values

Who issues (``setting_bmo.yml → bmo.forecast_issuer``):

    auto    page 9 reads the ledgers while the worker issues (its last attempt younger than ``STALE_AFTER``) and
            issues itself, as before, when the worker is silent (default)
    worker  page 9 only reads; without the worker the forecasts become unavailable once they are out of date
    page    page 9 issues as before

Every attempt, by the worker or by the page, is appended to ``attempts/YYYY-MM.jsonl`` in the forecast's storage
folder (time, issuer, origin or data row, status, whether a ledger record was written, seconds, error), and the latest
one is kept with its full record in ``latest_attempt.json``. A Si attempt whose inputs are not usable (no recent
sample, too many missing inputs) writes no ledger record; page 9 shows that attempt's status from this file.
``gaps`` lists the intervals without a ledger record and the attempts made within them (acceptance check of #285: no
gap above 10 min in the Si ledger over 7 days).

Outages (SV-6.5): the Si source's InfluxDB and HM/slag reads and the charged-coke refresh of the hourly dataset record
their source in ``services.source_health``; a worker run that skips because a source is down records a ``skipped``
attempt with the reason (``record_skip``), so the gap check names it and page 9 keeps reading.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from config.config_loader import load_config
from data.bmo import si_forecast_context
from data.bmo.si_forecast_context import LiveSiliconForecastSource
from data.coke_energy import sources as dataset_sources
from services import source_health
from services.events import ChargedCokeForecastData, ChargedCokePoint, SiForecastData, SiForecastPoint
from utils.bmo.charged_coke.bmo import _built_at
from utils.bmo.charged_coke.contract import _naive_ist
from utils.bmo.charged_coke.deployments import active_bundle as active_charged_bundle
from utils.bmo.charged_coke.ledger import ForecastLedger, PlannedChangeRegister
from utils.bmo.charged_coke.service import ChargedCokeEngine, issue_latest
from utils.bmo.si_forecast.deployments import active_bundle as active_si_bundle
from utils.bmo.si_forecast.ledger import SiliconForecastLedger
from utils.bmo.si_forecast.service import (
    LEGACY_MODE,
    MODES,
    SCHEMA_VERSION,
    SiliconForecastService,
    forecast_horizons,
    forecast_origin,
)
from utils.jsonl_log import JsonlLog
from utils.memo import memo

log = logging.getLogger(__name__)

PLANT_TZ = "Asia/Kolkata"
ROOT = Path(__file__).resolve().parents[2]
BMO_CONFIG = ROOT / "src" / "config" / "setting_bmo.yml"

Forecast = Literal["si", "charged_coke"]
Issuer = Literal["auto", "worker", "page"]
ISSUERS: tuple[str, ...] = ("auto", "worker", "page")
#: A worker whose last attempt is older than this is taken to be stopped (three Si runs; one hourly run plus the
#: retries of a late build).
STALE_AFTER: dict[str, pd.Timedelta] = {"si": pd.Timedelta(minutes=15), "charged_coke": pd.Timedelta(minutes=75)}
#: The hourly dataset is built at about HH:06; until HH:07 the previous build is the current one.
BUILD_MINUTE = 7


# --------------------------------------------------------------------------------------------- configuration
@memo(max_entries=2)
def _read_bmo_config(mtime_ns: int) -> dict[str, Any]:
    return dict(load_config("setting_bmo.yml").get("bmo", {}) or {})


def bmo_config() -> dict[str, Any]:
    """The ``bmo`` block of ``config/setting_bmo.yml``, re-read when the file changes. Treat as read-only."""
    return _read_bmo_config(BMO_CONFIG.stat().st_mtime_ns if BMO_CONFIG.exists() else 0)


def issuer(cfg: Mapping[str, Any] | None = None) -> str:
    """``bmo.forecast_issuer``: auto (default), worker or page."""
    value = str((bmo_config() if cfg is None else cfg).get("forecast_issuer", "auto") or "auto").strip().lower()
    return value if value in ISSUERS else "auto"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class SiPaths:
    mode: str  # legacy | shadow | forecast
    default_bundle: Path
    storage_dir: Path
    static_dataset: Path


def si_paths(cfg: Mapping[str, Any] | None = None) -> SiPaths:
    """Mode and paths of the production Si forecast (``bmo.si_forecast``); never the legacy blend-Si paths."""
    bmo = bmo_config() if cfg is None else cfg
    si = dict(bmo.get("si_forecast", {}) or {})
    mode = str(si.get("mode", "forecast")).strip().lower()
    if mode not in MODES:
        mode = "forecast"
    data_sources = bmo.get("data_sources", {}) or {}
    return SiPaths(
        mode=mode,
        default_bundle=repo_path(si.get("default_bundle_dir", "src/assets/models/bmo_si_forecast/20260924_initial")),
        storage_dir=repo_path(si.get("storage_dir", "src/storage/bmo_si_forecast")),
        static_dataset=repo_path(
            si.get(
                "static_dataset_path",
                data_sources.get("static_dataset_path", "src/assets/data/furnace_dataset.csv"),
            )
        ),
    )


@dataclass(frozen=True)
class ChargedPaths:
    dataset_path: Path
    storage_dir: Path
    default_bundle: Path
    dataset_url: str  # the published hourly dataset; empty: rebuilt from the database


def charged_paths(cfg: Mapping[str, Any] | None = None) -> ChargedPaths:
    """Dataset, ledger folder and default bundle of the charged-coke forecast (``bmo.data_driven_coke``)."""
    bmo = bmo_config() if cfg is None else cfg
    ddc = dict(bmo.get("data_driven_coke", {}) or {})
    charged = dict(ddc.get("charged_4h", {}) or {})
    return ChargedPaths(
        dataset_path=repo_path(ddc.get("dataset_path", "src/assets/data/furnace_dataset.csv")),
        storage_dir=repo_path(charged.get("storage_dir", "src/storage/bmo_charged_coke")),
        default_bundle=repo_path(charged.get("bundle_dir", "src/assets/models/bmo_charged_coke/20261007_0400")),
        dataset_url=str((bmo.get("data_sources", {}) or {}).get("static_dataset_url", "") or "").strip(),
    )


def si_cadence(bundle: Path) -> int:
    """Issue cadence of a Si bundle, minutes: 60 for a single-horizon bundle (``forest.npz``), else 5."""
    return 60 if (Path(bundle) / "forest.npz").is_file() else 5


# --------------------------------------------------------------------------------------------- attempts
def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (stamp.tz_localize(PLANT_TZ) if stamp.tzinfo is None else stamp).tz_convert("UTC")


class AttemptLog:
    """Attempts to issue one forecast: ``attempts/YYYY-MM.jsonl`` (IST month of the attempt) and ``latest_attempt.json``
    (the latest attempt with its full record)."""

    def __init__(self, storage_dir: str | Path) -> None:
        self.dir = Path(storage_dir)

    @property
    def latest_path(self) -> Path:
        return self.dir / "latest_attempt.json"

    def _month(self, stamp: pd.Timestamp) -> JsonlLog:
        return JsonlLog(self.dir / "attempts" / f"{stamp.tz_convert(PLANT_TZ):%Y-%m}.jsonl")

    def record(self, attempt: Mapping[str, Any], record: Mapping[str, Any] | None) -> None:
        clean = _jsonable(dict(attempt))
        self._month(_utc(clean["at"])).append(clean)
        latest = json.dumps({**clean, "record": _jsonable(record) if record is not None else None}, ensure_ascii=False)
        self.dir.mkdir(parents=True, exist_ok=True)
        temporary = self.latest_path.with_name(f".{self.latest_path.name}.{os.getpid()}.tmp")
        temporary.write_text(latest + "\n", encoding="utf-8")
        os.replace(temporary, self.latest_path)

    def latest(self) -> dict[str, Any] | None:
        """The latest attempt with its ``record``, or None."""
        try:
            return json.loads(self.latest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def between(self, start: Any, end: Any) -> list[dict[str, Any]]:
        """Attempts with ``start <= at < end``, oldest first."""
        start, end = _utc(start), _utc(end)
        months = pd.period_range(
            start.tz_convert(PLANT_TZ).tz_localize(None), end.tz_convert(PLANT_TZ).tz_localize(None), freq="M"
        )
        out = []
        for month in months:
            for row in JsonlLog(self.dir / "attempts" / f"{month}.jsonl").records():
                at = _utc(row["at"])
                if start <= at < end:
                    out.append(row)
        return sorted(out, key=lambda r: _utc(r["at"]))

    def last(self, *, issuer: str | None = None, now: Any = None, within: pd.Timedelta | None = None) -> dict | None:
        """The newest attempt (by ``issuer`` when given) of the current and the previous month."""
        now = _utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
        for month in (now, now.tz_convert(PLANT_TZ).replace(day=1) - pd.Timedelta(days=1)):
            for row in reversed(self._month(_utc(month)).records()):
                if issuer is None or row.get("issuer") == issuer:
                    if within is not None and now - _utc(row["at"]) > within:
                        return None
                    return row
        return None


def worker_active(storage_dir: str | Path, forecast: Forecast, now: Any = None) -> bool:
    """Whether the worker attempted this forecast within ``STALE_AFTER``."""
    return AttemptLog(storage_dir).last(issuer="worker", now=now, within=STALE_AFTER[forecast]) is not None


def page_issues(
    storage_dir: str | Path, forecast: Forecast, cfg: Mapping[str, Any] | None = None, now: Any = None
) -> bool:
    """Whether page 9 issues the forecast itself (issuer ``page``, or ``auto`` while the worker is silent)."""
    who = issuer(cfg)
    if who == "page":
        return True
    if who == "worker":
        return False
    return not worker_active(storage_dir, forecast, now)


# --------------------------------------------------------------------------------------------- Si
@memo(max_entries=4)
def si_service(bundle: str, storage: str, manifest_mtime_ns: int) -> SiliconForecastService:
    """One service per bundle version and ledger folder (``manifest_mtime_ns`` is the cache token only)."""
    return SiliconForecastService(bundle_dir=bundle, storage_dir=storage, furnace_id="BF2")


def _manifest_mtime(bundle: Path) -> int:
    try:
        return int((Path(bundle) / "manifest.json").stat().st_mtime_ns)
    except OSError:
        return 0


def si_failure_record(origin: pd.Timestamp, exc: BaseException) -> dict[str, Any]:
    """The record shown when the Si sources could not be read (as page 9 built it)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "furnace_id": "BF2",
        "status": "insufficient_data",
        "reason_codes": ["source_fetch_failed"],
        "reasons": [str(exc)],
        "si_pct": None,
        "origin_at": origin.isoformat(),
        "issued_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "target_start": origin.isoformat(),
        "target_end": (origin + pd.Timedelta(minutes=5)).isoformat(),
        "interval": None,
        "horizons": [],
    }


HM_SLAG_TABLE = "offline_feed.hot_metal_slag_analysis"


def _observed_online(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """The Si source's InfluxDB read, recorded as ``influxdb`` (a failure raises ``SourceDown``)."""
    return source_health.HEALTH.observe("influxdb", si_forecast_context.fetch_online_df, *args, **kwargs)


def _observed_offline(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """The Si source's Neon reads; the HM/slag table is recorded as ``hm_slag``."""
    if kwargs.get("table_name") == HM_SLAG_TABLE:
        return source_health.HEALTH.observe("hm_slag", si_forecast_context.fetch_offline_data, *args, **kwargs)
    return si_forecast_context.fetch_offline_data(*args, **kwargs)


def live_si_source(**kwargs: Any) -> LiveSiliconForecastSource:
    """``LiveSiliconForecastSource`` whose InfluxDB and HM/slag reads record their source's health (SV-6.5)."""
    return LiveSiliconForecastSource(online_fetch=_observed_online, offline_fetch=_observed_offline, **kwargs)


def issue_si(
    *,
    now: Any = None,
    origin: Any = None,
    cfg: Mapping[str, Any] | None = None,
    issued_by: Literal["worker", "page"] = "worker",
    source_factory: Callable[..., Any] = live_si_source,
) -> dict[str, Any]:
    """Issue (once per origin) the Si path and match matured outcomes, as page 9 did.

    Args:
        now: Issue time (default: the clock, also recorded as ``issued_at``); tests fix it.
        origin: The origin to issue (default: ``now`` floored to the active bundle's cadence in IST).
        cfg: The ``bmo`` settings (default: ``config/setting_bmo.yml``).
        issued_by: Recorded in the attempt log.
        source_factory: ``live_si_source`` (``LiveSiliconForecastSource`` recording the source health) or a test
            double with its constructor and ``fetch``.

    Returns:
        ``{"attempt": ..., "record": ...}``; ``attempt["written"]`` says whether a ledger record was written. A
        failure to read the sources is recorded as an attempt and raised.
    """
    t0 = time.monotonic()
    at = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    paths = si_paths(cfg)
    attempt: dict[str, Any] = {"forecast": "si", "at": at, "issuer": issued_by, "mode": paths.mode}
    if paths.mode == LEGACY_MODE:
        return {"attempt": {**attempt, "status": "skipped", "written": False, "reason": "legacy mode"}, "record": None}
    bundle = active_si_bundle(paths.storage_dir, paths.default_bundle)
    service = si_service(str(bundle), str(paths.storage_dir), _manifest_mtime(bundle))
    origin = forecast_origin(origin if origin is not None else at, cadence_minutes=si_cadence(bundle))
    attempt.update(key=origin, bundle=bundle.name)
    attempts = AttemptLog(paths.storage_dir)
    known = origin in service.ledger.issues()
    try:
        feeds = source_factory(bundle_dir=bundle, static_dataset_path=paths.static_dataset).fetch(origin)
        record = service.issue(feeds, origin=origin, mode=paths.mode, issued_at=at if now is not None else None)
    except Exception as exc:
        attempt.update(
            status="failed",
            written=False,
            seconds=round(time.monotonic() - t0, 3),
            error=f"{type(exc).__name__}: {exc}"[:400],
        )
        attempts.record(attempt, si_failure_record(origin, exc))
        raise
    written = not known and origin in service.ledger.issues()
    attempt.update(
        status=record.get("status"),
        written=written,
        existing=known,
        model_version=record.get("model_version"),
        seconds=round(time.monotonic() - t0, 3),
    )
    if issued_by == "worker" or written:  # page reruns that find the record are not logged
        attempts.record(attempt, record)
    return {"attempt": attempt, "record": record}


def record_skip(
    forecast: Forecast, reason: str, *, now: Any = None, cfg: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Record a worker run that skipped because a source is down (SV-6.5) as a ``skipped`` attempt with the reason
    (a Si attempt carries the failure record page 9 shows for the origin); None in the legacy Si mode, which issues
    nothing."""
    at = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    attempt: dict[str, Any] = {"forecast": forecast, "at": at, "issuer": "worker"}
    record = None
    if forecast == "si":
        paths = si_paths(cfg)
        if paths.mode == LEGACY_MODE:
            return None
        bundle = active_si_bundle(paths.storage_dir, paths.default_bundle)
        origin = forecast_origin(at, cadence_minutes=si_cadence(bundle))
        attempt.update(mode=paths.mode, key=origin, bundle=bundle.name)
        record = si_failure_record(origin, source_health.SourceDown("", reason))
        storage = paths.storage_dir
    else:
        storage = charged_paths(cfg).storage_dir
    attempt.update(status="skipped", written=False, seconds=0.0, error=reason[:400])
    AttemptLog(storage).record(attempt, record)
    return attempt


@dataclass(frozen=True)
class PageForecast:
    """What page 9 shows when it reads instead of issuing."""

    record: dict[str, Any] | None
    note: str | None  # shown above the panel when the record is not the current one
    current: bool  # the record is for the current origin or data row (or the one before, while the worker runs)


def _ist_text(value: Any) -> str:
    return f"{_utc(value).tz_convert(PLANT_TZ):%d %b %H:%M} IST"


def si_for_page(storage_dir: str | Path, bundle: Path, *, now: Any = None) -> PageForecast:
    """The Si record page 9 shows when it reads: the ledger's at the current origin, else that origin's attempt (an
    unusable status), else the previous origin's while the worker issues the current one. Anything older is out of
    date: it is not passed on (no usable horizon), and the note says since when nothing was issued."""
    at = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    cadence = pd.Timedelta(minutes=si_cadence(bundle))
    origin = forecast_origin(at, cadence_minutes=int(cadence.total_seconds() // 60))
    ledger_issues = SiliconForecastLedger(storage_dir).issues()
    if origin in ledger_issues:
        return PageForecast(copy.deepcopy(ledger_issues[origin]), None, True)
    latest = AttemptLog(storage_dir).latest()
    if latest and latest.get("record") and latest.get("key") and _utc(latest["key"]) == origin:
        return PageForecast(latest["record"], None, True)
    previous = origin - cadence
    candidates: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    older = [t for t in ledger_issues if t < origin]
    if older:
        candidates.append((older[-1], ledger_issues[older[-1]]))
    if latest and latest.get("record") and latest.get("key"):
        candidates.append((_utc(latest["key"]), latest["record"]))
    if not candidates:
        return PageForecast(None, "No HM Si forecast has been issued yet.", False)
    when, record = max(candidates, key=lambda c: c[0])
    if when >= previous:
        return PageForecast(
            copy.deepcopy(record),
            f"Forecast of {_ist_text(when)}; the {_ist_text(origin)} forecast is being issued.",
            True,
        )
    note = (
        f"No HM Si forecast has been issued since {_ist_text(when)}; one is due every "
        f"{int(cadence.total_seconds() // 60)} min."
    )
    stale = {
        **copy.deepcopy(record),
        "status": "insufficient_data",
        "si_pct": None,
        "interval": None,
        "horizons": [{**h, "si_pct": None, "lower_pct": None, "upper_pct": None} for h in record.get("horizons") or []],
        "reasons": [note],
        "reason_codes": ["forecast_out_of_date"],
    }
    return PageForecast(stale, note, False)


# --------------------------------------------------------------------------------------------- charged coke
@memo(max_entries=2)
def charged_engine(bundle: str) -> ChargedCokeEngine:
    """The model, policy and physics of one bundle, loaded once."""
    return ChargedCokeEngine.load(bundle)


def read_charged_source(path: str | Path) -> pd.DataFrame:
    """The published hourly dataset as page 9 reads it."""
    return pd.read_csv(path, parse_dates=["time"]).set_index("time")


def _meta(dataset_path: Path) -> dict[str, Any]:
    try:
        return json.loads((dataset_path.parent / "cache_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _build_time(dataset_path: Path) -> pd.Timestamp | None:
    built = _built_at(dataset_path.parent / "cache_meta.json")
    return _naive_ist(built) if built is not None else None


def expected_row(now: Any) -> pd.Timestamp:
    """The row the latest build holds complete at ``now`` (IST; naive values are IST): H-1 from HH:07, H-2 before."""
    return (_naive_ist(pd.Timestamp(now)) - pd.Timedelta(minutes=BUILD_MINUTE)).floor("h") - pd.Timedelta(hours=1)


def issue_charged(
    *,
    now: Any = None,
    cfg: Mapping[str, Any] | None = None,
    issued_by: Literal["worker", "page"] = "worker",
    refresh: bool = True,
    refresh_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = dataset_sources.refresh_static_dataset,
    source: pd.DataFrame | None = None,
    engine: ChargedCokeEngine | None = None,
) -> dict[str, Any]:
    """Issue (once per data row) the charged-coke forecast for the latest complete row, as page 9 did.

    The local hourly dataset is fetched again first when its build predates the current hour (``refresh``). The
    rows are read from the build's own versioned file (``cache_meta.json → csv_file``), so that rows and build time
    belong to the same build.

    Args:
        now: Issue time (default: the clock); tests fix it.
        cfg: The ``bmo`` settings (default: ``config/setting_bmo.yml``).
        issued_by: Recorded in the attempt log.
        refresh: Fetch the dataset when the local build predates the current hour.
        refresh_fn: The fetch (``data.coke_energy.sources.refresh_static_dataset``).
        source, engine: The dataset and the engine when the caller already holds them (page 9); the build time is
            then the local file's.

    Returns:
        ``{"attempt": ..., "record": ...}``; ``attempt["behind"]`` is true while the build of the current hour is not
        published (the record is then the previous hour's).
    """
    t0 = time.monotonic()
    now_ist = _naive_ist(pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=PLANT_TZ))
    paths = charged_paths(cfg)
    attempts = AttemptLog(paths.storage_dir)
    attempt: dict[str, Any] = {"forecast": "charged_coke", "at": _utc(now_ist), "issuer": issued_by}
    try:
        built = _build_time(paths.dataset_path)
        hour = now_ist.floor("h")
        if refresh and (built is None or built < hour):
            outcome = refresh_fn(
                {"static_dataset_path": str(paths.dataset_path), "static_dataset_url": paths.dataset_url}
            )
            attempt["refresh"] = {k: str(v) for k, v in dict(outcome or {}).items()}
            built = _build_time(paths.dataset_path)
            source_health.record_dataset_refresh(outcome)  # a failed refresh marks the source (SV-6.5)
        if source is None:
            meta = _meta(paths.dataset_path)
            versioned = paths.dataset_path.parent / str(meta.get("csv_file") or "")
            csv = versioned if meta.get("csv_file") and versioned.is_file() else paths.dataset_path
            source = read_charged_source(csv)
        if engine is None:
            engine = charged_engine(str(active_charged_bundle(paths.storage_dir, paths.default_bundle)))
        ledger = ForecastLedger(paths.storage_dir)
        rows_before = set(ledger.issues())
        record = issue_latest(
            source,
            engine,
            ledger=ledger,
            plans=PlannedChangeRegister(paths.storage_dir),
            now=now_ist,
            built_at=_built_at(paths.dataset_path.parent / "cache_meta.json"),
            model_version=str(engine.model.manifest.get("version", "")),
        )
    except Exception as exc:
        attempt.update(
            status="failed",
            written=False,
            seconds=round(time.monotonic() - t0, 3),
            error=f"{type(exc).__name__}: {exc}"[:400],
        )
        attempts.record(attempt, None)
        raise
    row = pd.Timestamp(record["data_row"]) if record.get("data_row") is not None else None
    target = expected_row(now_ist)
    attempt.update(
        key=row,
        status=record.get("state"),
        written=row is not None and row not in rows_before,
        built_at=built,
        expected_row=target,
        behind=row is None or row < target,
        model_version=record.get("model_version"),
        seconds=round(time.monotonic() - t0, 3),
    )
    if issued_by == "worker" or attempt["written"]:  # page reruns that find the record are not logged
        attempts.record(attempt, record)
    return {"attempt": attempt, "record": record}


def charged_for_page(storage_dir: str | Path, *, now: Any = None) -> PageForecast:
    """The charged-coke record page 9 uses when it reads: the ledger's for the row the latest build holds complete,
    else the previous row's (the worker has not run for this build yet). An older record is out of date: page 9 gets
    a record that is not shown, so the Data-Driven level is unavailable, as for a paused forecast."""
    now_ist = _naive_ist(pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=PLANT_TZ))
    target = expected_row(now_ist)
    issues = ForecastLedger(storage_dir).issues()
    for row, note in (
        (target, None),
        (target - pd.Timedelta(hours=1), "the forecast of the latest build is being issued"),
    ):
        if row in issues:
            record = copy.deepcopy(issues[row])
            return PageForecast(record, f"Forecast for the {row:%H:%M} row; {note}." if note else None, True)
    last = max(issues) if issues else None
    reason = (
        f"No charged-coke forecast has been issued since the {last:%d %b %H:%M} row (expected the {target:%H:%M} row)."
        if last is not None
        else "No charged-coke forecast has been issued yet."
    )
    placeholder = {
        "data_row": None,
        "state": "out_of_date",
        "shown": False,
        "prediction_kg_thm": None,
        "withheld_value_kg_thm": None,
        "reasons": [reason],
        "codes": ["forecast_out_of_date"],
        "last_issued_row": last,
    }
    return PageForecast(placeholder, reason, False)


# --------------------------------------------------------------------------------------------- events
def _ist_aware(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(PLANT_TZ) if stamp.tzinfo is None else stamp.tz_convert(PLANT_TZ)


def si_event(attempt: Mapping[str, Any], record: Mapping[str, Any]) -> SiForecastData:
    """The ``forecast.si`` payload of an attempt."""
    return SiForecastData(
        origin_at=_utc(record.get("origin_at") or attempt["key"]),
        issued_at=_utc(record.get("issued_at") or attempt["at"]),
        status=str(record.get("status") or "unknown"),
        recorded=bool(attempt.get("written") or attempt.get("existing")),
        mode=record.get("mode"),
        model_version=record.get("model_version"),
        si_pct=record.get("si_pct"),
        last_sample_si_pct=record.get("last_sample_si_pct"),
        last_sample_at=_utc(record["last_sample_at"]) if record.get("last_sample_at") else None,
        horizons=[
            SiForecastPoint(
                horizon_minutes=int(h.get("horizon_minutes", 0)),
                label=str(h.get("label") or ("Now" if not h.get("horizon_minutes") else "")),
                target_start=_utc(h["target_start"]),
                target_end=_utc(h["target_end"]),
                si_pct=h.get("si_pct"),
                lower_pct=h.get("lower_pct"),
                upper_pct=h.get("upper_pct"),
                status=h.get("status"),
            )
            for h in forecast_horizons(record)
        ],
        reasons=[str(r) for r in record.get("reasons") or []],
    )


def charged_event(record: Mapping[str, Any]) -> ChargedCokeForecastData | None:
    """The ``forecast.charged_coke`` payload of an issued record (None without a data row)."""
    if record.get("data_row") is None:
        return None
    window = record.get("coke_window") or [None, None]
    return ChargedCokeForecastData(
        data_row=_ist_aware(record["data_row"]),
        issued_at=_ist_aware(record.get("issued_at")) or _ist_aware(record["data_row"]),
        state=str(record.get("state") or "unknown"),
        shown=bool(record.get("shown")),
        prediction_kg_thm=record.get("prediction_kg_thm"),
        lower_kg_thm=record.get("lower_kg_thm"),
        upper_kg_thm=record.get("upper_kg_thm"),
        coke_start=_ist_aware(window[0]),
        coke_end=_ist_aware(window[1]),
        model_version=record.get("model_version"),
        path=[
            ChargedCokePoint(
                horizon=int(point["horizon"]),
                at=_ist_aware(point["at"]),
                prediction_kg_thm=point.get("prediction_kg_thm"),
                lower_kg_thm=point.get("lower_kg_thm"),
                upper_kg_thm=point.get("upper_kg_thm"),
            )
            for point in record.get("path") or []
        ],
        reasons=[str(r) for r in record.get("reasons") or []],
    )


def summary(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """The job result of an attempt: what was issued and how long it took."""
    keys = ("forecast", "key", "status", "written", "existing", "behind", "expected_row", "built_at", "seconds")
    return {k: _jsonable(attempt[k]) for k in keys if k in attempt}


# --------------------------------------------------------------------------------------------- gaps
@dataclass(frozen=True)
class Gap:
    start: pd.Timestamp  # last record before the gap (UTC)
    end: pd.Timestamp  # first record after it, or the end of the window
    minutes: float
    attempts: int
    statuses: dict[str, int]
    errors: list[str]


def ledger_times(forecast: Forecast, storage_dir: str | Path) -> list[pd.Timestamp]:
    """Origins (Si) or data rows (charged coke, IST) of the ledger's records, as UTC times."""
    if forecast == "si":
        return list(SiliconForecastLedger(storage_dir).issues())
    return [_utc(t) for t in ForecastLedger(storage_dir).issues()]


def gaps(forecast: Forecast, storage_dir: str | Path, *, start: Any, end: Any, limit: pd.Timedelta) -> list[Gap]:
    """Intervals of more than ``limit`` between consecutive ledger records in [start, end], with the attempts made
    within each (none: nothing ran; statuses: it ran, but the inputs were not usable or the run failed)."""
    start, end = _utc(start), _utc(end)
    times = [t for t in ledger_times(forecast, storage_dir) if start <= t <= end]
    marks = [start, *times, end]
    attempts = [
        r
        for r in AttemptLog(storage_dir).between(start - pd.Timedelta(hours=2), end + pd.Timedelta(hours=2))
        if r.get("forecast") == forecast
    ]

    def when(row: dict[str, Any]) -> pd.Timestamp:  # the origin or data row attempted; the attempt time without one
        return _utc(row["key"]) if row.get("key") else _utc(row["at"])

    out = []
    for a, b in zip(marks[:-1], marks[1:], strict=True):
        if b - a <= limit:
            continue
        within = [r for r in attempts if a < when(r) < b]
        statuses: dict[str, int] = {}
        for r in within:
            statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1
        errors = sorted({str(r["error"]) for r in within if r.get("error")})[:5]
        out.append(Gap(a, b, (b - a).total_seconds() / 60, len(within), statuses, errors))
    return out


__all__ = [
    "BUILD_MINUTE",
    "ISSUERS",
    "STALE_AFTER",
    "AttemptLog",
    "ChargedPaths",
    "Gap",
    "PageForecast",
    "SiPaths",
    "bmo_config",
    "charged_engine",
    "charged_event",
    "charged_for_page",
    "charged_paths",
    "expected_row",
    "gaps",
    "issue_charged",
    "issue_si",
    "issuer",
    "ledger_times",
    "live_si_source",
    "page_issues",
    "read_charged_source",
    "record_skip",
    "si_cadence",
    "si_event",
    "si_failure_record",
    "si_for_page",
    "si_paths",
    "si_service",
    "summary",
    "worker_active",
]
