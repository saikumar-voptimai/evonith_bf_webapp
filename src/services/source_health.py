"""Health of the data sources the twin reads (SV-6.5, #319): InfluxDB, the hourly dataset, HM/slag, the DPR and
Open-Meteo.

A process-wide registry (``HEALTH``) records every fetch: ``record_ok`` with the newest time the source held (and,
when given, its last value, which is kept and served while the source is down), ``record_failure`` with the error.
A source whose last fetch failed is ``failed`` (its badge is stale at once); for ``retry_after_s`` after the failure
it is ``down``: jobs that need it skip with the reason (``require`` raises ``SourceDown``) instead of timing out
again, and the next job after the hold probes it once more, so the recovery needs no manual step.

``observe_status_view`` records InfluxDB from a furnace-status snapshot (``services.status``: the worker's producer
every 10 s and the API's own snapshots, also the producer's view the API reads from Redis), ``services.twin.
load_frames`` records the four evaluation sources. The Live Twin's source rows (``twin_views.source_freshness``)
merge ``view()``: a source is stale when its data are older than the limit or its last fetch failed.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

log = logging.getLogger("services.sources")

SOURCE_LABELS = {
    "influxdb": "InfluxDB",
    "hourly": "Hourly dataset",
    "hm_slag": "HM/slag",
    "dpr": "DPR",
    "open_meteo": "Open-Meteo",
}
SOURCES = tuple(SOURCE_LABELS)
RETRY_AFTER_S = 300.0  # a job needing a source that failed within this long skips without a probe

IST = "Asia/Kolkata"


class SourceDown(RuntimeError):
    """A source the caller needs failed recently; ``reason`` is the logged text."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(reason)
        self.key = key
        self.reason = reason


@dataclass
class SourceState:
    key: str
    label: str
    last_ok_at: datetime | None = None
    data_end: datetime | None = None
    value: Any = None
    last_failure_at: datetime | None = None
    last_error: str | None = None
    failures: int = 0  # since the last success

    @property
    def failed(self) -> bool:
        """The last fetch failed (no success since)."""
        if self.last_failure_at is None:
            return False
        return self.last_ok_at is None or self.last_failure_at > self.last_ok_at

    def down(self, now: datetime, retry_after_s: float) -> bool:
        return self.failed and (now - self.last_failure_at).total_seconds() < retry_after_s  # type: ignore[operator]

    def retry_at(self, retry_after_s: float) -> datetime | None:
        if not self.failed or self.last_failure_at is None:
            return None
        return self.last_failure_at + timedelta(seconds=retry_after_s)


def _utc(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)  # plant times are IST
    return ts.tz_convert("UTC").to_pydatetime()


def _ist_text(value: datetime | None) -> str:
    if value is None:
        return "?"
    import pandas as pd

    return f"{pd.Timestamp(value).tz_convert(IST):%d %b %H:%M} IST"


def _short(exc: BaseException | str, limit: int = 300) -> str:
    if isinstance(exc, str):
        text = exc
    else:
        text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    text = text.splitlines()[0] if text else text
    return text if len(text) <= limit else text[: limit - 1] + "…"


_KEEP = object()


@dataclass
class SourceHealth:
    """The registry; thread-safe. ``clock`` gives UTC now (tests)."""

    retry_after_s: float = RETRY_AFTER_S
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    _states: dict[str, SourceState] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        for key, label in SOURCE_LABELS.items():
            self._states[key] = SourceState(key, label)

    def _state(self, key: str) -> SourceState:
        if key not in self._states:
            raise KeyError(f"unknown source {key}; one of {', '.join(SOURCES)}")
        return self._states[key]

    # ---------------------------------------------------------------------------------------- recording
    def record_ok(self, key: str, *, data_end: Any = None, value: Any = _KEEP, at: datetime | None = None) -> None:
        """A successful fetch; ``data_end`` the newest time the source held (IST naive or aware; kept when None),
        ``value`` the last value to keep for the source (left as it is when not given)."""
        now = at or self.clock()
        with self._lock:
            s = self._state(key)
            if s.failed:
                log.info("%s reachable again after %d failure%s", s.label, s.failures, "" if s.failures == 1 else "s")
            s.last_ok_at = now
            s.failures = 0
            end = _utc(data_end)
            if end is not None:
                s.data_end = end
            if value is not _KEEP:
                s.value = value

    def record_failure(self, key: str, error: BaseException | str, *, at: datetime | None = None) -> SourceState:
        now = at or self.clock()
        with self._lock:
            s = self._state(key)
            s.last_failure_at = now
            s.last_error = _short(error)
            s.failures += 1
            if s.failures == 1:
                log.warning("%s unreachable (%s); its last values are kept and marked stale", s.label, s.last_error)
            return replace(s)

    def observe(self, key: str, fn: Callable[..., Any], *args: Any, data_end: Callable[[Any], Any] | None = None,
                keep: bool = False, **kwargs: Any) -> Any:  # fmt: skip
        """Run ``fn`` and record the outcome; a failure is raised as ``SourceDown`` (from the error)."""
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            state = self.record_failure(key, exc)
            raise SourceDown(key, self.reason(key) or state.last_error or key) from exc
        end = data_end(result) if data_end is not None else None
        self.record_ok(key, data_end=end, value=result if keep else _KEEP)
        return result

    # ------------------------------------------------------------------------------------------- queries
    def state(self, key: str) -> SourceState:
        with self._lock:
            return copy.copy(self._state(key))

    def failed(self, key: str) -> bool:
        return self.state(key).failed

    def down(self, key: str, now: datetime | None = None) -> bool:
        return self.state(key).down(now or self.clock(), self.retry_after_s)

    def last_value(self, key: str) -> Any:
        return self.state(key).value

    def reason(self, key: str, now: datetime | None = None) -> str | None:
        """Why a job skips: the source, since when it fails, the error and the next try; None when it is up."""
        s = self.state(key)
        if not s.failed:
            return None
        now = now or self.clock()
        text = f"{s.label} unreachable since {_ist_text(s.last_failure_at)} ({s.last_error})"
        if s.down(now, self.retry_after_s):
            text += f"; next try after {_ist_text(s.retry_at(self.retry_after_s))}"
        return text

    def require(self, *keys: str, now: datetime | None = None) -> None:
        """Raise ``SourceDown`` for the first of ``keys`` that is down (failed within ``retry_after_s``)."""
        now = now or self.clock()
        for key in keys:
            if self.down(key, now):
                raise SourceDown(key, self.reason(key, now) or key)

    def view(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """One row per source for the API and the alarms."""
        now = now or self.clock()
        out = []
        with self._lock:
            states = [copy.copy(s) for s in self._states.values()]
        for s in states:
            out.append(
                {
                    "key": s.key,
                    "label": s.label,
                    "failed": s.failed,
                    "down": s.down(now, self.retry_after_s),
                    "last_ok_at": s.last_ok_at,
                    "data_end": s.data_end,
                    "last_failure_at": s.last_failure_at,
                    "last_error": s.last_error,
                    "failures": s.failures,
                    "retry_at": s.retry_at(self.retry_after_s),
                    "has_value": s.value is not None,
                }
            )
        return out

    def reset(self) -> None:
        with self._lock:
            for key, label in SOURCE_LABELS.items():
                self._states[key] = SourceState(key, label)


HEALTH = SourceHealth()


def observe_status_view(view: Any, *, health: SourceHealth = HEALTH) -> bool:
    """Record InfluxDB from a furnace-status snapshot view (``services.status.StatusSnapshotView``): a failure when
    measurements could not be read and the snapshot holds no reading (or is the stale copy of an earlier one, telemetry
    offline); else a success with the newest reading, keeping the view as the source's last value. Returns whether
    the source is up."""
    fresh = getattr(view, "freshness", None)
    if fresh is None:
        return not health.failed("influxdb")
    failed_sources: Iterable[str] = getattr(fresh, "failed_sources", None) or ()
    data_at = getattr(fresh, "data_at", None)
    at = getattr(fresh, "computed_at", None)
    if failed_sources and (data_at is None or getattr(fresh, "telemetry", None) == "offline"):
        names = ", ".join(failed_sources)
        health.record_failure("influxdb", f"no reading; measurements not read: {names}", at=_utc(at))
        return False
    health.record_ok("influxdb", data_end=data_at, value=view, at=_utc(at))  # the last good view is kept
    return True


__all__ = [
    "HEALTH",
    "RETRY_AFTER_S",
    "SOURCES",
    "SOURCE_LABELS",
    "SourceDown",
    "SourceHealth",
    "SourceState",
    "observe_status_view",
]
