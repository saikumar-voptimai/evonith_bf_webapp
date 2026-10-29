"""Buffered twin store of the worker (SV-6.5, #319): writes Neon cannot take now go to a local SQLite outbox
(``data.twin.outbox``) and are replayed in order when Neon returns.

``BufferedTwinStore`` wraps a ``TwinStore``. Its ``snapshots.upsert``, ``jobs.upsert`` and ``jobs.upsert_many`` are
buffered; every other repository and method is the wrapped store's. A write goes to Neon directly while the outbox is
empty and the last direct write did not fail within ``retry_after_s``; otherwise it is appended to the outbox, which
keeps the order of the writes. An outage is a connection-level error (``is_outage``); any other error (a bad payload,
an integrity violation) is raised to the caller as before.

``replay()`` writes the queued entries oldest first and stops at the first that fails: a failed replay of one entry
never lets a later one overtake it. Every operation is an upsert by its key, so an entry replayed twice (a crash
between the Neon commit and the outbox delete) leaves the same row. ``replay_pending()`` is called by the worker's
heartbeat every ``REPLAY_INTERVAL_S`` while entries are queued, so the recovery needs no manual step; the nightly
housekeeping reports the outbox in its result.

The process's store: ``buffered_twin_store()`` (the worker; None without a twin store). The outbox file is
``BF_TWIN_OUTBOX`` or ``src/storage/twin/outbox.sqlite``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from data.twin.outbox import Entry, Outbox
from data.twin.repositories import TwinStore
from services import store as store_service

log = logging.getLogger("services.outbox")

OUTBOX_ENV = "BF_TWIN_OUTBOX"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "storage" / "twin" / "outbox.sqlite"
RETRY_AFTER_S = 30.0  # after a failed direct write, writes go straight to the outbox for this long
REPLAY_INTERVAL_S = 30.0  # the heartbeat's replay cadence while entries are queued
REPLAY_BATCH = 200
POISON_TRIES = 10  # an entry Neon refuses (not an outage) this often is dropped with an error log

Operation = Callable[[TwinStore, Any], Any]
OPERATIONS: dict[str, Operation] = {
    "snapshot.upsert": lambda store, payload: store.snapshots.upsert(payload),
    "job.upsert": lambda store, payload: store.jobs.upsert(payload),
    "job.upsert_many": lambda store, payload: store.jobs.upsert_many(payload),
}


def is_outage(exc: BaseException) -> bool:
    """A connection-level failure (Neon down, DNS, timeout, a connection the pool found dead); not a schema or data
    error, which a replay would repeat."""
    if isinstance(exc, (OperationalError, InterfaceError)):
        return True
    if isinstance(exc, DBAPIError):
        return bool(exc.connection_invalidated)
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def outbox_path() -> Path:
    value = os.environ.get(OUTBOX_ENV, "").strip()
    return Path(value) if value else DEFAULT_PATH


def _short(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    text = text.splitlines()[0] if text else text
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ReplayReport:
    replayed: int = 0
    dropped: int = 0
    pending: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"replayed": self.replayed, "dropped": self.dropped, "pending": self.pending, "error": self.error}


class _BufferedRepository:
    """A repository whose named writes go through the buffer; everything else is the wrapped repository's."""

    def __init__(self, buffered: BufferedTwinStore, inner: Any) -> None:
        self._buffered = buffered
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _BufferedSnapshots(_BufferedRepository):
    def upsert(self, values: Mapping[str, Any]) -> Any:
        """The stored row, or None when the write was buffered."""
        key = str(values.get("window_end"))
        return self._buffered.write("snapshot.upsert", key, dict(values))


class _BufferedJobs(_BufferedRepository):
    def upsert(self, record: Mapping[str, Any], *, at: datetime | None = None) -> None:
        self._buffered.write("job.upsert", str(record.get("id")), dict(record))

    def upsert_many(self, records: Iterable[Mapping[str, Any]], *, at: datetime | None = None) -> int | None:
        rows = [dict(r) for r in records]
        if not rows:
            return 0
        key = str(rows[0].get("id")) if len(rows) == 1 else f"{rows[0].get('id')} +{len(rows) - 1}"
        return self._buffered.write("job.upsert_many", key, rows)


class BufferedTwinStore:
    """``TwinStore`` with buffered snapshot and job writes (see the module docstring)."""

    def __init__(
        self,
        store: TwinStore,
        outbox: Outbox,
        *,
        retry_after_s: float = RETRY_AFTER_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.outbox = outbox
        self.retry_after_s = float(retry_after_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._failed_at: float | None = None  # monotonic time of the last failed direct write or replay
        self.down_since: datetime | None = None
        self.last_error: str | None = None
        self.last_replay_at: datetime | None = None
        self.replayed = 0
        self.buffered = 0
        self.snapshots = _BufferedSnapshots(self, store.snapshots)
        self.jobs = _BufferedJobs(self, store.jobs)

    def __getattr__(self, name: str) -> Any:  # alarms, optimiser, shifts, ..., revision, dispose
        return getattr(self.store, name)

    # ------------------------------------------------------------------------------------------ state
    @property
    def pending(self) -> int:
        return self.outbox.pending()

    @property
    def holding(self) -> bool:
        """Writes go to the outbox without a try: a direct write failed less than ``retry_after_s`` ago."""
        return self._failed_at is not None and self._clock() - self._failed_at < self.retry_after_s

    def health(self) -> dict[str, Any]:
        return {
            "pending": self.pending,
            "oldest_queued_at": self.outbox.oldest_queued_at(),
            "down_since": self.down_since,
            "last_error": self.last_error,
            "last_replay_at": self.last_replay_at,
            "replayed": self.replayed,
            "buffered": self.buffered,
            "path": str(self.outbox.path),
        }

    # ----------------------------------------------------------------------------------------- writes
    def write(self, op: str, key: str, payload: Any) -> Any:
        """Apply ``op`` in Neon, or queue it. Returns the operation's result, or None when queued."""
        apply = OPERATIONS[op]
        with self._lock:
            if self.pending and not self.holding:
                self._replay_locked(REPLAY_BATCH)
            if self.pending or self.holding:
                return self._queue(op, key, payload)
            try:
                result = apply(self.store, payload)
            except Exception as exc:
                if not is_outage(exc):
                    raise
                self._failed(exc)
                log.warning("twin store unreachable (%s); %s %s buffered in %s", _short(exc), op, key, self.outbox.path)
                return self._queue(op, key, payload)
            self._recovered()
            return result

    def _queue(self, op: str, key: str, payload: Any) -> None:
        self.outbox.append(op, key, payload)
        self.buffered += 1
        return None

    def _failed(self, exc: BaseException) -> None:
        self._failed_at = self._clock()
        self.last_error = _short(exc)
        if self.down_since is None:
            self.down_since = datetime.now(UTC)

    def _recovered(self) -> None:
        if self.down_since is not None:
            log.info("twin store reachable again (down since %s)", self.down_since.isoformat(timespec="seconds"))
        self._failed_at = None
        self.down_since = None

    # ----------------------------------------------------------------------------------------- replay
    def replay(self, limit: int = REPLAY_BATCH) -> ReplayReport:
        """Write up to ``limit`` queued entries in order; stops at the first that Neon cannot take."""
        with self._lock:
            return self._replay_locked(limit)

    def _replay_locked(self, limit: int) -> ReplayReport:
        report = ReplayReport()
        for entry in self.outbox.oldest(limit):
            try:
                self._apply(entry)
            except Exception as exc:
                self.outbox.mark_failed(entry.seq, _short(exc))
                if is_outage(exc):
                    self._failed(exc)
                    report.error = _short(exc)
                    break
                if entry.tries + 1 >= POISON_TRIES:
                    log.error("outbox entry %d (%s %s) refused %d times (%s); dropped", entry.seq, entry.op,
                              entry.key, entry.tries + 1, _short(exc))  # fmt: skip
                    self.outbox.delete(entry.seq)
                    report.dropped += 1
                    continue
                log.error("outbox entry %d (%s %s) refused (%s); replay stops here", entry.seq, entry.op, entry.key,
                          _short(exc))  # fmt: skip
                report.error = _short(exc)
                break
            self.outbox.delete(entry.seq)
            report.replayed += 1
            self.replayed += 1
        if report.replayed:
            self.last_replay_at = datetime.now(UTC)
            self._recovered()
            log.info("outbox: %d write%s replayed into the twin store", report.replayed,
                     "" if report.replayed == 1 else "s")  # fmt: skip
        report.pending = self.pending
        return report

    def _apply(self, entry: Entry) -> None:
        apply = OPERATIONS.get(entry.op)
        if apply is None:
            raise ValueError(f"unknown outbox operation {entry.op}")
        apply(self.store, entry.payload)


# ---------------------------------------------------------------------------------------- the process's store
_UNSET = object()
_STORE: object = _UNSET
_WRAPPED: tuple[Any, Path, BufferedTwinStore | None] | None = None  # (inner store, outbox path, wrapper)
_LOCK = threading.Lock()


def buffered_twin_store(application_name: str = "twin-worker") -> BufferedTwinStore | None:
    """The process's buffered store over ``services.store.twin_store()`` (wrapped once per store and outbox path);
    None without a configured twin store."""
    global _WRAPPED
    with _LOCK:
        if _STORE is not _UNSET:
            return _STORE  # type: ignore[return-value]
        inner, path = store_service.twin_store(application_name), outbox_path()
        if _WRAPPED is None or _WRAPPED[0] is not inner or _WRAPPED[1] != path:
            _WRAPPED = (inner, path, None if inner is None else BufferedTwinStore(inner, Outbox(path)))
        return _WRAPPED[2]


def use(store: BufferedTwinStore | None) -> None:
    """Replace the process's buffered store (tests); ``reset()`` restores opening from the environment."""
    global _STORE
    with _LOCK:
        _STORE = store


def reset() -> None:
    global _STORE, _WRAPPED
    with _LOCK:
        _STORE, _WRAPPED = _UNSET, None


def replay_pending() -> ReplayReport | None:
    """Replay the queued entries of the process's store (the heartbeat, housekeeping); None without a store or
    nothing queued."""
    store = buffered_twin_store()
    if store is None or not store.pending or store.holding:
        return None
    return store.replay()


def outbox_health() -> dict[str, Any] | None:
    store = buffered_twin_store()
    return None if store is None else store.health()


__all__ = [
    "DEFAULT_PATH",
    "OPERATIONS",
    "REPLAY_INTERVAL_S",
    "BufferedTwinStore",
    "ReplayReport",
    "buffered_twin_store",
    "is_outage",
    "outbox_health",
    "outbox_path",
    "replay_pending",
    "reset",
    "use",
]
