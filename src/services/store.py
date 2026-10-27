"""The twin store of this process (``twin`` schema, ``data.twin``), opened on first use (SV-1.6, #286).

``twin_store()`` returns the store on ``BF_TWIN_DB_URL``, else ``DATABASE_URL``, or None when neither is set (or
``BF_TWIN_DB_URL=off``, as in the test suite). The API (twin and optimiser routes) and the worker (optimiser runs) use
it; the worker's job archive keeps its own (``services.job_archive``), which it closes on shutdown. Opening the store
makes no connection: a database that is down fails the first query, not the start of the process. ``use(store)``
replaces it (tests: a SQLite store).
"""

from __future__ import annotations

import threading

from data.twin.repositories import TwinStore

_UNSET = object()
_STORE: object = _UNSET
_LOCK = threading.Lock()


def twin_store(application_name: str = "twin") -> TwinStore | None:
    """The process's store (opened once with ``application_name``), or None without a configured database."""
    global _STORE
    with _LOCK:
        if _STORE is _UNSET:
            _STORE = TwinStore.from_url(application_name=application_name)
        return _STORE  # type: ignore[return-value]


def use(store: TwinStore | None) -> None:
    """Replace the process's store (tests); ``reset()`` restores opening from the environment."""
    global _STORE
    with _LOCK:
        _STORE = store


def reset() -> None:
    global _STORE
    with _LOCK:
        _STORE = _UNSET


__all__ = ["reset", "twin_store", "use"]
