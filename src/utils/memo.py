"""Process-wide memoisation for code that runs outside Streamlit (service layer, data adapters, API, worker).

It reproduces the parts of ``st.cache_data`` the services relied on:

* the key is the call's arguments by value (pickled), so small tokens - a file modification time, a clock rounded
  to the refresh interval, a refresh counter - decide when an entry is reused;
* ``ttl_seconds`` limits the age of an entry; ``max_entries`` evicts the least recently used entry;
* exceptions are not cached;
* ``copy=True`` returns a deep copy, so a caller that mutates the result cannot change the cached value
  (``st.cache_data`` always copies; here the caller opts in where results are mutated).

Thread-safe: the API serves requests from a thread pool. A slow miss runs outside the lock, so it does not block
calls with other keys; two simultaneous misses on one key compute twice and the later result is kept.
"""

from __future__ import annotations

import copy as _copy
import functools
import hashlib
import pickle
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

_REGISTRY: list[Memoized[..., Any]] = []
_REGISTRY_LOCK = threading.Lock()


def _key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    try:
        raw = pickle.dumps((args, sorted(kwargs.items())), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 - an unpicklable token falls back to its repr
        raw = repr((args, sorted(kwargs.items()))).encode()
    return hashlib.sha256(raw).hexdigest()


class Memoized[**P, R]:
    """A memoised function; call it like the original, ``clear()`` empties its cache."""

    def __init__(self, fn: Callable[P, R], ttl_seconds: float | None, max_entries: int | None, copy: bool) -> None:
        functools.update_wrapper(self, fn)
        self._fn = fn
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.copy = copy
        self._store: OrderedDict[str, tuple[float, R]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key = _key(args, kwargs)
        with self._lock:
            entry = self._store.get(key)
            if entry is not None and (self.ttl_seconds is None or time.monotonic() - entry[0] < self.ttl_seconds):
                self._store.move_to_end(key)
                self.hits += 1
                return self._out(entry[1])
        value = self._fn(*args, **kwargs)
        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            self.misses += 1
            while self.max_entries is not None and len(self._store) > self.max_entries:
                self._store.popitem(last=False)
        return self._out(value)

    def _out(self, value: R) -> R:
        return _copy.deepcopy(value) if self.copy else value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class _Decorator(Protocol):
    def __call__[**P, R](self, fn: Callable[P, R], /) -> Memoized[P, R]: ...


def memo(*, ttl_seconds: float | None = None, max_entries: int | None = None, copy: bool = False) -> _Decorator:
    """Decorator: ``@memo(ttl_seconds=5)`` / ``@memo(max_entries=2)`` / ``@memo(copy=True)``."""

    def wrap[**P, R](fn: Callable[P, R]) -> Memoized[P, R]:
        m = Memoized(fn, ttl_seconds, max_entries, copy)
        with _REGISTRY_LOCK:
            _REGISTRY.append(m)
        return m

    return wrap


def clear_all() -> None:
    """Empty every memoised function's cache (tests, a manual refresh of everything)."""
    with _REGISTRY_LOCK:
        registered = list(_REGISTRY)
    for m in registered:
        m.clear()
