"""Sign-in failure limit per client address and the revoked-session list.

Both hold their state in the API process. One API process (the Jetson service unit) shares them across all requests;
with several processes each keeps its own state, so the Redis-backed versions arrive with the job queue (SV-0.8, #277)
behind the same methods.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable


class FailureLimiter:
    """At most ``max_failures`` failed sign-ins per key (client address) within a sliding window of ``window_s``.

    A success does not clear the record, so successful sign-ins with one account cannot hide guesses at another.
    """

    def __init__(
        self,
        max_failures: int = 5,
        window_s: float = 60.0,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_failures = max_failures
        self.window_s = window_s
        self._max_keys = max_keys
        self._clock = clock
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float] | None:
        times = self._failures.get(key)
        if times is None:
            return None
        while times and times[0] <= now - self.window_s:
            times.popleft()
        if not times:
            del self._failures[key]
            return None
        return times

    def retry_after(self, key: str) -> float:
        """Seconds until ``key`` may try again; 0 when a further attempt is allowed."""
        with self._lock:
            now = self._clock()
            times = self._prune(key, now)
            if times is None or len(times) < self.max_failures:
                return 0.0
            return max(0.0, times[-self.max_failures] + self.window_s - now)

    def record_failure(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            times = self._prune(key, now)
            if times is None:
                times = deque()
                self._failures[key] = times
            times.append(now)
            self._failures.move_to_end(key)
            while len(self._failures) > self._max_keys:
                self._failures.popitem(last=False)


class DenyList:
    """Token ids of ended sessions, kept until the token would have expired anyway."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._entries: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, jti: str, expires_at: float) -> None:
        with self._lock:
            now = self._clock()
            for key in [k for k, exp in self._entries.items() if exp <= now]:
                del self._entries[key]
            if expires_at > now:
                self._entries[jti] = expires_at

    def contains(self, jti: str) -> bool:
        with self._lock:
            exp = self._entries.get(jti)
            return exp is not None and exp > self._clock()
