"""Lightweight in-process operational metrics."""

from __future__ import annotations

import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any


class MetricsService:
    """Store safe aggregate counters without external infrastructure."""

    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()
        self._requests_total = 0
        self._errors_total = 0
        self._duration_total_ms = 0.0
        self._duration_max_ms = 0.0
        self._status_codes: Counter[str] = Counter()
        self._methods: Counter[str] = Counter()
        self._routes: Counter[str] = Counter()
        self._error_codes: Counter[str] = Counter()

    def record_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_ms: float,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            self._requests_total += 1
            if status_code >= 500 or error_code:
                self._errors_total += 1
            self._duration_total_ms += float(duration_ms)
            self._duration_max_ms = max(self._duration_max_ms, float(duration_ms))
            self._status_codes[str(status_code)] += 1
            self._methods[str(method).upper()] += 1
            self._routes[str(route or "unknown")] += 1
            if error_code:
                self._error_codes[str(error_code)] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average = (
                self._duration_total_ms / self._requests_total
                if self._requests_total
                else 0.0
            )
            return {
                "started_at": self.started_at.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "requests_total": self._requests_total,
                "errors_total": self._errors_total,
                "duration_ms": {
                    "average": round(average, 3),
                    "max": round(self._duration_max_ms, 3),
                },
                "status_codes": dict(sorted(self._status_codes.items())),
                "methods": dict(sorted(self._methods.items())),
                "routes": dict(self._routes.most_common(50)),
                "error_codes": dict(self._error_codes.most_common(50)),
            }

    def reset(self) -> None:
        with self._lock:
            self.started_at = datetime.now(timezone.utc)
            self._requests_total = 0
            self._errors_total = 0
            self._duration_total_ms = 0.0
            self._duration_max_ms = 0.0
            self._status_codes.clear()
            self._methods.clear()
            self._routes.clear()
            self._error_codes.clear()

