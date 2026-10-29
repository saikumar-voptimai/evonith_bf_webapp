"""Local outbox of the twin store (SV-6.5, #319): writes Neon could not take, kept in a SQLite file.

One table, ``outbox``: a sequence number (the replay order), the time the write was queued, the operation
(``snapshot.upsert``, ``job.upsert``, ...), its key (window end, job id) and the payload as JSON. The file is written
in WAL mode with ``synchronous=FULL``: an entry that ``append`` returned survives a power loss. Entries are deleted
only after the write committed in Neon, so a crash between the two replays the entry again, which the upsert
operations take without a change (idempotent replay). ``services.outbox`` holds the store wrapper and the replay.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

DDL = """
CREATE TABLE IF NOT EXISTS outbox (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    queued_at  TEXT NOT NULL,
    op         TEXT NOT NULL,
    key        TEXT NOT NULL,
    payload    TEXT NOT NULL,
    tries      INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
)
"""


@dataclass(frozen=True)
class Entry:
    seq: int
    queued_at: datetime
    op: str
    key: str
    payload: Any
    tries: int
    last_error: str | None


# ---------------------------------------------------------------------------------------------------- JSON
def _default(value: Any) -> Any:
    if isinstance(value, datetime):  # pandas Timestamps included
        return {"$dt": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, uuid.UUID):
        return {"$uuid": str(value)}
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _clean(value: Any) -> Any:
    """NaN and infinities become null (JSON has none; the store takes None), containers are walked."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean(v) for v in value]
    return value


def _hook(obj: dict[str, Any]) -> Any:
    if len(obj) == 1:
        if "$dt" in obj:
            return datetime.fromisoformat(obj["$dt"])
        if "$date" in obj:
            return date.fromisoformat(obj["$date"])
        if "$uuid" in obj:
            return uuid.UUID(obj["$uuid"])
    return obj


def encode(payload: Any) -> str:
    return json.dumps(_clean(payload), default=_default, allow_nan=False, separators=(",", ":"))


def decode(text: str) -> Any:
    return json.loads(text, object_hook=_hook)


# -------------------------------------------------------------------------------------------------- outbox
class Outbox:
    """Append-only queue in one SQLite file; every method opens its own connection (safe from any thread)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(DDL)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=FULL")
        return con

    def append(self, op: str, key: str, payload: Any, *, at: datetime | None = None) -> int:
        """Queue one write; returns its sequence number once it is on disk."""
        queued = (at or datetime.now(UTC)).astimezone(UTC).isoformat()
        text = encode(payload)
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO outbox (queued_at, op, key, payload) VALUES (?, ?, ?, ?)", (queued, op, key, text)
            )
            return int(cur.lastrowid or 0)

    def pending(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM outbox").fetchone()[0])

    def oldest(self, limit: int = 100) -> list[Entry]:
        """The next entries to replay, in queue order."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT seq, queued_at, op, key, payload, tries, last_error FROM outbox ORDER BY seq LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            Entry(int(seq), datetime.fromisoformat(queued), op, key, decode(payload), int(tries), error)
            for seq, queued, op, key, payload, tries, error in rows
        ]

    def delete(self, seq: int) -> None:
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM outbox WHERE seq = ?", (int(seq),))

    def mark_failed(self, seq: int, error: str) -> None:
        with self._lock, self._connect() as con:
            con.execute("UPDATE outbox SET tries = tries + 1, last_error = ? WHERE seq = ?", (error[:400], int(seq)))

    def oldest_queued_at(self) -> datetime | None:
        with self._connect() as con:
            row = con.execute("SELECT MIN(queued_at) FROM outbox").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None


__all__ = ["DDL", "Entry", "Outbox", "decode", "encode"]
