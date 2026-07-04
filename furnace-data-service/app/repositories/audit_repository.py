"""SQLite audit event repository."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from furnace_data.runtime_paths import runtime_path


def default_audit_db_url() -> str:
    return f"sqlite:///{runtime_path('audit', 'audit.db', create_parent=True).as_posix()}"


def sqlite_url_to_path(database_url: str) -> Path | None:
    if database_url in {"", ":memory:", "sqlite:///:memory:"}:
        return None
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Only SQLite audit storage is supported in Phase 10.")
    return Path(database_url[len(prefix) :])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str)


def _from_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return dict(json.loads(value))
    except (TypeError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class AuditEventRecord:
    id: str
    timestamp: str
    request_id: str | None
    actor_user_id: str | None
    actor_username: str | None
    event_type: str
    resource_type: str | None
    resource_id: str | None
    action: str
    result: str
    status_code: int | None
    error_code: str | None
    ip_hash: str | None
    metadata: dict[str, Any]
    created_at: str


class AuditRepository:
    """Persist redacted audit events in SQLite."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or default_audit_db_url()
        self._schema_ready = False
        self._lock = threading.Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db_path = sqlite_url_to_path(self.database_url)
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(":memory:" if db_path is None else str(db_path))
        connection.row_factory = sqlite3.Row
        try:
            if db_path is not None:
                connection.execute("PRAGMA journal_mode=WAL")
            if not self._schema_ready:
                self._ensure_schema(connection)
                self._schema_ready = True
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                request_id TEXT,
                actor_user_id TEXT,
                actor_username TEXT,
                event_type TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                action TEXT NOT NULL,
                result TEXT NOT NULL,
                status_code INTEGER,
                error_code TEXT,
                ip_hash TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_audit_timestamp ON audit_events(timestamp);
            CREATE INDEX IF NOT EXISTS ix_audit_event_type ON audit_events(event_type);
            CREATE INDEX IF NOT EXISTS ix_audit_actor ON audit_events(actor_user_id);
            """
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> AuditEventRecord:
        return AuditEventRecord(
            id=str(row["id"]),
            timestamp=str(row["timestamp"]),
            request_id=row["request_id"],
            actor_user_id=row["actor_user_id"],
            actor_username=row["actor_username"],
            event_type=str(row["event_type"]),
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            action=str(row["action"]),
            result=str(row["result"]),
            status_code=row["status_code"],
            error_code=row["error_code"],
            ip_hash=row["ip_hash"],
            metadata=_from_json(row["metadata_json"]),
            created_at=str(row["created_at"]),
        )

    def ensure_schema(self) -> None:
        with self._lock, self._connect():
            return

    def insert_event(self, payload: dict[str, Any]) -> AuditEventRecord:
        event_id = payload.get("id") or f"audit_{uuid4().hex}"
        timestamp = payload.get("timestamp") or _now()
        created_at = payload.get("created_at") or _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    id, timestamp, request_id, actor_user_id, actor_username,
                    event_type, resource_type, resource_id, action, result,
                    status_code, error_code, ip_hash, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    timestamp,
                    payload.get("request_id"),
                    payload.get("actor_user_id"),
                    payload.get("actor_username"),
                    payload["event_type"],
                    payload.get("resource_type"),
                    payload.get("resource_id"),
                    payload.get("action") or "unknown",
                    payload.get("result") or "unknown",
                    payload.get("status_code"),
                    payload.get("error_code"),
                    payload.get("ip_hash"),
                    _json(payload.get("metadata") or {}),
                    created_at,
                ),
            )
            row = connection.execute("SELECT * FROM audit_events WHERE id = ?", (event_id,)).fetchone()
            return self._record(row)

    def list_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
    ) -> tuple[list[AuditEventRecord], int]:
        conditions: list[str] = []
        values: list[Any] = []
        if event_type:
            conditions.append("event_type = ?")
            values.append(event_type)
        if actor_user_id:
            conditions.append("actor_user_id = ?")
            values.append(actor_user_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._lock, self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM audit_events {where}",
                    values,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM audit_events
                {where}
                ORDER BY timestamp DESC, seq DESC
                LIMIT ? OFFSET ?
                """,
                [*values, min(500, max(1, int(limit))), max(0, int(offset))],
            ).fetchall()
            return [self._record(row) for row in rows], total

    def cleanup_before(self, cutoff_iso: str) -> int:
        with self._lock, self._connect() as connection:
            result = connection.execute(
                "DELETE FROM audit_events WHERE timestamp < ?",
                (cutoff_iso,),
            )
            return int(result.rowcount or 0)

