"""SQLite repository for FurnaceMind durable background tasks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from apps.backend_api.app.repositories.furnacemind_conversation_repository import (
    default_furnacemind_db_url,
    sqlite_url_to_path,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, default=str)


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def idempotency_hash(idempotency_key: str | None) -> str | None:
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FurnaceMindTaskRecord:
    id: str
    task_type: str
    resource_id: str | None
    owner_id: str | None
    status: str
    progress: float | None
    current_step: str | None
    error_code: str | None
    error_message: str | None
    warnings: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    result: dict[str, Any]
    request: dict[str, Any]
    idempotency_key_hash: str | None
    request_fingerprint: str | None
    created_at: datetime
    updated_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class FurnaceMindTaskEventRecord:
    id: str
    task_id: str
    task_type: str
    resource_id: str | None
    event_type: str
    sequence: int
    payload: dict[str, Any]
    created_at: datetime


class FurnaceMindTaskRepository:
    """Durable task/event storage with owner-scoped idempotency."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or default_furnacemind_db_url()
        self._schema_ready = False

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
            connection.execute("PRAGMA foreign_keys=ON")
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

    def db_path(self) -> Path | None:
        return sqlite_url_to_path(self.database_url)

    def ensure_schema(self) -> None:
        with self._connect():
            return

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS furnacemind_tasks (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                task_type TEXT NOT NULL,
                resource_id TEXT,
                owner_id TEXT,
                status TEXT NOT NULL,
                progress REAL,
                current_step TEXT,
                error_code TEXT,
                error_message TEXT,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                result_json TEXT NOT NULL DEFAULT '{}',
                request_json TEXT NOT NULL DEFAULT '{}',
                idempotency_key_hash TEXT,
                request_fingerprint TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_fm_tasks_resource
                ON furnacemind_tasks(task_type, resource_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_fm_tasks_owner
                ON furnacemind_tasks(owner_id, task_type, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_fm_tasks_owner_type_idempotency
                ON furnacemind_tasks(owner_id, task_type, idempotency_key_hash)
                WHERE idempotency_key_hash IS NOT NULL;

            CREATE TABLE IF NOT EXISTS furnacemind_task_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                task_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                resource_id TEXT,
                event_type TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES furnacemind_tasks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_fm_task_events_task_sequence
                ON furnacemind_task_events(task_id, sequence);
            """
        )

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> FurnaceMindTaskRecord:
        return FurnaceMindTaskRecord(
            id=str(row["id"]),
            task_type=str(row["task_type"]),
            resource_id=row["resource_id"],
            owner_id=row["owner_id"],
            status=str(row["status"]),
            progress=row["progress"],
            current_step=row["current_step"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            warnings=list(_from_json(row["warnings_json"], [])),
            artifacts=list(_from_json(row["artifacts_json"], [])),
            result=dict(_from_json(row["result_json"], {})),
            request=dict(_from_json(row["request_json"], {})),
            idempotency_key_hash=row["idempotency_key_hash"],
            request_fingerprint=row["request_fingerprint"],
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]),
            completed_at=_parse_dt(row["completed_at"]),
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> FurnaceMindTaskEventRecord:
        return FurnaceMindTaskEventRecord(
            id=str(row["id"]),
            task_id=str(row["task_id"]),
            task_type=str(row["task_type"]),
            resource_id=row["resource_id"],
            event_type=str(row["event_type"]),
            sequence=int(row["sequence"]),
            payload=dict(_from_json(row["payload_json"], {})),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
        )

    def create_task(self, payload: dict[str, Any]) -> FurnaceMindTaskRecord:
        task_id = payload.get("id") or f"fmt_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO furnacemind_tasks (
                    id, task_type, resource_id, owner_id, status, progress,
                    current_step, error_code, error_message, warnings_json,
                    artifacts_json, result_json, request_json,
                    idempotency_key_hash, request_fingerprint, created_at,
                    updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    payload["task_type"],
                    payload.get("resource_id"),
                    payload.get("owner_id"),
                    payload.get("status") or "pending",
                    payload.get("progress"),
                    payload.get("current_step"),
                    payload.get("error_code"),
                    payload.get("error_message"),
                    _json(payload.get("warnings") or []),
                    _json(payload.get("artifacts") or []),
                    _json(payload.get("result") or {}),
                    _json(payload.get("request") or {}),
                    payload.get("idempotency_key_hash"),
                    payload.get("request_fingerprint"),
                    now,
                    now,
                    payload.get("completed_at"),
                ),
            )
            row = connection.execute("SELECT * FROM furnacemind_tasks WHERE id = ?", (task_id,)).fetchone()
            return self._task_from_row(row)

    def get_task(self, task_id: str) -> FurnaceMindTaskRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM furnacemind_tasks WHERE id = ?", (task_id,)).fetchone()
            return self._task_from_row(row) if row else None

    def find_by_idempotency(self, *, owner_id: str, task_type: str, idempotency_key_hash: str) -> FurnaceMindTaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM furnacemind_tasks
                WHERE owner_id = ? AND task_type = ? AND idempotency_key_hash = ?
                """,
                (owner_id, task_type, idempotency_key_hash),
            ).fetchone()
            return self._task_from_row(row) if row else None

    def latest_for_resource(self, *, task_type: str, resource_id: str, owner_id: str | None = None) -> FurnaceMindTaskRecord | None:
        clauses = ["task_type = ?", "resource_id = ?"]
        values: list[Any] = [task_type, resource_id]
        if owner_id is not None:
            clauses.append("owner_id = ?")
            values.append(owner_id)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM furnacemind_tasks WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, seq DESC LIMIT 1",
                values,
            ).fetchone()
            return self._task_from_row(row) if row else None

    def list_tasks(self, filters: dict[str, Any]) -> tuple[list[FurnaceMindTaskRecord], int]:
        clauses: list[str] = []
        values: list[Any] = []
        for key in ("owner_id", "task_type", "resource_id", "status"):
            if filters.get(key) is not None:
                clauses.append(f"{key} = ?")
                values.append(str(filters[key]))
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = min(200, max(1, int(filters.get("limit") or 50)))
        offset = max(0, int(filters.get("offset") or 0))
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM furnacemind_tasks{where_sql}", values).fetchone()[0])
            rows = connection.execute(
                "SELECT * FROM furnacemind_tasks"
                + where_sql
                + " ORDER BY created_at DESC, seq DESC LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
            return [self._task_from_row(row) for row in rows], total

    def update_task_status(self, task_id: str, **updates: Any) -> FurnaceMindTaskRecord | None:
        allowed = {
            "resource_id",
            "status",
            "progress",
            "current_step",
            "error_code",
            "error_message",
            "warnings_json",
            "artifacts_json",
            "result_json",
            "completed_at",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if "warnings" in updates:
            values["warnings_json"] = _json(updates["warnings"])
        if "artifacts" in updates:
            values["artifacts_json"] = _json(updates["artifacts"])
        if "result" in updates:
            values["result_json"] = _json(updates["result"])
        values["updated_at"] = _iso()
        if values.get("status") in {"completed", "failed", "cancelled"} and not values.get("completed_at"):
            values["completed_at"] = _iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            result = connection.execute(
                f"UPDATE furnacemind_tasks SET {assignments} WHERE id = ?",
                [*values.values(), task_id],
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute("SELECT * FROM furnacemind_tasks WHERE id = ?", (task_id,)).fetchone()
            return self._task_from_row(row) if row else None

    def append_event(self, payload: dict[str, Any]) -> FurnaceMindTaskEventRecord:
        event_id = f"fmte_{uuid4().hex}"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM furnacemind_task_events WHERE task_id = ?",
                (payload["task_id"],),
            ).fetchone()
            sequence = int(row[0])
            connection.execute(
                """
                INSERT INTO furnacemind_task_events (
                    id, task_id, task_type, resource_id, event_type,
                    sequence, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    payload["task_id"],
                    payload["task_type"],
                    payload.get("resource_id"),
                    payload["event_type"],
                    sequence,
                    _json(payload.get("payload") or {}),
                    _iso(),
                ),
            )
            row = connection.execute("SELECT * FROM furnacemind_task_events WHERE id = ?", (event_id,)).fetchone()
            return self._event_from_row(row)

    def list_events(self, task_id: str, *, limit: int = 500, offset: int = 0) -> list[FurnaceMindTaskEventRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM furnacemind_task_events
                WHERE task_id = ?
                ORDER BY sequence ASC
                LIMIT ? OFFSET ?
                """,
                (task_id, min(1000, max(1, int(limit))), max(0, int(offset))),
            ).fetchall()
            return [self._event_from_row(row) for row in rows]