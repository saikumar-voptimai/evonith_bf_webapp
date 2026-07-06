"""SQLite repository for FurnaceMind runs and events."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from app.repositories.furnacemind_conversation_repository import (
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


@dataclass(frozen=True)
class RunRecord:
    id: str
    conversation_id: str
    owner_id: str | None
    status: str
    user_message_id: str | None
    assistant_message_id: str | None
    progress: float | None
    current_step: str | None
    error_code: str | None
    error_message: str | None
    warnings: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class RunEventRecord:
    id: str
    run_id: str
    conversation_id: str
    event_type: str
    sequence: int
    payload: dict[str, Any]
    created_at: datetime


class FurnaceMindRunRepository:
    """Run/event storage with lazy SQLite initialization."""

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
            CREATE TABLE IF NOT EXISTS furnacemind_runs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                owner_id TEXT,
                status TEXT NOT NULL,
                user_message_id TEXT,
                assistant_message_id TEXT,
                progress REAL,
                current_step TEXT,
                error_code TEXT,
                error_message TEXT,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_fm_runs_conversation
                ON furnacemind_runs(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_fm_runs_owner
                ON furnacemind_runs(owner_id, created_at);

            CREATE TABLE IF NOT EXISTS furnacemind_run_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_fm_events_run_sequence
                ON furnacemind_run_events(run_id, sequence);
            """
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            owner_id=row["owner_id"],
            status=str(row["status"]),
            user_message_id=row["user_message_id"],
            assistant_message_id=row["assistant_message_id"],
            progress=row["progress"],
            current_step=row["current_step"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            warnings=list(_from_json(row["warnings_json"], [])),
            artifacts=list(_from_json(row["artifacts_json"], [])),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]),
            completed_at=_parse_dt(row["completed_at"]),
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> RunEventRecord:
        return RunEventRecord(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            conversation_id=str(row["conversation_id"]),
            event_type=str(row["event_type"]),
            sequence=int(row["sequence"]),
            payload=dict(_from_json(row["payload_json"], {})),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
        )

    def create_run(self, payload: dict[str, Any]) -> RunRecord:
        run_id = payload.get("id") or f"fmr_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO furnacemind_runs (
                    id, conversation_id, owner_id, status, user_message_id,
                    assistant_message_id, progress, current_step, error_code,
                    error_message, warnings_json, artifacts_json, created_at, updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    payload["conversation_id"],
                    payload.get("owner_id"),
                    payload.get("status") or "pending",
                    payload.get("user_message_id"),
                    payload.get("assistant_message_id"),
                    payload.get("progress"),
                    payload.get("current_step"),
                    payload.get("error_code"),
                    payload.get("error_message"),
                    _json(payload.get("warnings") or []),
                    _json(payload.get("artifacts") or []),
                    now,
                    now,
                    payload.get("completed_at"),
                ),
            )
            row = connection.execute("SELECT * FROM furnacemind_runs WHERE id = ?", (run_id,)).fetchone()
            return self._run_from_row(row)

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM furnacemind_runs WHERE id = ?", (run_id,)).fetchone()
            return self._run_from_row(row) if row else None

    def update_run_status(self, run_id: str, **updates: Any) -> RunRecord | None:
        allowed = {
            "status",
            "assistant_message_id",
            "progress",
            "current_step",
            "error_code",
            "error_message",
            "warnings_json",
            "artifacts_json",
            "completed_at",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if "warnings" in updates:
            values["warnings_json"] = _json(updates["warnings"])
        if "artifacts" in updates:
            values["artifacts_json"] = _json(updates["artifacts"])
        values["updated_at"] = _iso()
        if values.get("status") in {"completed", "failed", "cancelled"} and not values.get("completed_at"):
            values["completed_at"] = _iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            result = connection.execute(
                f"UPDATE furnacemind_runs SET {assignments} WHERE id = ?",
                [*values.values(), run_id],
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute("SELECT * FROM furnacemind_runs WHERE id = ?", (run_id,)).fetchone()
            return self._run_from_row(row) if row else None

    def append_event(self, payload: dict[str, Any]) -> RunEventRecord:
        event_id = f"fme_{uuid4().hex}"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM furnacemind_run_events WHERE run_id = ?",
                (payload["run_id"],),
            ).fetchone()
            sequence = int(row[0])
            connection.execute(
                """
                INSERT INTO furnacemind_run_events (
                    id, run_id, conversation_id, event_type, sequence, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    payload["run_id"],
                    payload["conversation_id"],
                    payload["event_type"],
                    sequence,
                    _json(payload.get("payload") or {}),
                    _iso(),
                ),
            )
            row = connection.execute("SELECT * FROM furnacemind_run_events WHERE id = ?", (event_id,)).fetchone()
            return self._event_from_row(row)

    def list_events(self, run_id: str, *, limit: int = 500, offset: int = 0) -> list[RunEventRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM furnacemind_run_events
                WHERE run_id = ?
                ORDER BY sequence ASC
                LIMIT ? OFFSET ?
                """,
                (run_id, min(1000, max(1, int(limit))), max(0, int(offset))),
            ).fetchall()
            return [self._event_from_row(row) for row in rows]

    def trim_events(self, run_id: str, *, max_events: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM furnacemind_run_events
                WHERE run_id = ?
                  AND sequence NOT IN (
                    SELECT sequence FROM furnacemind_run_events
                    WHERE run_id = ?
                    ORDER BY sequence DESC
                    LIMIT ?
                  )
                """,
                (run_id, run_id, max(1, int(max_events))),
            )
