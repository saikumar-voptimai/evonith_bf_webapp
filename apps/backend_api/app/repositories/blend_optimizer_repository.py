"""SQLite persistence for Blend Optimizer contexts, runs, events, and preferences."""

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

from furnace_data.runtime_paths import runtime_path


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


def default_blend_optimizer_db_url() -> str:
    path = runtime_path("blend_optimizer", "blend_optimizer.db", create_parent=True)
    return f"sqlite:///{path.as_posix()}"


def sqlite_url_to_path(database_url: str) -> Path | None:
    if database_url in {"", ":memory:", "sqlite:///:memory:"}:
        return None
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Blend Optimizer repository currently supports SQLite URLs.")
    raw_path = database_url[len(prefix) :]
    path = Path(raw_path)
    return path if path.is_absolute() else Path.cwd() / path


def idempotency_hash(idempotency_key: str | None) -> str | None:
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BlendOptimizerContextRecord:
    id: str
    owner_id: str | None
    version: str
    fingerprint: str
    status: str
    request: dict[str, Any]
    snapshot: dict[str, Any]
    diagnostics: dict[str, Any]
    warnings: list[dict[str, Any]]
    idempotency_key_hash: str | None
    request_fingerprint: str | None
    created_at: datetime
    expires_at: datetime | None


@dataclass(frozen=True)
class BlendOptimizerPreferenceRecord:
    owner_id: str
    version: int
    preferences: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BlendOptimizerRunRecord:
    id: str
    owner_id: str | None
    mode: str
    context_id: str
    context_version: str
    status: str
    progress: float | None
    current_step: str | None
    request: dict[str, Any]
    result: dict[str, Any]
    warnings: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    error_code: str | None
    error_message: str | None
    idempotency_key_hash: str | None
    request_fingerprint: str | None
    created_at: datetime
    updated_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class BlendOptimizerRunEventRecord:
    id: str
    run_id: str
    owner_id: str | None
    event_type: str
    sequence: int
    payload: dict[str, Any]
    created_at: datetime


class BlendOptimizerRepository:
    """Repository-owned migrations for BMO durable resources."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or default_blend_optimizer_db_url()
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

    def ensure_schema(self) -> None:
        with self._connect():
            return

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS bmo_contexts (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                owner_id TEXT,
                version TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL DEFAULT '{}',
                snapshot_json TEXT NOT NULL DEFAULT '{}',
                diagnostics_json TEXT NOT NULL DEFAULT '{}',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                idempotency_key_hash TEXT,
                request_fingerprint TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_bmo_context_owner_idempotency
                ON bmo_contexts(owner_id, idempotency_key_hash)
                WHERE idempotency_key_hash IS NOT NULL;
            CREATE INDEX IF NOT EXISTS ix_bmo_context_owner_created
                ON bmo_contexts(owner_id, created_at);

            CREATE TABLE IF NOT EXISTS bmo_preferences (
                owner_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                preferences_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bmo_runs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                owner_id TEXT,
                mode TEXT NOT NULL,
                context_id TEXT NOT NULL,
                context_version TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL,
                current_step TEXT,
                request_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                error_code TEXT,
                error_message TEXT,
                idempotency_key_hash TEXT,
                request_fingerprint TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                completed_at TEXT,
                FOREIGN KEY(context_id) REFERENCES bmo_contexts(id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_bmo_run_owner_idempotency
                ON bmo_runs(owner_id, idempotency_key_hash)
                WHERE idempotency_key_hash IS NOT NULL;
            CREATE INDEX IF NOT EXISTS ix_bmo_run_owner_created
                ON bmo_runs(owner_id, created_at);

            CREATE TABLE IF NOT EXISTS bmo_run_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL,
                owner_id TEXT,
                event_type TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES bmo_runs(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_bmo_run_events_run_sequence
                ON bmo_run_events(run_id, sequence);
            """
        )

    @staticmethod
    def _context(row: sqlite3.Row) -> BlendOptimizerContextRecord:
        return BlendOptimizerContextRecord(
            id=str(row["id"]),
            owner_id=row["owner_id"],
            version=str(row["version"]),
            fingerprint=str(row["fingerprint"]),
            status=str(row["status"]),
            request=dict(_from_json(row["request_json"], {})),
            snapshot=dict(_from_json(row["snapshot_json"], {})),
            diagnostics=dict(_from_json(row["diagnostics_json"], {})),
            warnings=list(_from_json(row["warnings_json"], [])),
            idempotency_key_hash=row["idempotency_key_hash"],
            request_fingerprint=row["request_fingerprint"],
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            expires_at=_parse_dt(row["expires_at"]),
        )

    @staticmethod
    def _preference(row: sqlite3.Row) -> BlendOptimizerPreferenceRecord:
        return BlendOptimizerPreferenceRecord(
            owner_id=str(row["owner_id"]),
            version=int(row["version"]),
            preferences=dict(_from_json(row["preferences_json"], {})),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
        )

    @staticmethod
    def _run(row: sqlite3.Row) -> BlendOptimizerRunRecord:
        return BlendOptimizerRunRecord(
            id=str(row["id"]),
            owner_id=row["owner_id"],
            mode=str(row["mode"]),
            context_id=str(row["context_id"]),
            context_version=str(row["context_version"]),
            status=str(row["status"]),
            progress=row["progress"],
            current_step=row["current_step"],
            request=dict(_from_json(row["request_json"], {})),
            result=dict(_from_json(row["result_json"], {})),
            warnings=list(_from_json(row["warnings_json"], [])),
            artifacts=list(_from_json(row["artifacts_json"], [])),
            error_code=row["error_code"],
            error_message=row["error_message"],
            idempotency_key_hash=row["idempotency_key_hash"],
            request_fingerprint=row["request_fingerprint"],
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]),
            completed_at=_parse_dt(row["completed_at"]),
        )

    @staticmethod
    def _event(row: sqlite3.Row) -> BlendOptimizerRunEventRecord:
        return BlendOptimizerRunEventRecord(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            owner_id=row["owner_id"],
            event_type=str(row["event_type"]),
            sequence=int(row["sequence"]),
            payload=dict(_from_json(row["payload_json"], {})),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
        )

    def create_context(self, payload: dict[str, Any]) -> BlendOptimizerContextRecord:
        context_id = payload.get("id") or f"bmc_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO bmo_contexts (
                    id, owner_id, version, fingerprint, status, request_json,
                    snapshot_json, diagnostics_json, warnings_json,
                    idempotency_key_hash, request_fingerprint, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context_id,
                    payload.get("owner_id"),
                    payload["version"],
                    payload["fingerprint"],
                    payload.get("status") or "available",
                    _json(payload.get("request") or {}),
                    _json(payload.get("snapshot") or {}),
                    _json(payload.get("diagnostics") or {}),
                    _json(payload.get("warnings") or []),
                    payload.get("idempotency_key_hash"),
                    payload.get("request_fingerprint"),
                    _iso(),
                    payload.get("expires_at"),
                ),
            )
            row = connection.execute("SELECT * FROM bmo_contexts WHERE id = ?", (context_id,)).fetchone()
            return self._context(row)

    def get_context(self, context_id: str) -> BlendOptimizerContextRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM bmo_contexts WHERE id = ?", (context_id,)).fetchone()
            return self._context(row) if row else None

    def find_context_by_idempotency(self, *, owner_id: str, key_hash: str) -> BlendOptimizerContextRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM bmo_contexts WHERE owner_id = ? AND idempotency_key_hash = ?",
                (owner_id, key_hash),
            ).fetchone()
            return self._context(row) if row else None

    def get_preferences(self, owner_id: str) -> BlendOptimizerPreferenceRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM bmo_preferences WHERE owner_id = ?", (owner_id,)).fetchone()
            return self._preference(row) if row else None

    def upsert_preferences(self, *, owner_id: str, preferences: dict[str, Any], expected_version: int | None) -> BlendOptimizerPreferenceRecord:
        now = _iso()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM bmo_preferences WHERE owner_id = ?", (owner_id,)).fetchone()
            if row is None:
                if expected_version not in {None, 0}:
                    raise ValueError("version_conflict")
                connection.execute(
                    "INSERT INTO bmo_preferences (owner_id, version, preferences_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (owner_id, 1, _json(preferences), now, now),
                )
            else:
                current = int(row["version"])
                if expected_version is not None and expected_version != current:
                    raise ValueError("version_conflict")
                connection.execute(
                    "UPDATE bmo_preferences SET version = ?, preferences_json = ?, updated_at = ? WHERE owner_id = ?",
                    (current + 1, _json(preferences), now, owner_id),
                )
            row = connection.execute("SELECT * FROM bmo_preferences WHERE owner_id = ?", (owner_id,)).fetchone()
            return self._preference(row)

    def create_run(self, payload: dict[str, Any]) -> BlendOptimizerRunRecord:
        run_id = payload.get("id") or f"bmr_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO bmo_runs (
                    id, owner_id, mode, context_id, context_version, status,
                    progress, current_step, request_json, result_json,
                    warnings_json, artifacts_json, error_code, error_message,
                    idempotency_key_hash, request_fingerprint, created_at,
                    updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    payload.get("owner_id"),
                    payload["mode"],
                    payload["context_id"],
                    payload["context_version"],
                    payload.get("status") or "queued",
                    payload.get("progress"),
                    payload.get("current_step"),
                    _json(payload.get("request") or {}),
                    _json(payload.get("result") or {}),
                    _json(payload.get("warnings") or []),
                    _json(payload.get("artifacts") or []),
                    payload.get("error_code"),
                    payload.get("error_message"),
                    payload.get("idempotency_key_hash"),
                    payload.get("request_fingerprint"),
                    now,
                    now,
                    payload.get("completed_at"),
                ),
            )
            row = connection.execute("SELECT * FROM bmo_runs WHERE id = ?", (run_id,)).fetchone()
            return self._run(row)

    def find_run_by_idempotency(self, *, owner_id: str, key_hash: str) -> BlendOptimizerRunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM bmo_runs WHERE owner_id = ? AND idempotency_key_hash = ?",
                (owner_id, key_hash),
            ).fetchone()
            return self._run(row) if row else None

    def get_run(self, run_id: str) -> BlendOptimizerRunRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM bmo_runs WHERE id = ?", (run_id,)).fetchone()
            return self._run(row) if row else None

    def update_run(self, run_id: str, **updates: Any) -> BlendOptimizerRunRecord | None:
        values: dict[str, Any] = {}
        for key in ("status", "progress", "current_step", "error_code", "error_message"):
            if key in updates:
                values[key] = updates[key]
        if "request" in updates:
            values["request_json"] = _json(updates["request"])
        if "result" in updates:
            values["result_json"] = _json(updates["result"])
        if "warnings" in updates:
            values["warnings_json"] = _json(updates["warnings"])
        if "artifacts" in updates:
            values["artifacts_json"] = _json(updates["artifacts"])
        values["updated_at"] = _iso()
        if values.get("status") in {"completed", "failed", "cancelled", "timed_out"}:
            values["completed_at"] = _iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            result = connection.execute(f"UPDATE bmo_runs SET {assignments} WHERE id = ?", [*values.values(), run_id])
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute("SELECT * FROM bmo_runs WHERE id = ?", (run_id,)).fetchone()
            return self._run(row) if row else None

    def append_event(self, *, run_id: str, owner_id: str | None, event_type: str, payload: dict[str, Any] | None = None) -> BlendOptimizerRunEventRecord:
        event_id = f"bmre_{uuid4().hex}"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM bmo_run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row[0])
            connection.execute(
                """
                INSERT INTO bmo_run_events (id, run_id, owner_id, event_type, sequence, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, run_id, owner_id, event_type, sequence, _json(payload or {}), _iso()),
            )
            row = connection.execute("SELECT * FROM bmo_run_events WHERE id = ?", (event_id,)).fetchone()
            return self._event(row)

    def list_events(self, run_id: str, *, after: int | None = None, limit: int = 500) -> list[BlendOptimizerRunEventRecord]:
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if after is not None:
            clauses.append("sequence > ?")
            values.append(int(after))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM bmo_run_events WHERE {' AND '.join(clauses)} ORDER BY sequence ASC LIMIT ?",
                [*values, min(1000, max(1, int(limit)))],
            ).fetchall()
            return [self._event(row) for row in rows]
