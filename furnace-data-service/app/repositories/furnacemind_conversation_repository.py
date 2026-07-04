"""SQLite repositories for FurnaceMind conversations, messages, and feedback."""

from __future__ import annotations

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


def default_furnacemind_db_url() -> str:
    path = runtime_path("furnacemind", "furnacemind.db", create_parent=True)
    return f"sqlite:///{path.as_posix()}"


def sqlite_url_to_path(database_url: str) -> Path | None:
    if database_url == "sqlite:///:memory:":
        return None
    if not database_url.startswith("sqlite:///"):
        raise ValueError("FurnaceMind repository currently supports SQLite URLs.")
    raw_path = database_url.replace("sqlite:///", "", 1)
    path = Path(raw_path)
    return path if path.is_absolute() else Path.cwd() / path


@dataclass(frozen=True)
class ConversationRecord:
    id: str
    title: str | None
    owner_id: str | None
    created_at: datetime
    updated_at: datetime
    archived: bool
    message_count: int
    last_message_at: datetime | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MessageRecord:
    id: str
    conversation_id: str
    role: str
    content: str
    status: str
    created_at: datetime
    updated_at: datetime | None
    run_id: str | None
    parent_message_id: str | None
    metadata: dict[str, Any]
    artifacts: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


@dataclass(frozen=True)
class MessageFeedbackRecord:
    id: str
    message_id: str
    conversation_id: str
    owner_id: str | None
    rating: int | None
    helpful: bool | None
    comment: str | None
    tags: list[str]
    created_at: datetime


class FurnaceMindConversationRepository:
    """Durable conversation/message repository with lazy SQLite setup."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or default_furnacemind_db_url()
        self._schema_ready = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db_path = sqlite_url_to_path(self.database_url)
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            ":memory:" if db_path is None else str(db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
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
            CREATE TABLE IF NOT EXISTS furnacemind_conversations (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                title TEXT,
                owner_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_fm_conversations_owner
                ON furnacemind_conversations(owner_id, archived, updated_at);

            CREATE TABLE IF NOT EXISTS furnacemind_messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                run_id TEXT,
                parent_message_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY(conversation_id) REFERENCES furnacemind_conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_fm_messages_conversation_created
                ON furnacemind_messages(conversation_id, created_at, seq);
            CREATE INDEX IF NOT EXISTS ix_fm_messages_run
                ON furnacemind_messages(run_id);

            CREATE TABLE IF NOT EXISTS furnacemind_message_feedback (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                message_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                owner_id TEXT,
                rating INTEGER,
                helpful INTEGER,
                comment TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY(message_id) REFERENCES furnacemind_messages(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_fm_feedback_message
                ON furnacemind_message_feedback(message_id, created_at);
            """
        )

    def db_path(self) -> Path | None:
        return sqlite_url_to_path(self.database_url)

    def _conversation_from_row(self, row: sqlite3.Row) -> ConversationRecord:
        return ConversationRecord(
            id=str(row["id"]),
            title=row["title"],
            owner_id=row["owner_id"],
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
            archived=bool(row["archived"]),
            message_count=int(row["message_count"] or 0),
            last_message_at=_parse_dt(row["last_message_at"]),
            metadata=dict(_from_json(row["metadata_json"], {})),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            status=str(row["status"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]),
            run_id=row["run_id"],
            parent_message_id=row["parent_message_id"],
            metadata=dict(_from_json(row["metadata_json"], {})),
            artifacts=list(_from_json(row["artifacts_json"], [])),
            warnings=list(_from_json(row["warnings_json"], [])),
        )

    @staticmethod
    def _conversation_select() -> str:
        return """
            SELECT
                c.*,
                (SELECT COUNT(*) FROM furnacemind_messages m WHERE m.conversation_id = c.id) AS message_count,
                (SELECT MAX(m.created_at) FROM furnacemind_messages m WHERE m.conversation_id = c.id) AS last_message_at
            FROM furnacemind_conversations c
        """

    def create_conversation(self, payload: dict[str, Any]) -> ConversationRecord:
        conversation_id = f"fmc_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO furnacemind_conversations (
                    id, title, owner_id, archived, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    conversation_id,
                    payload.get("title"),
                    payload.get("owner_id"),
                    _json(payload.get("metadata") or {}),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                self._conversation_select() + " WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
            return self._conversation_from_row(row)

    def list_conversations(self, filters: dict[str, Any]) -> tuple[list[ConversationRecord], int]:
        clauses: list[str] = []
        values: list[Any] = []
        owner_id = filters.get("owner_id")
        if owner_id is not None:
            clauses.append("c.owner_id = ?")
            values.append(str(owner_id))
        if filters.get("include_archived") is not True:
            clauses.append("c.archived = 0")
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = min(200, max(1, int(filters.get("limit") or 50)))
        offset = max(0, int(filters.get("offset") or 0))
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM furnacemind_conversations c{where_sql}",
                    values,
                ).fetchone()[0]
            )
            rows = connection.execute(
                self._conversation_select()
                + where_sql
                + " ORDER BY c.updated_at DESC, c.seq DESC LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
            return [self._conversation_from_row(row) for row in rows], total

    def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                self._conversation_select() + " WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
            return self._conversation_from_row(row) if row else None

    def update_conversation(self, conversation_id: str, changes: dict[str, Any]) -> ConversationRecord | None:
        allowed = {"title", "archived", "metadata_json"}
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return self.get_conversation(conversation_id)
        values["updated_at"] = _iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            result = connection.execute(
                f"UPDATE furnacemind_conversations SET {assignments} WHERE id = ?",
                [*values.values(), conversation_id],
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute(
                self._conversation_select() + " WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
            return self._conversation_from_row(row) if row else None

    def archive_conversation(self, conversation_id: str) -> ConversationRecord | None:
        return self.update_conversation(conversation_id, {"archived": 1})

    def create_message(self, payload: dict[str, Any]) -> MessageRecord:
        message_id = payload.get("id") or f"fmm_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO furnacemind_messages (
                    id, conversation_id, role, content, status, run_id,
                    parent_message_id, metadata_json, artifacts_json,
                    warnings_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    payload["conversation_id"],
                    payload["role"],
                    payload["content"],
                    payload.get("status") or "complete",
                    payload.get("run_id"),
                    payload.get("parent_message_id"),
                    _json(payload.get("metadata") or {}),
                    _json(payload.get("artifacts") or []),
                    _json(payload.get("warnings") or []),
                    now,
                    payload.get("updated_at"),
                ),
            )
            connection.execute(
                "UPDATE furnacemind_conversations SET updated_at = ? WHERE id = ?",
                (now, payload["conversation_id"]),
            )
            row = connection.execute(
                "SELECT * FROM furnacemind_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            return self._message_from_row(row)

    def list_messages(self, conversation_id: str, *, limit: int = 100, offset: int = 0) -> list[MessageRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM furnacemind_messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, seq ASC
                LIMIT ? OFFSET ?
                """,
                (conversation_id, min(500, max(1, int(limit))), max(0, int(offset))),
            ).fetchall()
            return [self._message_from_row(row) for row in rows]

    def get_message(self, message_id: str) -> MessageRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM furnacemind_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            return self._message_from_row(row) if row else None

    def update_message_status(self, message_id: str, status: str) -> MessageRecord | None:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE furnacemind_messages SET status = ?, updated_at = ? WHERE id = ?",
                (status, _iso(), message_id),
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute(
                "SELECT * FROM furnacemind_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            return self._message_from_row(row) if row else None

    def attach_artifacts(self, message_id: str, artifacts: list[dict[str, Any]]) -> MessageRecord | None:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE furnacemind_messages SET artifacts_json = ?, updated_at = ? WHERE id = ?",
                (_json(artifacts), _iso(), message_id),
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute(
                "SELECT * FROM furnacemind_messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            return self._message_from_row(row) if row else None

    def store_message_feedback(self, payload: dict[str, Any]) -> MessageFeedbackRecord:
        feedback_id = f"fmf_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO furnacemind_message_feedback (
                    id, message_id, conversation_id, owner_id, rating, helpful,
                    comment, tags_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    payload["message_id"],
                    payload["conversation_id"],
                    payload.get("owner_id"),
                    payload.get("rating"),
                    None if payload.get("helpful") is None else int(bool(payload.get("helpful"))),
                    payload.get("comment"),
                    _json(payload.get("tags") or []),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM furnacemind_message_feedback WHERE id = ?",
                (feedback_id,),
            ).fetchone()
            return MessageFeedbackRecord(
                id=str(row["id"]),
                message_id=str(row["message_id"]),
                conversation_id=str(row["conversation_id"]),
                owner_id=row["owner_id"],
                rating=row["rating"],
                helpful=None if row["helpful"] is None else bool(row["helpful"]),
                comment=row["comment"],
                tags=list(_from_json(row["tags_json"], [])),
                created_at=_parse_dt(row["created_at"]) or utc_now(),
            )
