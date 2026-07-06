"""SQLite-backed feedback repository for backend-owned ticket storage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from furnace_data.runtime_paths import get_feedback_db_path


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return default
    return loaded


def _to_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def default_feedback_db_url() -> str:
    """Return the default runtime-backed SQLite feedback metadata URL."""
    return f"sqlite:///{get_feedback_db_path().as_posix()}"


def _sqlite_url_to_path(database_url: str) -> Path | None:
    if database_url == "sqlite:///:memory:":
        return None
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Phase 6 feedback repository currently supports SQLite URLs.")
    raw_path = database_url.replace("sqlite:///", "", 1)
    path = Path(raw_path)
    return path if path.is_absolute() else Path.cwd() / path


@dataclass(frozen=True)
class FeedbackTicketRecord:
    id: str
    ticket_number: str
    title: str
    description: str
    category: str | None
    priority: str
    status: str
    page: str | None
    tags: list[str]
    created_by: str | None
    created_by_username: str | None
    assigned_to: str | None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    attachment_count: int
    comment_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FeedbackCommentRecord:
    id: str
    ticket_id: str
    body: str
    created_by: str | None
    created_by_username: str | None
    created_at: datetime


@dataclass(frozen=True)
class FeedbackAttachmentRecord:
    id: str
    ticket_id: str
    filename: str
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    created_by: str | None
    created_by_username: str | None
    created_at: datetime
    deleted_at: datetime | None


class FeedbackRepository:
    """Durable feedback repository with lazy SQLite initialization."""

    def __init__(self, database_url: str | None = None, *, ticket_prefix: str = "FB") -> None:
        self.database_url = database_url or default_feedback_db_url()
        self.ticket_prefix = ticket_prefix or "FB"
        self._schema_ready = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db_path = _sqlite_url_to_path(self.database_url)
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
        """Create backend feedback tables if they do not exist."""
        with self._connect():
            return

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback_tickets (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                ticket_number TEXT UNIQUE,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                page TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_by TEXT,
                created_by_username TEXT,
                assigned_to TEXT,
                resolution_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                closed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_status
                ON feedback_tickets(status);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_priority
                ON feedback_tickets(priority);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_created_by
                ON feedback_tickets(created_by);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_created_at
                ON feedback_tickets(created_at);

            CREATE TABLE IF NOT EXISTS feedback_comments (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                ticket_id TEXT NOT NULL,
                body TEXT NOT NULL,
                created_by TEXT,
                created_by_username TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(ticket_id) REFERENCES feedback_tickets(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_feedback_comments_ticket_created
                ON feedback_comments(ticket_id, created_at);

            CREATE TABLE IF NOT EXISTS feedback_attachments (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                ticket_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_by TEXT,
                created_by_username TEXT,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                FOREIGN KEY(ticket_id) REFERENCES feedback_tickets(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_feedback_attachments_ticket_created
                ON feedback_attachments(ticket_id, created_at);
            """
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _ticket_from_row(self, row: sqlite3.Row) -> FeedbackTicketRecord:
        return FeedbackTicketRecord(
            id=str(row["id"]),
            ticket_number=str(row["ticket_number"] or ""),
            title=str(row["title"]),
            description=str(row["description"]),
            category=row["category"],
            priority=str(row["priority"]),
            status=str(row["status"]),
            page=row["page"],
            tags=list(_from_json(row["tags_json"], [])),
            created_by=row["created_by"],
            created_by_username=row["created_by_username"],
            assigned_to=row["assigned_to"],
            created_at=self._parse_datetime(row["created_at"]) or utc_now(),
            updated_at=self._parse_datetime(row["updated_at"]) or utc_now(),
            closed_at=self._parse_datetime(row["closed_at"]),
            attachment_count=int(row["attachment_count"] or 0),
            comment_count=int(row["comment_count"] or 0),
            metadata=dict(_from_json(row["metadata_json"], {})),
        )

    def _comment_from_row(self, row: sqlite3.Row) -> FeedbackCommentRecord:
        return FeedbackCommentRecord(
            id=str(row["id"]),
            ticket_id=str(row["ticket_id"]),
            body=str(row["body"]),
            created_by=row["created_by"],
            created_by_username=row["created_by_username"],
            created_at=self._parse_datetime(row["created_at"]) or utc_now(),
        )

    def _attachment_from_row(self, row: sqlite3.Row) -> FeedbackAttachmentRecord:
        return FeedbackAttachmentRecord(
            id=str(row["id"]),
            ticket_id=str(row["ticket_id"]),
            filename=str(row["filename"]),
            original_filename=str(row["original_filename"]),
            stored_filename=str(row["stored_filename"]),
            content_type=str(row["content_type"]),
            size_bytes=int(row["size_bytes"]),
            created_by=row["created_by"],
            created_by_username=row["created_by_username"],
            created_at=self._parse_datetime(row["created_at"]) or utc_now(),
            deleted_at=self._parse_datetime(row["deleted_at"]),
        )

    def create_ticket(self, payload: dict[str, Any]) -> FeedbackTicketRecord:
        """Create and return one ticket."""
        ticket_id = f"fb_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback_tickets (
                    id, title, description, category, priority, status, page,
                    tags_json, metadata_json, created_by, created_by_username,
                    assigned_to, created_at, updated_at, closed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    payload["title"],
                    payload["description"],
                    payload.get("category"),
                    payload["priority"],
                    payload["status"],
                    payload.get("page"),
                    _to_json(payload.get("tags") or []),
                    _to_json(payload.get("metadata") or {}),
                    payload.get("created_by"),
                    payload.get("created_by_username"),
                    payload.get("assigned_to"),
                    now,
                    now,
                    payload.get("closed_at"),
                ),
            )
            ticket_number = f"{self.ticket_prefix}-{int(cursor.lastrowid):06d}"
            connection.execute(
                "UPDATE feedback_tickets SET ticket_number = ? WHERE id = ?",
                (ticket_number, ticket_id),
            )
            row = self._get_ticket_row(connection, ticket_id)
            if row is None:
                raise RuntimeError("Created feedback ticket could not be loaded.")
            return self._ticket_from_row(row)

    def _ticket_select(self) -> str:
        return """
            SELECT
                t.*,
                (
                    SELECT COUNT(*)
                    FROM feedback_attachments a
                    WHERE a.ticket_id = t.id AND a.deleted_at IS NULL
                ) AS attachment_count,
                (
                    SELECT COUNT(*)
                    FROM feedback_comments c
                    WHERE c.ticket_id = t.id
                ) AS comment_count
            FROM feedback_tickets t
        """

    def _get_ticket_row(self, connection: sqlite3.Connection, ticket_id: str) -> sqlite3.Row | None:
        return connection.execute(
            self._ticket_select() + " WHERE t.id = ?",
            (ticket_id,),
        ).fetchone()

    def get_ticket(self, ticket_id: str) -> FeedbackTicketRecord | None:
        """Return one ticket by id or ticket number."""
        with self._connect() as connection:
            row = connection.execute(
                self._ticket_select() + " WHERE t.id = ? OR t.ticket_number = ?",
                (ticket_id, ticket_id),
            ).fetchone()
            return self._ticket_from_row(row) if row else None

    def list_tickets(self, filters: dict[str, Any]) -> tuple[list[FeedbackTicketRecord], int]:
        """Return filtered tickets and total count."""
        clauses: list[str] = []
        values: list[Any] = []
        for key, column in (
            ("status", "t.status"),
            ("priority", "t.priority"),
            ("category", "t.category"),
            ("created_by", "t.created_by"),
            ("assigned_to", "t.assigned_to"),
        ):
            value = filters.get(key)
            if value:
                clauses.append(f"{column} = ?")
                values.append(str(value))

        search = str(filters.get("search") or "").strip()
        if search:
            token = f"%{search.lower()}%"
            clauses.append(
                "("
                "LOWER(t.ticket_number) LIKE ? OR LOWER(t.title) LIKE ? OR "
                "LOWER(t.description) LIKE ? OR LOWER(COALESCE(t.page, '')) LIKE ? OR "
                "LOWER(COALESCE(t.created_by_username, '')) LIKE ?"
                ")"
            )
            values.extend([token] * 5)

        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = int(filters.get("limit") or 50)
        offset = int(filters.get("offset") or 0)

        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM feedback_tickets t{where_sql}",
                    values,
                ).fetchone()[0]
            )
            rows = connection.execute(
                self._ticket_select()
                + where_sql
                + " ORDER BY t.updated_at DESC, t.seq DESC LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
            return [self._ticket_from_row(row) for row in rows], total

    def update_ticket(self, ticket_id: str, changes: dict[str, Any]) -> FeedbackTicketRecord | None:
        """Patch allowed ticket fields."""
        allowed = {
            "title",
            "description",
            "category",
            "priority",
            "status",
            "assigned_to",
            "resolution_notes",
            "tags_json",
            "metadata_json",
            "closed_at",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return self.get_ticket(ticket_id)

        values["updated_at"] = _iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        params = [*values.values(), ticket_id, ticket_id]
        with self._connect() as connection:
            result = connection.execute(
                f"UPDATE feedback_tickets SET {assignments} WHERE id = ? OR ticket_number = ?",
                params,
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute(
                self._ticket_select() + " WHERE t.id = ? OR t.ticket_number = ?",
                (ticket_id, ticket_id),
            ).fetchone()
            return self._ticket_from_row(row) if row else None

    def add_comment(self, payload: dict[str, Any]) -> FeedbackCommentRecord:
        """Create a ticket comment."""
        comment_id = f"fbc_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_comments (
                    id, ticket_id, body, created_by, created_by_username, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    comment_id,
                    payload["ticket_id"],
                    payload["body"],
                    payload.get("created_by"),
                    payload.get("created_by_username"),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM feedback_comments WHERE id = ?",
                (comment_id,),
            ).fetchone()
            return self._comment_from_row(row)

    def list_comments(self, ticket_id: str) -> list[FeedbackCommentRecord]:
        """Return comments for one ticket."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM feedback_comments
                WHERE ticket_id = ?
                ORDER BY created_at ASC, seq ASC
                """,
                (ticket_id,),
            ).fetchall()
            return [self._comment_from_row(row) for row in rows]

    def add_attachment_metadata(self, payload: dict[str, Any]) -> FeedbackAttachmentRecord:
        """Create attachment metadata."""
        attachment_id = payload.get("id") or f"fba_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_attachments (
                    id, ticket_id, filename, original_filename, stored_filename,
                    content_type, size_bytes, created_by, created_by_username, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment_id,
                    payload["ticket_id"],
                    payload["filename"],
                    payload["original_filename"],
                    payload["stored_filename"],
                    payload["content_type"],
                    int(payload["size_bytes"]),
                    payload.get("created_by"),
                    payload.get("created_by_username"),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM feedback_attachments WHERE id = ?",
                (attachment_id,),
            ).fetchone()
            return self._attachment_from_row(row)

    def list_attachments(self, ticket_id: str) -> list[FeedbackAttachmentRecord]:
        """List active attachments for one ticket."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM feedback_attachments
                WHERE ticket_id = ? AND deleted_at IS NULL
                ORDER BY created_at DESC, seq DESC
                """,
                (ticket_id,),
            ).fetchall()
            return [self._attachment_from_row(row) for row in rows]

    def get_attachment(self, attachment_id: str) -> FeedbackAttachmentRecord | None:
        """Return active attachment metadata."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_attachments WHERE id = ? AND deleted_at IS NULL",
                (attachment_id,),
            ).fetchone()
            return self._attachment_from_row(row) if row else None

    def delete_attachment_metadata(self, attachment_id: str) -> FeedbackAttachmentRecord | None:
        """Mark attachment metadata as deleted and return previous row."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_attachments WHERE id = ? AND deleted_at IS NULL",
                (attachment_id,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE feedback_attachments SET deleted_at = ? WHERE id = ?",
                (_iso(), attachment_id),
            )
            return self._attachment_from_row(row)
