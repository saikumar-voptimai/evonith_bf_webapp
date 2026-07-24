
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

from furnace_data.app_catalog import canonical_page_id, page_label
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


def _as_list(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def default_feedback_db_url() -> str:
    """Return the default runtime-backed SQLite feedback metadata URL."""
    return f"sqlite:///{get_feedback_db_path().as_posix()}"


def _sqlite_url_to_path(database_url: str) -> Path | None:
    if database_url == "sqlite:///:memory:":
        return None
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Feedback repository currently supports SQLite URLs.")
    raw_path = database_url.replace("sqlite:///", "", 1)
    path = Path(raw_path)
    return path if path.is_absolute() else Path.cwd() / path


@dataclass(frozen=True)
class FeedbackTicketRecord:
    id: str
    ticket_number: str
    version: int
    title: str
    description: str
    ideal_closure: str | None
    category: str | None
    priority: str
    status: str
    page_id: str | None
    page: str | None
    tags: list[str]
    created_by: str | None
    created_by_username: str | None
    updated_by: str | None
    updated_by_username: str | None
    assigned_to: str | None
    resolution_notes: str | None
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    resolved_at: datetime | None
    closed_at: datetime | None
    deleted_at: datetime | None
    attachment_count: int
    comment_count: int
    event_count: int
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
    checksum_sha256: str | None
    storage_status: str
    created_by: str | None
    created_by_username: str | None
    created_at: datetime
    deleted_at: datetime | None


@dataclass(frozen=True)
class FeedbackEventRecord:
    id: str
    ticket_id: str
    event_type: str
    sequence: int
    actor_user_id: str | None
    actor_username: str | None
    old_status: str | None
    new_status: str | None
    note: str | None
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class FeedbackIdempotencyRecord:
    owner_key: str
    operation: str
    key_hash: str
    request_fingerprint: str
    resource_type: str
    resource_id: str
    created_at: datetime


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
    def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
        return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _add_column_if_missing(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        if column not in FeedbackRepository._column_names(connection, table):
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback_tickets (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                ticket_number TEXT UNIQUE,
                version INTEGER NOT NULL DEFAULT 1,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                ideal_closure TEXT,
                category TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                page_id TEXT,
                page TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_by TEXT,
                created_by_username TEXT,
                updated_by TEXT,
                updated_by_username TEXT,
                assigned_to TEXT,
                resolution_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_activity_at TEXT,
                resolved_at TEXT,
                closed_at TEXT,
                deleted_at TEXT,
                delete_reason TEXT
            );
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
            CREATE TABLE IF NOT EXISTS feedback_attachments (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                ticket_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                checksum_sha256 TEXT,
                storage_status TEXT NOT NULL DEFAULT 'stored',
                created_by TEXT,
                created_by_username TEXT,
                created_at TEXT NOT NULL,
                deleted_at TEXT,
                FOREIGN KEY(ticket_id) REFERENCES feedback_tickets(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS feedback_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                ticket_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor_user_id TEXT,
                actor_username TEXT,
                old_status TEXT,
                new_status TEXT,
                note TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                sequence INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(ticket_id) REFERENCES feedback_tickets(id) ON DELETE CASCADE,
                UNIQUE(ticket_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS feedback_idempotency_keys (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_key TEXT NOT NULL,
                operation TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(owner_key, operation, key_hash)
            );
            """
        )
        for column, definition in (
            ("version", "version INTEGER NOT NULL DEFAULT 1"),
            ("ideal_closure", "ideal_closure TEXT"),
            ("page_id", "page_id TEXT"),
            ("updated_by", "updated_by TEXT"),
            ("updated_by_username", "updated_by_username TEXT"),
            ("last_activity_at", "last_activity_at TEXT"),
            ("resolved_at", "resolved_at TEXT"),
            ("deleted_at", "deleted_at TEXT"),
            ("delete_reason", "delete_reason TEXT"),
        ):
            FeedbackRepository._add_column_if_missing(connection, "feedback_tickets", column, definition)
        for column, definition in (
            ("checksum_sha256", "checksum_sha256 TEXT"),
            ("storage_status", "storage_status TEXT NOT NULL DEFAULT 'stored'"),
        ):
            FeedbackRepository._add_column_if_missing(connection, "feedback_attachments", column, definition)
        connection.execute("UPDATE feedback_tickets SET last_activity_at = COALESCE(last_activity_at, updated_at, created_at)")
        connection.execute("UPDATE feedback_tickets SET page_id = COALESCE(page_id, page) WHERE page_id IS NULL")
        for row in connection.execute("SELECT id, page_id, page, category FROM feedback_tickets"):
            stable_page_id = (
                canonical_page_id(row["page_id"])
                or canonical_page_id(row["page"])
                or canonical_page_id(row["category"])
            )
            if stable_page_id and stable_page_id != row["page_id"]:
                connection.execute(
                    "UPDATE feedback_tickets SET page_id = ?, page = ? WHERE id = ?",
                    (stable_page_id, page_label(stable_page_id), row["id"]),
                )
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_status ON feedback_tickets(status);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_priority ON feedback_tickets(priority);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_page_id ON feedback_tickets(page_id);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_created_by ON feedback_tickets(created_by);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_created_at ON feedback_tickets(created_at);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_last_activity ON feedback_tickets(last_activity_at, seq);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_number ON feedback_tickets(ticket_number);
            CREATE INDEX IF NOT EXISTS ix_feedback_tickets_deleted ON feedback_tickets(deleted_at);
            CREATE INDEX IF NOT EXISTS ix_feedback_comments_ticket_created ON feedback_comments(ticket_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_feedback_attachments_ticket_created ON feedback_attachments(ticket_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_feedback_events_ticket_sequence ON feedback_events(ticket_id, sequence);
            CREATE INDEX IF NOT EXISTS ix_feedback_events_type ON feedback_events(event_type);
            CREATE INDEX IF NOT EXISTS ix_feedback_idempotency_lookup ON feedback_idempotency_keys(owner_key, operation, key_hash);
            """
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _ticket_from_row(self, row: sqlite3.Row) -> FeedbackTicketRecord:
        raw_page_id = row["page_id"] if "page_id" in row.keys() else None
        page_id = canonical_page_id(raw_page_id) or canonical_page_id(row["page"])
        label = page_label(page_id) if page_id else row["page"]
        return FeedbackTicketRecord(
            id=str(row["id"]),
            ticket_number=str(row["ticket_number"] or ""),
            version=int(row["version"] or 1),
            title=str(row["title"]),
            description=str(row["description"]),
            ideal_closure=row["ideal_closure"],
            category=row["category"],
            priority=str(row["priority"]),
            status=str(row["status"]),
            page_id=page_id,
            page=label,
            tags=list(_from_json(row["tags_json"], [])),
            created_by=row["created_by"],
            created_by_username=row["created_by_username"],
            updated_by=row["updated_by"],
            updated_by_username=row["updated_by_username"],
            assigned_to=row["assigned_to"],
            resolution_notes=row["resolution_notes"],
            created_at=self._parse_datetime(row["created_at"]) or utc_now(),
            updated_at=self._parse_datetime(row["updated_at"]) or utc_now(),
            last_activity_at=self._parse_datetime(row["last_activity_at"]) or self._parse_datetime(row["updated_at"]) or utc_now(),
            resolved_at=self._parse_datetime(row["resolved_at"]),
            closed_at=self._parse_datetime(row["closed_at"]),
            deleted_at=self._parse_datetime(row["deleted_at"]),
            attachment_count=int(row["attachment_count"] or 0),
            comment_count=int(row["comment_count"] or 0),
            event_count=int(row["event_count"] or 0),
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
            checksum_sha256=row["checksum_sha256"],
            storage_status=str(row["storage_status"] or "stored"),
            created_by=row["created_by"],
            created_by_username=row["created_by_username"],
            created_at=self._parse_datetime(row["created_at"]) or utc_now(),
            deleted_at=self._parse_datetime(row["deleted_at"]),
        )

    def _event_from_row(self, row: sqlite3.Row) -> FeedbackEventRecord:
        return FeedbackEventRecord(
            id=str(row["id"]),
            ticket_id=str(row["ticket_id"]),
            event_type=str(row["event_type"]),
            sequence=int(row["sequence"]),
            actor_user_id=row["actor_user_id"],
            actor_username=row["actor_username"],
            old_status=row["old_status"],
            new_status=row["new_status"],
            note=row["note"],
            payload=dict(_from_json(row["payload_json"], {})),
            created_at=self._parse_datetime(row["created_at"]) or utc_now(),
        )

    @staticmethod
    def _idempotency_from_row(row: sqlite3.Row) -> FeedbackIdempotencyRecord:
        return FeedbackIdempotencyRecord(
            owner_key=str(row["owner_key"]),
            operation=str(row["operation"]),
            key_hash=str(row["key_hash"]),
            request_fingerprint=str(row["request_fingerprint"]),
            resource_type=str(row["resource_type"]),
            resource_id=str(row["resource_id"]),
            created_at=FeedbackRepository._parse_datetime(row["created_at"]) or utc_now(),
        )

    def _ticket_select(self) -> str:
        return """
            SELECT t.*,
                (SELECT COUNT(*) FROM feedback_attachments a WHERE a.ticket_id = t.id AND a.deleted_at IS NULL) AS attachment_count,
                (SELECT COUNT(*) FROM feedback_comments c WHERE c.ticket_id = t.id) AS comment_count,
                (SELECT COUNT(*) FROM feedback_events e WHERE e.ticket_id = t.id) AS event_count
            FROM feedback_tickets t
        """

    def _get_ticket_row(self, connection: sqlite3.Connection, ticket_id: str, *, include_deleted: bool = False) -> sqlite3.Row | None:
        deleted_clause = "" if include_deleted else " AND t.deleted_at IS NULL"
        return connection.execute(
            self._ticket_select() + f" WHERE (t.id = ? OR t.ticket_number = ?){deleted_clause}",
            (ticket_id, ticket_id),
        ).fetchone()

    def create_ticket(self, payload: dict[str, Any]) -> FeedbackTicketRecord:
        """Create and return one ticket."""
        ticket_id = payload.get("id") or f"fb_{uuid4().hex}"
        now = _iso()
        page_id = canonical_page_id(payload.get("page_id") or payload.get("page"))
        page = page_label(page_id) if page_id else payload.get("page")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback_tickets (
                    id, version, title, description, ideal_closure, category,
                    priority, status, page_id, page, tags_json, metadata_json,
                    created_by, created_by_username, updated_by, updated_by_username,
                    assigned_to, resolution_notes, created_at, updated_at,
                    last_activity_at, resolved_at, closed_at, deleted_at
                )
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    ticket_id,
                    payload["title"],
                    payload["description"],
                    payload.get("ideal_closure"),
                    payload.get("category") or page,
                    payload["priority"],
                    payload["status"],
                    page_id,
                    page,
                    _to_json(payload.get("tags") or []),
                    _to_json(payload.get("metadata") or {}),
                    payload.get("created_by"),
                    payload.get("created_by_username"),
                    payload.get("updated_by") or payload.get("created_by"),
                    payload.get("updated_by_username") or payload.get("created_by_username"),
                    payload.get("assigned_to"),
                    payload.get("resolution_notes"),
                    payload.get("created_at") or now,
                    payload.get("updated_at") or now,
                    payload.get("last_activity_at") or now,
                    payload.get("resolved_at"),
                    payload.get("closed_at"),
                ),
            )
            ticket_number = payload.get("ticket_number") or f"{self.ticket_prefix}-{int(cursor.lastrowid):06d}"
            connection.execute("UPDATE feedback_tickets SET ticket_number = ? WHERE id = ?", (ticket_number, ticket_id))
            row = self._get_ticket_row(connection, ticket_id)
            if row is None:
                raise RuntimeError("Created feedback ticket could not be loaded.")
            return self._ticket_from_row(row)

    def get_ticket(self, ticket_id: str, *, include_deleted: bool = False) -> FeedbackTicketRecord | None:
        """Return one ticket by id or ticket number."""
        with self._connect() as connection:
            row = self._get_ticket_row(connection, ticket_id, include_deleted=include_deleted)
            return self._ticket_from_row(row) if row else None

    @staticmethod
    def _append_multi_filter(clauses: list[str], values: list[Any], column: str, candidates: Any) -> None:
        items = _as_list(candidates)
        if not items:
            return
        placeholders = ", ".join("?" for _ in items)
        clauses.append(f"{column} IN ({placeholders})")
        values.extend(items)

    @staticmethod
    def _where(filters: dict[str, Any], *, include_deleted: bool = False) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if not include_deleted:
            clauses.append("t.deleted_at IS NULL")
        FeedbackRepository._append_multi_filter(clauses, values, "t.status", filters.get("status"))
        FeedbackRepository._append_multi_filter(clauses, values, "t.priority", filters.get("priority"))
        FeedbackRepository._append_multi_filter(clauses, values, "t.page_id", filters.get("page_id"))
        FeedbackRepository._append_multi_filter(clauses, values, "t.category", filters.get("category"))
        for key, column in (("created_by", "t.created_by"), ("reporter_user_id", "t.created_by"), ("assigned_to", "t.assigned_to")):
            value = filters.get(key)
            if value:
                clauses.append(f"{column} = ?")
                values.append(str(value))
        if filters.get("created_from"):
            clauses.append("t.created_at >= ?")
            values.append(str(filters["created_from"]))
        if filters.get("created_to"):
            clauses.append("t.created_at <= ?")
            values.append(str(filters["created_to"]))
        search = str(filters.get("search") or "").strip()
        if search:
            token = f"%{search.lower()}%"
            clauses.append(
                "(LOWER(t.ticket_number) LIKE ? OR LOWER(t.title) LIKE ? OR "
                "LOWER(t.description) LIKE ? OR LOWER(COALESCE(t.page, '')) LIKE ? OR "
                "LOWER(COALESCE(t.created_by_username, '')) LIKE ?)"
            )
            values.extend([token] * 5)
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where_sql, values

    def list_tickets(self, filters: dict[str, Any]) -> tuple[list[FeedbackTicketRecord], int]:
        """Return filtered tickets and total count."""
        where_sql, values = self._where(filters)
        limit = int(filters.get("limit") or 50)
        offset = int(filters.get("offset") or 0)
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM feedback_tickets t{where_sql}", values).fetchone()[0])
            rows = connection.execute(
                self._ticket_select()
                + where_sql
                + " ORDER BY COALESCE(t.last_activity_at, t.updated_at, t.created_at) DESC, t.seq DESC LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
            return [self._ticket_from_row(row) for row in rows], total

    def summary(self, filters: dict[str, Any], *, include_reporters: bool) -> dict[str, Any]:
        """Return aggregate feedback counts over the filtered scope."""
        where_sql, values = self._where(filters)
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM feedback_tickets t{where_sql}", values).fetchone()[0])
            status_rows = connection.execute(
                f"SELECT t.status, COUNT(*) AS count FROM feedback_tickets t{where_sql} GROUP BY t.status",
                values,
            ).fetchall()
            priority_rows = connection.execute(
                f"SELECT t.priority, COUNT(*) AS count FROM feedback_tickets t{where_sql} GROUP BY t.priority",
                values,
            ).fetchall()
            page_rows = connection.execute(
                f"SELECT t.page_id, t.page, COUNT(*) AS count FROM feedback_tickets t{where_sql} GROUP BY t.page_id, t.page ORDER BY t.page",
                values,
            ).fetchall()
            reporter_rows: list[sqlite3.Row] = []
            if include_reporters:
                reporter_rows = connection.execute(
                    f"SELECT t.created_by, t.created_by_username, COUNT(*) AS count FROM feedback_tickets t{where_sql} GROUP BY t.created_by, t.created_by_username ORDER BY t.created_by_username",
                    values,
                ).fetchall()
        status_counts = {str(row["status"]): int(row["count"]) for row in status_rows}
        priority_counts = {str(row["priority"]): int(row["count"]) for row in priority_rows}
        return {
            "total": total,
            "counts_by_status": status_counts,
            "counts_by_priority": priority_counts,
            "resolved_or_closed_count": sum(status_counts.get(item, 0) for item in ("resolved", "closed")),
            "dependency_conflict_count": status_counts.get("dependency_conflict", 0),
            "rejected_count": status_counts.get("rejected", 0),
            "high_or_critical_count": sum(priority_counts.get(item, 0) for item in ("high", "critical")),
            "facets": {
                "pages": [
                    {
                        "page_id": canonical_page_id(row["page_id"]) or row["page_id"],
                        "label": page_label(canonical_page_id(row["page_id"])) or row["page"] or row["page_id"],
                        "count": int(row["count"]),
                    }
                    for row in page_rows
                    if row["page_id"] or row["page"]
                ],
                "reporters": [
                    {"user_id": row["created_by"], "username": row["created_by_username"], "count": int(row["count"])}
                    for row in reporter_rows
                ],
            },
        }

    def update_ticket(self, ticket_id: str, changes: dict[str, Any], *, expected_version: int | None = None) -> FeedbackTicketRecord | None:
        """Patch allowed ticket fields and increment version."""
        allowed = {
            "title", "description", "ideal_closure", "category", "priority", "status",
            "page_id", "page", "assigned_to", "resolution_notes", "tags_json",
            "metadata_json", "updated_by", "updated_by_username", "updated_at",
            "last_activity_at", "resolved_at", "closed_at",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return self.get_ticket(ticket_id)
        now = _iso()
        values.setdefault("updated_at", now)
        values.setdefault("last_activity_at", now)
        assignments = [f"{key} = ?" for key in values]
        assignments.append("version = version + 1")
        where = "WHERE (id = ? OR ticket_number = ?) AND deleted_at IS NULL"
        params: list[Any] = [*values.values(), ticket_id, ticket_id]
        if expected_version is not None:
            where += " AND version = ?"
            params.append(int(expected_version))
        with self._connect() as connection:
            result = connection.execute(f"UPDATE feedback_tickets SET {', '.join(assignments)} {where}", params)
            if int(result.rowcount or 0) <= 0:
                return None
            row = self._get_ticket_row(connection, ticket_id)
            return self._ticket_from_row(row) if row else None

    def touch_ticket(self, ticket_id: str, *, actor_user_id: str | None, actor_username: str | None) -> None:
        """Update ticket activity ordering without changing ticket version."""
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE feedback_tickets
                SET updated_at = ?, last_activity_at = ?, updated_by = ?, updated_by_username = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now, now, actor_user_id, actor_username, ticket_id),
            )

    def soft_delete_ticket(self, ticket_id: str, *, expected_version: int, actor_user_id: str | None, actor_username: str | None, reason: str | None = None) -> FeedbackTicketRecord | None:
        """Soft-delete one ticket and return its deleted record."""
        now = _iso()
        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE feedback_tickets
                SET deleted_at = ?, delete_reason = ?, updated_at = ?, last_activity_at = ?,
                    updated_by = ?, updated_by_username = ?, version = version + 1
                WHERE (id = ? OR ticket_number = ?) AND deleted_at IS NULL AND version = ?
                """,
                (now, reason, now, now, actor_user_id, actor_username, ticket_id, ticket_id, int(expected_version)),
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = self._get_ticket_row(connection, ticket_id, include_deleted=True)
            return self._ticket_from_row(row) if row else None

    def add_comment(self, payload: dict[str, Any]) -> FeedbackCommentRecord:
        """Create a ticket comment and update ticket activity."""
        comment_id = payload.get("id") or f"fbc_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_comments (id, ticket_id, body, created_by, created_by_username, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (comment_id, payload["ticket_id"], payload["body"], payload.get("created_by"), payload.get("created_by_username"), now),
            )
            connection.execute(
                """
                UPDATE feedback_tickets
                SET updated_at = ?, last_activity_at = ?, updated_by = ?, updated_by_username = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now, now, payload.get("created_by"), payload.get("created_by_username"), payload["ticket_id"]),
            )
            row = connection.execute("SELECT * FROM feedback_comments WHERE id = ?", (comment_id,)).fetchone()
            return self._comment_from_row(row)

    def list_comments(self, ticket_id: str, *, limit: int = 50, offset: int = 0) -> tuple[list[FeedbackCommentRecord], int]:
        """Return comments for one ticket."""
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM feedback_comments WHERE ticket_id = ?", (ticket_id,)).fetchone()[0])
            rows = connection.execute(
                """
                SELECT * FROM feedback_comments
                WHERE ticket_id = ?
                ORDER BY created_at ASC, seq ASC
                LIMIT ? OFFSET ?
                """,
                (ticket_id, int(limit), int(offset)),
            ).fetchall()
            return [self._comment_from_row(row) for row in rows], total

    def add_attachment_metadata(self, payload: dict[str, Any]) -> FeedbackAttachmentRecord:
        """Create attachment metadata and update ticket activity."""
        attachment_id = payload.get("id") or f"fba_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_attachments (
                    id, ticket_id, filename, original_filename, stored_filename,
                    content_type, size_bytes, checksum_sha256, storage_status,
                    created_by, created_by_username, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment_id,
                    payload["ticket_id"],
                    payload["filename"],
                    payload["original_filename"],
                    payload["stored_filename"],
                    payload["content_type"],
                    int(payload["size_bytes"]),
                    payload.get("checksum_sha256"),
                    payload.get("storage_status") or "stored",
                    payload.get("created_by"),
                    payload.get("created_by_username"),
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE feedback_tickets
                SET updated_at = ?, last_activity_at = ?, updated_by = ?, updated_by_username = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now, now, payload.get("created_by"), payload.get("created_by_username"), payload["ticket_id"]),
            )
            row = connection.execute("SELECT * FROM feedback_attachments WHERE id = ?", (attachment_id,)).fetchone()
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

    def get_attachment(self, attachment_id: str, *, include_deleted: bool = False) -> FeedbackAttachmentRecord | None:
        """Return attachment metadata."""
        clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._connect() as connection:
            row = connection.execute(f"SELECT * FROM feedback_attachments WHERE id = ?{clause}", (attachment_id,)).fetchone()
            return self._attachment_from_row(row) if row else None

    def delete_attachment_metadata(self, attachment_id: str) -> FeedbackAttachmentRecord | None:
        """Mark attachment metadata as deleted and return previous row."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM feedback_attachments WHERE id = ? AND deleted_at IS NULL", (attachment_id,)).fetchone()
            if row is None:
                return None
            record = self._attachment_from_row(row)
            now = _iso()
            connection.execute("UPDATE feedback_attachments SET deleted_at = ?, storage_status = ? WHERE id = ?", (now, "deleted", attachment_id))
            connection.execute(
                """
                UPDATE feedback_tickets
                SET updated_at = ?, last_activity_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now, now, record.ticket_id),
            )
            return record

    def add_event(self, payload: dict[str, Any]) -> FeedbackEventRecord:
        """Append a durable lifecycle event."""
        event_id = payload.get("id") or f"fbe_{uuid4().hex}"
        now = payload.get("created_at") or _iso()
        with self._connect() as connection:
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM feedback_events WHERE ticket_id = ?",
                    (payload["ticket_id"],),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO feedback_events (
                    id, ticket_id, event_type, actor_user_id, actor_username,
                    old_status, new_status, note, payload_json, sequence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    payload["ticket_id"],
                    payload["event_type"],
                    payload.get("actor_user_id"),
                    payload.get("actor_username"),
                    payload.get("old_status"),
                    payload.get("new_status"),
                    payload.get("note"),
                    _to_json(payload.get("payload") or {}),
                    sequence,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM feedback_events WHERE id = ?", (event_id,)).fetchone()
            return self._event_from_row(row)

    def list_events(self, ticket_id: str, *, limit: int = 50, offset: int = 0) -> tuple[list[FeedbackEventRecord], int]:
        """Return events for one ticket."""
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM feedback_events WHERE ticket_id = ?", (ticket_id,)).fetchone()[0])
            rows = connection.execute(
                """
                SELECT * FROM feedback_events
                WHERE ticket_id = ?
                ORDER BY sequence ASC
                LIMIT ? OFFSET ?
                """,
                (ticket_id, int(limit), int(offset)),
            ).fetchall()
            return [self._event_from_row(row) for row in rows], total

    def find_idempotency_record(self, *, owner_key: str, operation: str, key_hash: str) -> FeedbackIdempotencyRecord | None:
        """Return a prior idempotency record, if present."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM feedback_idempotency_keys
                WHERE owner_key = ? AND operation = ? AND key_hash = ?
                """,
                (owner_key, operation, key_hash),
            ).fetchone()
            return self._idempotency_from_row(row) if row else None

    def store_idempotency_record(
        self,
        *,
        owner_key: str,
        operation: str,
        key_hash: str,
        request_fingerprint: str,
        resource_type: str,
        resource_id: str,
    ) -> FeedbackIdempotencyRecord:
        """Persist a completed mutation idempotency key."""
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_idempotency_keys (
                    owner_key, operation, key_hash, request_fingerprint,
                    resource_type, resource_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (owner_key, operation, key_hash, request_fingerprint, resource_type, resource_id, now),
            )
            row = connection.execute(
                """
                SELECT * FROM feedback_idempotency_keys
                WHERE owner_key = ? AND operation = ? AND key_hash = ?
                """,
                (owner_key, operation, key_hash),
            ).fetchone()
            return self._idempotency_from_row(row)

