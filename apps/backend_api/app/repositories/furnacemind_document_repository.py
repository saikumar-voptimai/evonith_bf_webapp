"""SQLite repository for FurnaceMind document metadata and chunks."""

from __future__ import annotations

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


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    filename: str
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    status: str
    indexed: bool
    chunk_count: int | None
    owner_id: str | None
    created_at: datetime
    updated_at: datetime | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentChunkRecord:
    id: str
    document_id: str
    sequence: int
    content: str
    metadata: dict[str, Any]


class FurnaceMindDocumentRepository:
    """Document metadata/chunk repository with no path exposure in DTOs."""

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
            CREATE TABLE IF NOT EXISTS furnacemind_documents (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                indexed INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER,
                owner_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_fm_documents_owner
                ON furnacemind_documents(owner_id, created_at);

            CREATE TABLE IF NOT EXISTS furnacemind_document_chunks (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                document_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(document_id) REFERENCES furnacemind_documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_fm_chunks_document
                ON furnacemind_document_chunks(document_id, sequence);
            """
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=str(row["id"]),
            filename=str(row["filename"]),
            original_filename=str(row["original_filename"]),
            stored_filename=str(row["stored_filename"]),
            content_type=str(row["content_type"]),
            size_bytes=int(row["size_bytes"]),
            status=str(row["status"]),
            indexed=bool(row["indexed"]),
            chunk_count=row["chunk_count"],
            owner_id=row["owner_id"],
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]),
            metadata=dict(_from_json(row["metadata_json"], {})),
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> DocumentChunkRecord:
        return DocumentChunkRecord(
            id=str(row["id"]),
            document_id=str(row["document_id"]),
            sequence=int(row["sequence"]),
            content=str(row["content"]),
            metadata=dict(_from_json(row["metadata_json"], {})),
        )

    def create_document_metadata(self, payload: dict[str, Any]) -> DocumentRecord:
        document_id = payload.get("id") or f"fmd_{uuid4().hex}"
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO furnacemind_documents (
                    id, filename, original_filename, stored_filename, content_type,
                    size_bytes, status, indexed, chunk_count, owner_id,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    payload["filename"],
                    payload["original_filename"],
                    payload["stored_filename"],
                    payload["content_type"],
                    int(payload["size_bytes"]),
                    payload.get("status") or "uploaded",
                    int(bool(payload.get("indexed", False))),
                    payload.get("chunk_count"),
                    payload.get("owner_id"),
                    _json(payload.get("metadata") or {}),
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM furnacemind_documents WHERE id = ?", (document_id,)).fetchone()
            return self._document_from_row(row)

    def update_document_status(self, document_id: str, **updates: Any) -> DocumentRecord | None:
        allowed = {"status", "indexed", "chunk_count", "metadata_json"}
        values = {key: value for key, value in updates.items() if key in allowed}
        if "metadata" in updates:
            values["metadata_json"] = _json(updates["metadata"])
        if "indexed" in values:
            values["indexed"] = int(bool(values["indexed"]))
        values["updated_at"] = _iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            result = connection.execute(
                f"UPDATE furnacemind_documents SET {assignments} WHERE id = ?",
                [*values.values(), document_id],
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute("SELECT * FROM furnacemind_documents WHERE id = ?", (document_id,)).fetchone()
            return self._document_from_row(row) if row else None

    def list_documents(self, filters: dict[str, Any]) -> tuple[list[DocumentRecord], int]:
        clauses: list[str] = []
        values: list[Any] = []
        owner_id = filters.get("owner_id")
        if owner_id is not None:
            clauses.append("owner_id = ?")
            values.append(str(owner_id))
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = min(200, max(1, int(filters.get("limit") or 50)))
        offset = max(0, int(filters.get("offset") or 0))
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM furnacemind_documents{where_sql}", values).fetchone()[0])
            rows = connection.execute(
                "SELECT * FROM furnacemind_documents"
                + where_sql
                + " ORDER BY created_at DESC, seq DESC LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
            return [self._document_from_row(row) for row in rows], total

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM furnacemind_documents WHERE id = ?", (document_id,)).fetchone()
            return self._document_from_row(row) if row else None

    def delete_document_metadata(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM furnacemind_documents WHERE id = ?", (document_id,)).fetchone()
            if row is None:
                return None
            record = self._document_from_row(row)
            connection.execute("DELETE FROM furnacemind_documents WHERE id = ?", (document_id,))
            return record

    def replace_chunks(self, document_id: str, chunks: list[str], metadata: dict[str, Any] | None = None) -> list[DocumentChunkRecord]:
        with self._connect() as connection:
            connection.execute("DELETE FROM furnacemind_document_chunks WHERE document_id = ?", (document_id,))
            records = []
            for sequence, content in enumerate(chunks, start=1):
                chunk_id = f"fmdc_{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO furnacemind_document_chunks (
                        id, document_id, sequence, content, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chunk_id, document_id, sequence, content, _json(metadata or {})),
                )
                row = connection.execute(
                    "SELECT * FROM furnacemind_document_chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
                records.append(self._chunk_from_row(row))
            return records

    def list_chunks(self, document_id: str, *, limit: int = 100) -> list[DocumentChunkRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM furnacemind_document_chunks
                WHERE document_id = ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (document_id, min(500, max(1, int(limit)))),
            ).fetchall()
            return [self._chunk_from_row(row) for row in rows]
