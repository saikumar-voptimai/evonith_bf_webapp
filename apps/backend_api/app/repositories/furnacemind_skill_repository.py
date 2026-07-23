"""SQLite repository for backend-owned FurnaceMind skills."""

from __future__ import annotations

import hashlib
import json
import re
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

_SLUG_RE = re.compile(r"[^a-z0-9]+")


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


def slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", str(value or "").strip().lower()).strip("-")
    return slug[:80] or "skill"


def idempotency_hash(idempotency_key: str | None) -> str | None:
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillRecord:
    id: str
    slug: str
    name: str
    symbol: str | None
    description: str | None
    instruction: str
    source_type: str
    owner_id: str | None
    is_active: bool
    indexed: bool
    index_status: str
    metadata: dict[str, Any]
    idempotency_key_hash: str | None
    request_fingerprint: str | None
    created_at: datetime
    updated_at: datetime | None


class FurnaceMindSkillRepository:
    """Persist prompt-only FurnaceMind skills for the backend API."""

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

    def ensure_schema(self) -> None:
        with self._connect():
            return

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS furnacemind_skills (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                symbol TEXT,
                description TEXT,
                instruction TEXT NOT NULL,
                source_type TEXT NOT NULL,
                owner_id TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                indexed INTEGER NOT NULL DEFAULT 0,
                index_status TEXT NOT NULL DEFAULT 'not_indexed',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                idempotency_key_hash TEXT,
                request_fingerprint TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_fm_skills_owner_active
                ON furnacemind_skills(owner_id, is_active, updated_at);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_fm_skills_owner_idempotency
                ON furnacemind_skills(owner_id, idempotency_key_hash)
                WHERE idempotency_key_hash IS NOT NULL;
            """
        )

    @staticmethod
    def _skill_from_row(row: sqlite3.Row) -> SkillRecord:
        return SkillRecord(
            id=str(row["id"]),
            slug=str(row["slug"]),
            name=str(row["name"]),
            symbol=row["symbol"],
            description=row["description"],
            instruction=str(row["instruction"]),
            source_type=str(row["source_type"]),
            owner_id=row["owner_id"],
            is_active=bool(row["is_active"]),
            indexed=bool(row["indexed"]),
            index_status=str(row["index_status"]),
            metadata=dict(_from_json(row["metadata_json"], {})),
            idempotency_key_hash=row["idempotency_key_hash"],
            request_fingerprint=row["request_fingerprint"],
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]),
        )

    def create_skill(self, payload: dict[str, Any]) -> SkillRecord:
        skill_id = payload.get("id") or f"fms_{uuid4().hex}"
        base_slug = slugify(payload.get("slug") or payload["name"])
        slug = self._unique_slug(base_slug)
        now = _iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO furnacemind_skills (
                    id, slug, name, symbol, description, instruction,
                    source_type, owner_id, is_active, indexed, index_status,
                    metadata_json, idempotency_key_hash, request_fingerprint,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    skill_id,
                    slug,
                    payload["name"],
                    payload.get("symbol"),
                    payload.get("description"),
                    payload["instruction"],
                    payload.get("source_type") or "api",
                    payload.get("owner_id"),
                    int(bool(payload.get("is_active", True))),
                    int(bool(payload.get("indexed", False))),
                    payload.get("index_status") or "not_indexed",
                    _json(payload.get("metadata") or {}),
                    payload.get("idempotency_key_hash"),
                    payload.get("request_fingerprint"),
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM furnacemind_skills WHERE id = ?", (skill_id,)).fetchone()
            return self._skill_from_row(row)

    def _unique_slug(self, base_slug: str) -> str:
        slug = base_slug
        with self._connect() as connection:
            index = 2
            while connection.execute("SELECT 1 FROM furnacemind_skills WHERE slug = ?", (slug,)).fetchone():
                suffix = f"-{index}"
                slug = f"{base_slug[:80 - len(suffix)]}{suffix}"
                index += 1
            return slug

    def find_by_idempotency(self, *, owner_id: str, idempotency_key_hash: str) -> SkillRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM furnacemind_skills WHERE owner_id = ? AND idempotency_key_hash = ?",
                (owner_id, idempotency_key_hash),
            ).fetchone()
            return self._skill_from_row(row) if row else None

    def get_skill(self, skill_id: str) -> SkillRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM furnacemind_skills WHERE id = ?", (skill_id,)).fetchone()
            return self._skill_from_row(row) if row else None

    def list_skills(self, filters: dict[str, Any]) -> tuple[list[SkillRecord], int]:
        clauses: list[str] = []
        values: list[Any] = []
        if filters.get("owner_id") is not None:
            clauses.append("(owner_id = ? OR source_type = 'builtin')")
            values.append(str(filters["owner_id"]))
        if filters.get("active_only") is True:
            clauses.append("is_active = 1")
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = min(200, max(1, int(filters.get("limit") or 50)))
        offset = max(0, int(filters.get("offset") or 0))
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM furnacemind_skills{where_sql}", values).fetchone()[0])
            rows = connection.execute(
                "SELECT * FROM furnacemind_skills"
                + where_sql
                + " ORDER BY is_active DESC, updated_at DESC, seq DESC LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
            return [self._skill_from_row(row) for row in rows], total

    def update_skill(self, skill_id: str, changes: dict[str, Any]) -> SkillRecord | None:
        allowed = {
            "name",
            "symbol",
            "description",
            "instruction",
            "is_active",
            "indexed",
            "index_status",
            "metadata_json",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        if "metadata" in changes:
            values["metadata_json"] = _json(changes["metadata"])
        if "is_active" in values:
            values["is_active"] = int(bool(values["is_active"]))
        if "indexed" in values:
            values["indexed"] = int(bool(values["indexed"]))
        values["updated_at"] = _iso()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._connect() as connection:
            result = connection.execute(
                f"UPDATE furnacemind_skills SET {assignments} WHERE id = ?",
                [*values.values(), skill_id],
            )
            if int(result.rowcount or 0) <= 0:
                return None
            row = connection.execute("SELECT * FROM furnacemind_skills WHERE id = ?", (skill_id,)).fetchone()
            return self._skill_from_row(row) if row else None