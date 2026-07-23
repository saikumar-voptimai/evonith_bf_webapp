"""Runtime repository for versioned Material Balance configuration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from apps.backend_api.app.core.errors import ApiError
from furnace_data.runtime_paths import runtime_path


@dataclass(frozen=True)
class MaterialBalanceConfigRevision:
    revision_id: int
    profile_key: str
    version: str
    config: dict[str, Any]
    packaged_default_checksum: str
    created_at: str
    created_by: str | None
    request_id: str | None
    client_metadata: dict[str, Any]


def default_material_balance_db_path() -> Path:
    return runtime_path("material_balance", "config.sqlite", create_parent=True)


def checksum(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _loads(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default


class MaterialBalanceConfigRepository:
    """SQLite-backed immutable configuration revisions."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_material_balance_db_path()
        self._lock = threading.RLock()
        self._schema_ready = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            with self._lock:
                if not self._schema_ready:
                    self.ensure_schema(connection)
                    self._schema_ready = True
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS material_balance_config_revisions (
                revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_key TEXT NOT NULL,
                version TEXT NOT NULL UNIQUE,
                config_json TEXT NOT NULL,
                packaged_default_checksum TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT,
                request_id TEXT,
                client_metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS ix_material_balance_config_profile
                ON material_balance_config_revisions(profile_key, revision_id DESC);
            """
        )

    def latest_revision(self, profile_key: str = "plant-default") -> MaterialBalanceConfigRevision | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM material_balance_config_revisions
                WHERE profile_key = ?
                ORDER BY revision_id DESC
                LIMIT 1
                """,
                (profile_key,),
            ).fetchone()
        return self._revision_from_row(row) if row is not None else None

    def create_revision(
        self,
        *,
        profile_key: str,
        expected_config_version: str,
        config: dict[str, Any],
        packaged_default_checksum: str,
        actor_user_id: str | None,
        request_id: str | None,
        client_metadata: dict[str, Any] | None = None,
    ) -> MaterialBalanceConfigRevision:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM material_balance_config_revisions
                WHERE profile_key = ?
                ORDER BY revision_id DESC
                LIMIT 1
                """,
                (profile_key,),
            ).fetchone()
            current = self._revision_from_row(row) if row is not None else None
            current_version = current.version if current is not None else expected_config_version
            if str(expected_config_version) != str(current_version):
                raise ApiError(
                    "MATERIAL_BALANCE_CONFIG_VERSION_CONFLICT",
                    "Material Balance configuration version conflict.",
                    status_code=409,
                    details={"current_version": current_version},
                )
            next_number = 1 if current is None else current.revision_id + 1
            version = f"mbcfg-{next_number}"
            created_at = _now()
            clean = dict(config)
            clean["version"] = version
            connection.execute(
                """
                INSERT INTO material_balance_config_revisions (
                    profile_key, version, config_json, packaged_default_checksum,
                    created_at, created_by, request_id, client_metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_key,
                    version,
                    _json(clean),
                    packaged_default_checksum,
                    created_at,
                    actor_user_id,
                    request_id,
                    _json(client_metadata or {}),
                ),
            )
            row = connection.execute(
                "SELECT * FROM material_balance_config_revisions WHERE version = ?",
                (version,),
            ).fetchone()
        return self._revision_from_row(row)

    @staticmethod
    def fallback_version(default_config: dict[str, Any]) -> str:
        return f"mbcfg-{checksum(default_config)[:12]}"

    @staticmethod
    def _revision_from_row(row: sqlite3.Row) -> MaterialBalanceConfigRevision:
        return MaterialBalanceConfigRevision(
            revision_id=int(row["revision_id"]),
            profile_key=str(row["profile_key"]),
            version=str(row["version"]),
            config=dict(_loads(row["config_json"], {})),
            packaged_default_checksum=str(row["packaged_default_checksum"]),
            created_at=str(row["created_at"]),
            created_by=row["created_by"],
            request_id=row["request_id"],
            client_metadata=dict(_loads(row["client_metadata_json"], {})),
        )