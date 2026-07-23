"""Durable SQLite repository for V-Sense contexts, profiles, and idempotency."""

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


def default_vsense_db_path() -> Path:
    """Return the runtime SQLite path for V-Sense durable state."""

    return runtime_path("vsense", "vsense.sqlite", create_parent=True)


def fingerprint(payload: Any) -> str:
    """Return a stable fingerprint for an idempotent V-Sense request."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _key_hash(idempotency_key: str | None) -> str | None:
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IdempotencyRecord:
    response: dict[str, Any]
    request_fingerprint: str


class VSenseRepository:
    """SQLite-backed V-Sense repository with optimistic profile writes."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_vsense_db_path()
        self._lock = threading.RLock()
        self._schema_ready = False

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            with self._lock:
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
            CREATE TABLE IF NOT EXISTS vsense_contexts (
                context_id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                optimization_type_id TEXT NOT NULL,
                catalog_version TEXT NOT NULL,
                algorithm_version TEXT NOT NULL,
                dataset_version TEXT,
                control_profile_version INTEGER,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                as_of TEXT NOT NULL,
                context_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_vsense_context_owner
                ON vsense_contexts(owner_user_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_vsense_context_expiry
                ON vsense_contexts(expires_at);

            CREATE TABLE IF NOT EXISTS vsense_control_profiles (
                profile_id TEXT NOT NULL,
                optimization_type_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                catalog_version TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                updated_by_user_id TEXT,
                updated_by_username TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, optimization_type_id)
            );

            CREATE TABLE IF NOT EXISTS vsense_control_profile_history (
                profile_id TEXT NOT NULL,
                optimization_type_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                catalog_version TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                updated_by_user_id TEXT,
                updated_by_username TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, optimization_type_id, version)
            );

            CREATE TABLE IF NOT EXISTS vsense_runs (
                run_id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                owner_username TEXT,
                optimization_type_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                request_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(context_id) REFERENCES vsense_contexts(context_id) ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS ix_vsense_runs_owner
                ON vsense_runs(owner_user_id, created_at);

            CREATE TABLE IF NOT EXISTS vsense_idempotency (
                owner_user_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                idempotency_key_hash TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (owner_user_id, scope, idempotency_key_hash)
            );
            """
        )

    def get_idempotent_response(
        self,
        *,
        owner_user_id: str,
        scope: str,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> dict[str, Any] | None:
        key_hash = _key_hash(idempotency_key)
        if not key_hash:
            raise ApiError(
                "INVALID_IDEMPOTENCY_KEY",
                "Idempotency-Key is required.",
                status_code=422,
            )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_fingerprint, response_json
                FROM vsense_idempotency
                WHERE owner_user_id = ? AND scope = ? AND idempotency_key_hash = ?
                """,
                (owner_user_id, scope, key_hash),
            ).fetchone()
        if row is None:
            return None
        if str(row["request_fingerprint"]) != request_fingerprint:
            raise ApiError(
                "IDEMPOTENCY_KEY_REUSED",
                "Idempotency-Key was already used for a different V-Sense request.",
                status_code=409,
            )
        return dict(_from_json(row["response_json"], {}))

    def store_idempotent_response(
        self,
        *,
        owner_user_id: str,
        scope: str,
        idempotency_key: str,
        request_fingerprint: str,
        response: dict[str, Any],
    ) -> None:
        key_hash = _key_hash(idempotency_key)
        if not key_hash:
            raise ApiError(
                "INVALID_IDEMPOTENCY_KEY",
                "Idempotency-Key is required.",
                status_code=422,
            )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO vsense_idempotency (
                    owner_user_id, scope, idempotency_key_hash,
                    request_fingerprint, response_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    scope,
                    key_hash,
                    request_fingerprint,
                    _json(response),
                    _now(),
                ),
            )

    def store_context(self, context: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO vsense_contexts (
                    context_id, owner_user_id, optimization_type_id,
                    catalog_version, algorithm_version, dataset_version,
                    control_profile_version, created_at, expires_at, as_of,
                    context_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context["context_id"],
                    str(context.get("owner_user_id") or ""),
                    context["optimization_type_id"],
                    context["catalog_version"],
                    context["algorithm_version"],
                    context.get("dataset", {}).get("version"),
                    context.get("control_profile", {}).get("version"),
                    context["created_at"],
                    context["expires_at"],
                    context["as_of"],
                    _json(context),
                ),
            )
        return context

    def get_context(self, context_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT context_json FROM vsense_contexts WHERE context_id = ?",
                (context_id,),
            ).fetchone()
        return dict(_from_json(row["context_json"], {})) if row is not None else None

    def get_or_create_profile(
        self,
        *,
        optimization_type_id: str,
        default_parameters: list[dict[str, Any]],
        catalog_version: str,
    ) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM vsense_control_profiles
                WHERE profile_id = 'plant-default' AND optimization_type_id = ?
                """,
                (optimization_type_id,),
            ).fetchone()
            if row is not None:
                return self._profile_from_row(row)
            now = _now()
            connection.execute(
                """
                INSERT INTO vsense_control_profiles (
                    profile_id, optimization_type_id, version, catalog_version,
                    parameters_json, created_at, updated_at
                )
                VALUES ('plant-default', ?, 1, ?, ?, ?, ?)
                """,
                (optimization_type_id, catalog_version, _json(default_parameters), now, now),
            )
            connection.execute(
                """
                INSERT INTO vsense_control_profile_history (
                    profile_id, optimization_type_id, version, catalog_version,
                    parameters_json, created_at
                )
                VALUES ('plant-default', ?, 1, ?, ?, ?)
                """,
                (optimization_type_id, catalog_version, _json(default_parameters), now),
            )
            row = connection.execute(
                """
                SELECT * FROM vsense_control_profiles
                WHERE profile_id = 'plant-default' AND optimization_type_id = ?
                """,
                (optimization_type_id,),
            ).fetchone()
        return self._profile_from_row(row)

    def update_profile(
        self,
        *,
        optimization_type_id: str,
        profile_id: str,
        expected_version: int,
        parameters: list[dict[str, Any]],
        catalog_version: str,
        actor_user_id: str | None,
        actor_username: str | None,
    ) -> dict[str, Any]:
        if profile_id != "plant-default":
            raise ApiError(
                "VSENSE_INVALID_CONTROL_PLAN",
                "Only plant-default control profile is supported in this release.",
                status_code=400,
            )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM vsense_control_profiles
                WHERE profile_id = ? AND optimization_type_id = ?
                """,
                (profile_id, optimization_type_id),
            ).fetchone()
            if row is None:
                raise ApiError(
                    "VSENSE_INVALID_OPTIMIZATION_TYPE",
                    "Control profile does not exist.",
                    status_code=404,
                )
            current = self._profile_from_row(row)
            if int(current["version"]) != int(expected_version):
                raise ApiError(
                    "VSENSE_CONTROL_PROFILE_VERSION_CONFLICT",
                    "Control profile version conflict.",
                    status_code=409,
                    details={"current_version": current["version"]},
                )
            new_version = int(current["version"]) + 1
            now = _now()
            connection.execute(
                """
                UPDATE vsense_control_profiles
                SET version = ?, catalog_version = ?, parameters_json = ?,
                    updated_by_user_id = ?, updated_by_username = ?, updated_at = ?
                WHERE profile_id = ? AND optimization_type_id = ?
                """,
                (
                    new_version,
                    catalog_version,
                    _json(parameters),
                    actor_user_id,
                    actor_username,
                    now,
                    profile_id,
                    optimization_type_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO vsense_control_profile_history (
                    profile_id, optimization_type_id, version, catalog_version,
                    parameters_json, updated_by_user_id, updated_by_username, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    optimization_type_id,
                    new_version,
                    catalog_version,
                    _json(parameters),
                    actor_user_id,
                    actor_username,
                    now,
                ),
            )
            updated = connection.execute(
                """
                SELECT * FROM vsense_control_profiles
                WHERE profile_id = ? AND optimization_type_id = ?
                """,
                (profile_id, optimization_type_id),
            ).fetchone()
        return self._profile_from_row(updated)

    def store_run_metadata(
        self,
        *,
        run_id: str,
        owner_user_id: str,
        owner_username: str | None,
        optimization_type_id: str,
        context_id: str,
        request_payload: dict[str, Any],
        created_at: str,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO vsense_runs (
                    run_id, owner_user_id, owner_username, optimization_type_id,
                    context_id, request_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    owner_user_id,
                    owner_username,
                    optimization_type_id,
                    context_id,
                    _json(request_payload),
                    created_at,
                ),
            )

    def get_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vsense_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": str(row["run_id"]),
            "owner_user_id": str(row["owner_user_id"]),
            "owner_username": row["owner_username"],
            "optimization_type_id": str(row["optimization_type_id"]),
            "context_id": str(row["context_id"]),
            "request": dict(_from_json(row["request_json"], {})),
            "created_at": str(row["created_at"]),
        }
    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "profile_id": str(row["profile_id"]),
            "optimization_type_id": str(row["optimization_type_id"]),
            "version": int(row["version"]),
            "catalog_version": str(row["catalog_version"]),
            "parameters": list(_from_json(row["parameters_json"], [])),
            "updated_by_user_id": row["updated_by_user_id"],
            "updated_by_username": row["updated_by_username"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }


