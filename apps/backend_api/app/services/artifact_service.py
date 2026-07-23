"""Runtime-backed, owner-aware artifact storage for data and dataset downloads."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Iterator

import pandas as pd

from furnace_data.runtime_paths import get_dataset_results_dir

_ARTIFACT_DIRNAME = "artifacts"
_IDEMPOTENCY_DB_FILENAME = "artifact_idempotency.sqlite"
_SAFE_ID = re.compile(r"^[a-f0-9]{32}$")
_ARTIFACT_LOCK = Lock()


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when artifact metadata cannot be found safely."""


class ArtifactExpiredError(FileNotFoundError):
    """Raised after an expired artifact has been removed."""


class ArtifactIdempotencyConflictError(RuntimeError):
    """Raised when one owner reuses an export key for a different request."""


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_id: str
    # ``filename`` is the safe user-visible download name. ``storage_name`` is
    # intentionally server-only so a response never discloses a local path.
    filename: str
    content_type: str
    row_count: int | None
    created_at: str
    expires_at: str | None = None
    owner_user_id: str | None = None
    query_fingerprint: str | None = None
    idempotency_key_hash: str | None = None
    artifact_kind: str = "data_export"
    storage_name: str | None = None


def _artifact_root() -> Path:
    path = get_dataset_results_dir() / _ARTIFACT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _idempotency_database_path() -> Path:
    """Return the durable, process-shared idempotency registry location."""
    return _artifact_root() / _IDEMPOTENCY_DB_FILENAME


def _ensure_idempotency_schema(connection: sqlite3.Connection) -> None:
    """Create the small registry used to coordinate API worker processes.

    The key itself is intentionally hashed before it reaches disk. The primary
    key is owner + key hash (rather than including the fingerprint): a replay
    with another fingerprint must remain a conflict, not create a second
    artifact.
    """
    # Avoid ``executescript`` here: Python's sqlite wrapper commits before an
    # executescript call, which would release the BEGIN IMMEDIATE lock between
    # schema initialization and the owner/key lookup below.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS artifact_idempotency (
            owner_user_id TEXT NOT NULL,
            idempotency_key_hash TEXT NOT NULL,
            query_fingerprint TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (owner_user_id, idempotency_key_hash)
        )
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS ix_artifact_idempotency_artifact
        ON artifact_idempotency(artifact_id)
        """)


@contextmanager
def _idempotency_connection(*, write: bool) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection that safely serializes cross-process writes."""
    connection = sqlite3.connect(
        str(_idempotency_database_path()),
        timeout=30,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        if write:
            # Acquire this before creating the table or reading a key record.
            # Two Uvicorn workers cannot both decide one owner/key is absent.
            connection.execute("BEGIN IMMEDIATE")
        _ensure_idempotency_schema(connection)
        if not write:
            connection.commit()
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def sanitize_filename(name: str) -> str:
    """Return a conservative download filename without path components."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("._") or "artifact"
    return cleaned[:120]


def _metadata_path(artifact_id: str) -> Path:
    return _artifact_root() / f"{artifact_id}.json"


def _expired_metadata_path(artifact_id: str) -> Path:
    return _artifact_root() / f"{artifact_id}.expired.json"


def _validate_artifact_id(artifact_id: str) -> None:
    if not _SAFE_ID.match(str(artifact_id or "")):
        raise ValueError("Invalid artifact id")


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_expired(metadata: ArtifactMetadata, *, now: datetime | None = None) -> bool:
    if not metadata.expires_at:
        return False
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return _parse_datetime(metadata.expires_at) <= now_utc


def _artifact_file_path(metadata: ArtifactMetadata) -> Path:
    storage_name = metadata.storage_name or metadata.filename
    path = (_artifact_root() / storage_name).resolve()
    root = _artifact_root().resolve()
    if root not in path.parents:
        raise ValueError("Invalid artifact path")
    return path


def _remove_idempotency_records_for_artifact(
    artifact_id: str,
    *,
    connection: sqlite3.Connection | None = None,
) -> None:
    """Remove mappings after an artifact expires or is no longer usable."""
    if connection is not None:
        connection.execute(
            "DELETE FROM artifact_idempotency WHERE artifact_id = ?", (artifact_id,)
        )
        return
    try:
        with _idempotency_connection(write=True) as managed_connection:
            managed_connection.execute(
                "DELETE FROM artifact_idempotency WHERE artifact_id = ?",
                (artifact_id,),
            )
    except sqlite3.Error:
        # Expiry must continue to yield a tombstone even if the registry is
        # temporarily unavailable. A later lookup will discard stale records.
        return


def _delete_artifact(
    metadata: ArtifactMetadata,
    *,
    idempotency_connection: sqlite3.Connection | None = None,
) -> None:
    """Tombstone an expired artifact so every later request remains a 410."""
    artifact_path = _artifact_file_path(metadata)
    tombstone_path = _expired_metadata_path(metadata.artifact_id)
    temporary_path = tombstone_path.with_name(
        f".{tombstone_path.name}.{uuid.uuid4().hex}.tmp"
    )
    payload = {
        "artifact_id": metadata.artifact_id,
        "expired_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        temporary_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary_path.replace(tombstone_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    artifact_path.unlink(missing_ok=True)
    _metadata_path(metadata.artifact_id).unlink(missing_ok=True)
    _remove_idempotency_records_for_artifact(
        metadata.artifact_id,
        connection=idempotency_connection,
    )


def _read_artifact_metadata(artifact_id: str) -> ArtifactMetadata:
    path = _metadata_path(artifact_id)
    if not path.exists():
        raise ArtifactNotFoundError(artifact_id)
    try:
        return ArtifactMetadata(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactNotFoundError(artifact_id) from exc


def get_artifact_metadata(artifact_id: str) -> ArtifactMetadata:
    """Load metadata and preserve expired-artifact semantics with a tombstone."""
    _validate_artifact_id(artifact_id)
    if _expired_metadata_path(artifact_id).exists():
        raise ArtifactExpiredError(artifact_id)
    metadata = _read_artifact_metadata(artifact_id)
    if _is_expired(metadata):
        _delete_artifact(metadata)
        raise ArtifactExpiredError(artifact_id)
    return metadata


def get_artifact_path(artifact_id: str) -> Path:
    metadata = get_artifact_metadata(artifact_id)
    path = _artifact_file_path(metadata)
    if not path.exists():
        raise ArtifactNotFoundError(artifact_id)
    return path


def _idempotency_key_hash(idempotency_key: str | None) -> str | None:
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _delete_idempotency_record(
    connection: sqlite3.Connection,
    *,
    owner_user_id: str,
    key_hash: str,
) -> None:
    connection.execute(
        """
        DELETE FROM artifact_idempotency
        WHERE owner_user_id = ? AND idempotency_key_hash = ?
        """,
        (owner_user_id, key_hash),
    )


def _artifact_for_idempotency_record(
    connection: sqlite3.Connection,
    *,
    owner_user_id: str,
    key_hash: str,
    record: sqlite3.Row,
) -> ArtifactMetadata | None:
    """Resolve a registry record only while it still points at a live artifact."""
    artifact_id = str(record["artifact_id"])
    try:
        metadata = _read_artifact_metadata(artifact_id)
    except ArtifactNotFoundError:
        _delete_idempotency_record(
            connection,
            owner_user_id=owner_user_id,
            key_hash=key_hash,
        )
        return None

    if _is_expired(metadata):
        _delete_artifact(metadata, idempotency_connection=connection)
        return None
    if (
        metadata.owner_user_id != owner_user_id
        or metadata.idempotency_key_hash != key_hash
        or metadata.query_fingerprint != str(record["query_fingerprint"])
    ):
        _delete_idempotency_record(
            connection,
            owner_user_id=owner_user_id,
            key_hash=key_hash,
        )
        return None
    try:
        if not _artifact_file_path(metadata).exists():
            _delete_idempotency_record(
                connection,
                owner_user_id=owner_user_id,
                key_hash=key_hash,
            )
            return None
    except ValueError:
        _delete_idempotency_record(
            connection,
            owner_user_id=owner_user_id,
            key_hash=key_hash,
        )
        return None
    return metadata


def _find_legacy_idempotent_artifact(
    connection: sqlite3.Connection,
    *,
    owner_user_id: str,
    query_fingerprint: str,
    key_hash: str,
) -> ArtifactMetadata | None:
    """Backfill a durable record for artifacts created before this registry."""
    for metadata_file in _artifact_root().glob("*.json"):
        try:
            metadata = ArtifactMetadata(
                **json.loads(metadata_file.read_text(encoding="utf-8"))
            )
            if _is_expired(metadata):
                _delete_artifact(metadata, idempotency_connection=connection)
                continue
            if (
                metadata.owner_user_id != owner_user_id
                or metadata.idempotency_key_hash != key_hash
            ):
                continue
            if metadata.query_fingerprint != query_fingerprint:
                raise ArtifactIdempotencyConflictError(
                    "Idempotency-Key was already used for a different export request."
                )
            if not _artifact_file_path(metadata).exists():
                continue
            connection.execute(
                """
                INSERT INTO artifact_idempotency (
                    owner_user_id, idempotency_key_hash, query_fingerprint,
                    artifact_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    owner_user_id,
                    key_hash,
                    query_fingerprint,
                    metadata.artifact_id,
                    metadata.created_at,
                ),
            )
            return metadata
        except ArtifactIdempotencyConflictError:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # Corrupt metadata is ignored here; it will never be downloadable.
            continue
    return None


def _find_idempotent_artifact_in_registry(
    connection: sqlite3.Connection,
    *,
    owner_user_id: str,
    query_fingerprint: str,
    idempotency_key: str,
) -> ArtifactMetadata | None:
    """Resolve a same-owner export under an already-acquired SQLite lock."""
    key_hash = _idempotency_key_hash(idempotency_key)
    if not key_hash:
        return None
    record = connection.execute(
        """
        SELECT owner_user_id, idempotency_key_hash, query_fingerprint, artifact_id
        FROM artifact_idempotency
        WHERE owner_user_id = ? AND idempotency_key_hash = ?
        """,
        (owner_user_id, key_hash),
    ).fetchone()
    if record is not None:
        existing = _artifact_for_idempotency_record(
            connection,
            owner_user_id=owner_user_id,
            key_hash=key_hash,
            record=record,
        )
        if existing is not None:
            if str(record["query_fingerprint"]) != query_fingerprint:
                raise ArtifactIdempotencyConflictError(
                    "Idempotency-Key was already used for a different export request."
                )
            return existing
    return _find_legacy_idempotent_artifact(
        connection,
        owner_user_id=owner_user_id,
        query_fingerprint=query_fingerprint,
        key_hash=key_hash,
    )


def find_idempotent_artifact(
    *,
    owner_user_id: str,
    query_fingerprint: str,
    idempotency_key: str,
) -> ArtifactMetadata | None:
    """Return a live artifact for the same owner/key/query, if one exists."""
    if not _idempotency_key_hash(idempotency_key):
        return None
    with _ARTIFACT_LOCK:
        with _idempotency_connection(write=True) as connection:
            return _find_idempotent_artifact_in_registry(
                connection,
                owner_user_id=str(owner_user_id),
                query_fingerprint=query_fingerprint,
                idempotency_key=idempotency_key,
            )


def _write_csv_artifact(
    dataframe: pd.DataFrame,
    filename_prefix: str,
    *,
    ttl_hours: int,
    owner_user_id: str | None,
    query_fingerprint: str | None,
    idempotency_key: str | None,
    artifact_kind: str,
) -> ArtifactMetadata:
    """Write the artifact files before their idempotency record is committed."""
    visible_name = sanitize_filename(filename_prefix)
    if not visible_name.lower().endswith(".csv"):
        visible_name = f"{visible_name}.csv"

    artifact_id = uuid.uuid4().hex
    storage_name = f"{artifact_id}_{visible_name}"
    path = _artifact_root() / storage_name
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(hours=max(1, ttl_hours))
    frame = _normalize_csv_timestamps(dataframe)
    try:
        frame.to_csv(
            temporary_path,
            index=True,
            date_format="%Y-%m-%dT%H:%M:%SZ",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

    metadata = ArtifactMetadata(
        artifact_id=artifact_id,
        filename=visible_name,
        content_type="text/csv",
        row_count=len(frame),
        created_at=created_at.isoformat(),
        expires_at=expires_at.isoformat(),
        owner_user_id=str(owner_user_id) if owner_user_id else None,
        query_fingerprint=query_fingerprint,
        idempotency_key_hash=_idempotency_key_hash(idempotency_key),
        artifact_kind=sanitize_filename(artifact_kind) or "data_export",
        storage_name=storage_name,
    )
    metadata_path = _metadata_path(artifact_id)
    metadata_temporary_path = metadata_path.with_name(
        f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        metadata_temporary_path.write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata_temporary_path.replace(metadata_path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        metadata_temporary_path.unlink(missing_ok=True)
    return metadata


def _discard_unpublished_artifact(metadata: ArtifactMetadata) -> None:
    """Remove files created before a failed registry transaction commits."""
    _artifact_file_path(metadata).unlink(missing_ok=True)
    _metadata_path(metadata.artifact_id).unlink(missing_ok=True)


def create_csv_artifact(
    dataframe: pd.DataFrame,
    filename_prefix: str,
    *,
    ttl_hours: int = 24,
    owner_user_id: str | None = None,
    query_fingerprint: str | None = None,
    idempotency_key: str | None = None,
    artifact_kind: str = "data_export",
) -> ArtifactMetadata:
    """Persist a CSV artifact and metadata without exposing its storage path.

    Idempotent exports are serialized through SQLite instead of a process-local
    lock. The registry record is committed only after both CSV and metadata are
    durable, so another Uvicorn worker either replays one complete artifact or
    creates it itself after a failed writer rolls back.
    """
    use_idempotency_registry = bool(
        owner_user_id and query_fingerprint and _idempotency_key_hash(idempotency_key)
    )
    with _ARTIFACT_LOCK:
        if not use_idempotency_registry:
            return _write_csv_artifact(
                dataframe,
                filename_prefix,
                ttl_hours=ttl_hours,
                owner_user_id=owner_user_id,
                query_fingerprint=query_fingerprint,
                idempotency_key=idempotency_key,
                artifact_kind=artifact_kind,
            )

        owner = str(owner_user_id)
        with _idempotency_connection(write=True) as connection:
            existing = _find_idempotent_artifact_in_registry(
                connection,
                owner_user_id=owner,
                query_fingerprint=str(query_fingerprint),
                idempotency_key=str(idempotency_key),
            )
            if existing is not None:
                return existing

            artifact = _write_csv_artifact(
                dataframe,
                filename_prefix,
                ttl_hours=ttl_hours,
                owner_user_id=owner,
                query_fingerprint=str(query_fingerprint),
                idempotency_key=idempotency_key,
                artifact_kind=artifact_kind,
            )
            try:
                connection.execute(
                    """
                    INSERT INTO artifact_idempotency (
                        owner_user_id, idempotency_key_hash, query_fingerprint,
                        artifact_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        owner,
                        _idempotency_key_hash(idempotency_key),
                        str(query_fingerprint),
                        artifact.artifact_id,
                        artifact.created_at,
                    ),
                )
            except Exception:
                _discard_unpublished_artifact(artifact)
                raise
            return artifact


def _normalize_csv_timestamps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame datetimes to UTC before writing a stable CSV."""
    frame = dataframe.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        index = pd.to_datetime(frame.index, errors="coerce", utc=True)
        frame.index = index
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def artifact_is_accessible_by(
    metadata: ArtifactMetadata,
    *,
    user_id: str,
    permissions: set[str] | frozenset[str] | list[str] | tuple[str, ...],
) -> bool:
    """Check owner access or the explicit elevated export permission."""
    permission_set = {str(item) for item in permissions}
    return metadata.owner_user_id == str(user_id) or "data:export:any" in permission_set


def delete_expired_artifacts(now: datetime | None = None) -> int:
    """Delete expired metadata and files, returning the number of artifacts removed."""
    deleted = 0
    for metadata_file in _artifact_root().glob("*.json"):
        try:
            metadata = ArtifactMetadata(
                **json.loads(metadata_file.read_text(encoding="utf-8"))
            )
            if not _is_expired(metadata, now=now):
                continue
            _delete_artifact(metadata)
            deleted += 1
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return deleted
