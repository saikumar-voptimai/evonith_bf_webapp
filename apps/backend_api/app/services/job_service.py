"""Durable job storage and controlled workers for static dataset operations.

The previous implementation kept jobs in a process-local dictionary and used
untracked daemon threads.  This module keeps the small compatibility surface
(``create_job``, ``update_job`` and ``run_background``), while making job state,
progress events, ownership and idempotency durable in runtime SQLite storage.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from apps.backend_api.app.config import settings as legacy_settings
from apps.backend_api.app.core.errors import ApiError
from furnace_data.runtime_paths import runtime_path


JOB_STATUSES = {"pending", "running", "completed", "failed", "cancelled", "expired"}
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "expired"}
_MUTATING_OPERATIONS = {"build_range", "extend", "override"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _utc_now()).astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _configured_positive_int(env_name: str, setting_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is not None:
        try:
            return max(1, int(raw_value))
        except ValueError:
            return default
    try:
        return max(1, int(getattr(legacy_settings, setting_name)))
    except (AttributeError, TypeError, ValueError):
        return default


def _worker_count() -> int:
    return _configured_positive_int("DATASET_JOB_WORKERS", "dataset_job_workers", 1)


def _job_ttl_hours() -> int:
    """Return durable-job retention in hours, with a safe positive default."""
    return _configured_positive_int("DATASET_JOB_TTL_HOURS", "dataset_job_ttl_hours", 24)


def _recovery_stale_minutes() -> int:
    """Avoid interrupting jobs that may be active in another worker process."""
    try:
        return max(1, int(os.getenv("DATASET_JOB_RECOVERY_STALE_MINUTES", "10")))
    except ValueError:
        return 10


@dataclass
class JobState:
    job_id: str
    status: str = "pending"
    progress: float | None = None
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifact_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    operation: str = "legacy_refresh"
    owner_user_id: str | None = None
    owner_username: str | None = None
    idempotency_key: str | None = None
    request_fingerprint: str | None = None
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    expected_dataset_version: str | None = None
    result: dict[str, Any] | None = None
    cancel_requested: bool = False


@dataclass(frozen=True)
class JobEvent:
    job_id: str
    sequence: int
    stage: str
    percent: float
    message: str
    created_at: datetime


class JobService:
    """SQLite-backed static dataset job store with a bounded worker executor."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        max_workers: int | None = None,
    ) -> None:
        self._configured_db_path = Path(db_path) if db_path is not None else None
        self._lock = threading.RLock()
        self._canonical_mutation_lock = threading.RLock()
        self._jobs: dict[str, JobState] = {}  # Backward-compatible in-process cache.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or _worker_count(),
            thread_name_prefix="dataset-job",
        )
        self._futures: dict[str, Future[None]] = {}
        self._initialized_paths: set[str] = set()

    def _database_path(self) -> Path:
        if self._configured_db_path is not None:
            self._configured_db_path.parent.mkdir(parents=True, exist_ok=True)
            return self._configured_db_path
        return runtime_path("jobs", "dataset_jobs.sqlite", create_parent=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        path = self._database_path()
        path_key = str(path.resolve())
        connection = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            with self._lock:
                if path_key not in self._initialized_paths:
                    self._ensure_schema(connection)
                    self._recover_interrupted_jobs(connection)
                    self._initialized_paths.add(path_key)
                self._expire_terminal_jobs(connection)
                # Recovery/retention may have written state before callers begin
                # their own explicit IMMEDIATE transaction.
                connection.commit()
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
            CREATE TABLE IF NOT EXISTS dataset_jobs (
                job_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                owner_user_id TEXT,
                owner_username TEXT,
                status TEXT NOT NULL,
                progress REAL,
                message TEXT,
                error_code TEXT,
                error_message TEXT,
                artifact_id TEXT,
                idempotency_key TEXT,
                request_fingerprint TEXT,
                request_json TEXT,
                result_json TEXT,
                requested_start TEXT,
                requested_end TEXT,
                expected_dataset_version TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_dataset_jobs_status
                ON dataset_jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS ix_dataset_jobs_owner
                ON dataset_jobs(owner_user_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_dataset_jobs_artifact
                ON dataset_jobs(artifact_id);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_dataset_jobs_owner_idempotency
                ON dataset_jobs(owner_user_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS dataset_job_events (
                job_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                stage TEXT NOT NULL,
                percent REAL NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (job_id, sequence),
                FOREIGN KEY(job_id) REFERENCES dataset_jobs(job_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_dataset_job_events_job_sequence
                ON dataset_job_events(job_id, sequence);
            """
        )

    @staticmethod
    def _recover_interrupted_jobs(connection: sqlite3.Connection) -> None:
        """Fail only stale queued/running jobs left behind by a prior process."""
        now = _iso()
        stale_before = _utc_now() - timedelta(minutes=_recovery_stale_minutes())
        rows = connection.execute(
            """
            SELECT job_id, updated_at, created_at
            FROM dataset_jobs
            WHERE status IN ('pending', 'running')
            """
        ).fetchall()
        for row in rows:
            last_activity = (
                _parse_datetime(row["updated_at"])
                or _parse_datetime(row["created_at"])
            )
            if last_activity is not None and last_activity > stale_before:
                continue
            job_id = str(row["job_id"])
            connection.execute(
                """
                UPDATE dataset_jobs
                SET status = 'failed', progress = 100, message = ?,
                    error_code = ?, error_message = ?, updated_at = ?, completed_at = ?
                WHERE job_id = ?
                """,
                (
                    "Dataset job interrupted by backend restart",
                    "DATASET_JOB_INTERRUPTED",
                    "Dataset job interrupted by backend restart",
                    now,
                    now,
                    job_id,
                ),
            )
            JobService._append_event_connection(
                connection,
                job_id=job_id,
                stage="interrupted",
                percent=100,
                message="Dataset job interrupted by backend restart",
                created_at=now,
            )

    def _expire_terminal_jobs(self, connection: sqlite3.Connection) -> None:
        """Mark terminal jobs expired while retaining their status and artifact reference."""
        cutoff = _utc_now() - timedelta(hours=_job_ttl_hours())
        rows = connection.execute(
            """
            SELECT job_id, completed_at, updated_at, created_at
            FROM dataset_jobs
            WHERE status IN ('completed', 'failed', 'cancelled')
            """
        ).fetchall()
        expired_ids = [
            str(row["job_id"])
            for row in rows
            if (_parse_datetime(row["completed_at"])
                or _parse_datetime(row["updated_at"])
                or _parse_datetime(row["created_at"])
                or _utc_now()) <= cutoff
        ]
        if not expired_ids:
            return
        now = _iso()
        for job_id in expired_ids:
            connection.execute(
                """
                UPDATE dataset_jobs
                SET status = 'expired', progress = 100, message = ?,
                    idempotency_key = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                ("Dataset job retention expired", now, job_id),
            )
            self._append_event_connection(
                connection,
                job_id=job_id,
                stage="expired",
                percent=100,
                message="Dataset job retention expired",
                created_at=now,
            )
            self._jobs.pop(job_id, None)

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobState:
        return JobState(
            job_id=str(row["job_id"]),
            operation=str(row["operation"]),
            owner_user_id=row["owner_user_id"],
            owner_username=row["owner_username"],
            status=str(row["status"]),
            progress=float(row["progress"]) if row["progress"] is not None else None,
            message=row["message"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            artifact_id=row["artifact_id"],
            idempotency_key=row["idempotency_key"],
            request_fingerprint=row["request_fingerprint"],
            requested_start=_parse_datetime(row["requested_start"]),
            requested_end=_parse_datetime(row["requested_end"]),
            expected_dataset_version=row["expected_dataset_version"],
            result=_json_load(row["result_json"], None),
            cancel_requested=bool(row["cancel_requested"]),
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
        )

    def _remember(self, job: JobState) -> JobState:
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> JobEvent:
        return JobEvent(
            job_id=str(row["job_id"]),
            sequence=int(row["sequence"]),
            stage=str(row["stage"]),
            percent=float(row["percent"]),
            message=str(row["message"]),
            created_at=_parse_datetime(row["created_at"]) or _utc_now(),
        )

    def _insert_job(
        self,
        connection: sqlite3.Connection,
        *,
        operation: str,
        message: str | None,
        owner_user_id: str | None,
        owner_username: str | None,
        idempotency_key: str | None,
        request_fingerprint: str | None,
        request_payload: dict[str, Any] | None,
        requested_start: datetime | None,
        requested_end: datetime | None,
        expected_dataset_version: str | None,
    ) -> JobState:
        now = _iso()
        job = JobState(
            job_id=uuid.uuid4().hex,
            operation=operation,
            owner_user_id=owner_user_id,
            owner_username=owner_username,
            status="pending",
            progress=0.0,
            message=message,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            requested_start=requested_start,
            requested_end=requested_end,
            expected_dataset_version=expected_dataset_version,
            created_at=_parse_datetime(now),
            updated_at=_parse_datetime(now),
        )
        connection.execute(
            """
            INSERT INTO dataset_jobs (
                job_id, operation, owner_user_id, owner_username, status, progress,
                message, idempotency_key, request_fingerprint, request_json,
                requested_start, requested_end, expected_dataset_version,
                cancel_requested, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                job.job_id,
                job.operation,
                job.owner_user_id,
                job.owner_username,
                job.status,
                job.progress,
                job.message,
                job.idempotency_key,
                job.request_fingerprint,
                _json_dump(request_payload),
                _iso(job.requested_start) if job.requested_start else None,
                _iso(job.requested_end) if job.requested_end else None,
                job.expected_dataset_version,
                now,
                now,
            ),
        )
        self._append_event_connection(
            connection,
            job_id=job.job_id,
            stage="queued",
            percent=0,
            message=message or "Dataset job queued",
            created_at=now,
        )
        return self._remember(job)

    def create_job(
        self,
        message: str | None = None,
        *,
        operation: str = "legacy_refresh",
        owner_user_id: str | None = None,
        owner_username: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        request_payload: dict[str, Any] | None = None,
        requested_start: datetime | None = None,
        requested_end: datetime | None = None,
        expected_dataset_version: str | None = None,
    ) -> JobState:
        """Create one job without applying the static-mutation conflict policy.

        This preserves the legacy refresh route.  New static job requests should
        use :meth:`create_or_get_dataset_job` below.
        """
        with self._connect() as connection:
            return self._insert_job(
                connection,
                operation=operation,
                message=message,
                owner_user_id=owner_user_id,
                owner_username=owner_username,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                request_payload=request_payload,
                requested_start=requested_start,
                requested_end=requested_end,
                expected_dataset_version=expected_dataset_version,
            )

    def create_or_get_dataset_job(
        self,
        *,
        operation: str,
        owner_user_id: str,
        owner_username: str | None,
        idempotency_key: str,
        request_fingerprint: str,
        request_payload: dict[str, Any],
        requested_start: datetime | None,
        requested_end: datetime | None,
        expected_dataset_version: str | None,
        message: str = "Dataset job queued",
    ) -> tuple[JobState, bool]:
        """Atomically enforce idempotency and the one-mutating-job policy."""
        if operation not in _MUTATING_OPERATIONS:
            raise ValueError(f"Unsupported dataset job operation: {operation}")
        key = str(idempotency_key or "").strip()
        if not key or len(key) > 255:
            raise ApiError("INVALID_IDEMPOTENCY_KEY", "A valid Idempotency-Key is required.", 422)

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_row = connection.execute(
                """
                SELECT * FROM dataset_jobs
                WHERE owner_user_id = ? AND idempotency_key = ?
                """,
                (owner_user_id, key),
            ).fetchone()
            if existing_row is not None:
                existing = self._remember(self._job_from_row(existing_row))
                if existing.request_fingerprint and existing.request_fingerprint != request_fingerprint:
                    raise ApiError(
                        "IDEMPOTENCY_KEY_REUSED",
                        "Idempotency-Key was already used for a different dataset request.",
                        409,
                    )
                return existing, True

            active = connection.execute(
                """
                SELECT job_id FROM dataset_jobs
                WHERE operation IN ('build_range', 'extend', 'override')
                  AND status IN ('pending', 'running')
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if active is not None:
                raise ApiError(
                    "DATASET_JOB_ALREADY_RUNNING",
                    "Another static dataset mutation is already running.",
                    409,
                )

            job = self._insert_job(
                connection,
                operation=operation,
                message=message,
                owner_user_id=owner_user_id,
                owner_username=owner_username,
                idempotency_key=key,
                request_fingerprint=request_fingerprint,
                request_payload=request_payload,
                requested_start=requested_start,
                requested_end=requested_end,
                expected_dataset_version=expected_dataset_version,
            )
            return job, False

    def get_job(self, job_id: str) -> JobState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._remember(self._job_from_row(row)) if row is not None else None

    def list_jobs(self, *, limit: int = 500, offset: int = 0) -> list[JobState]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dataset_jobs
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (max(1, min(500, int(limit))), max(0, int(offset))),
            ).fetchall()
        return [self._remember(self._job_from_row(row)) for row in rows]

    def find_by_artifact(self, artifact_id: str) -> JobState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_jobs WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        return self._remember(self._job_from_row(row)) if row is not None else None

    @staticmethod
    def _append_event_connection(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        stage: str,
        percent: float,
        message: str,
        created_at: str | None = None,
    ) -> int:
        next_sequence = int(
            connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM dataset_job_events WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO dataset_job_events (job_id, sequence, stage, percent, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                next_sequence,
                str(stage)[:80],
                min(100.0, max(0.0, float(percent))),
                str(message)[:1000],
                created_at or _iso(),
            ),
        )
        return next_sequence

    def append_event(
        self,
        job_id: str,
        *,
        stage: str,
        percent: float,
        message: str,
    ) -> JobEvent | None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM dataset_jobs WHERE job_id = ?", (job_id,)
            ).fetchone() is None:
                return None
            sequence = self._append_event_connection(
                connection,
                job_id=job_id,
                stage=stage,
                percent=percent,
                message=message,
            )
            now = _iso()
            connection.execute(
                "UPDATE dataset_jobs SET updated_at = ? WHERE job_id = ?", (now, job_id)
            )
            row = connection.execute(
                """
                SELECT * FROM dataset_job_events
                WHERE job_id = ? AND sequence = ?
                """,
                (job_id, sequence),
            ).fetchone()
        return self._event_from_row(row) if row is not None else None

    def get_events(self, job_id: str, *, after: int = 0, limit: int = 500) -> list[JobEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dataset_job_events
                WHERE job_id = ? AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (job_id, max(0, int(after)), max(1, min(500, int(limit)))),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def update_job(self, job_id: str, **updates: Any) -> JobState | None:
        """Update an allowed job field and persist terminal timestamps."""
        allowed = {
            "status",
            "progress",
            "message",
            "error_code",
            "error_message",
            "artifact_id",
            "result",
            "cancel_requested",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return self.get_job(job_id)

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM dataset_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            current = self._job_from_row(row)
            if "status" in values and values["status"] not in JOB_STATUSES:
                raise ValueError("Invalid job status")
            if "progress" in values and values["progress"] is not None:
                values["progress"] = min(100.0, max(0.0, float(values["progress"])))
            if "cancel_requested" in values:
                values["cancel_requested"] = int(bool(values["cancel_requested"]))
            if "result" in values:
                values["result_json"] = _json_dump(values.pop("result"))
            now = _iso()
            if values.get("status", current.status) in _TERMINAL_JOB_STATUSES and current.completed_at is None:
                values["completed_at"] = now
            values["updated_at"] = now
            assignments = ", ".join(f"{column} = ?" for column in values)
            connection.execute(
                f"UPDATE dataset_jobs SET {assignments} WHERE job_id = ?",
                [*values.values(), job_id],
            )
            updated_row = connection.execute(
                "SELECT * FROM dataset_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._remember(self._job_from_row(updated_row)) if updated_row else None

    def request_cancel(self, job_id: str) -> JobState | None:
        """Request cooperative cancellation; pending jobs are cancelled immediately."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM dataset_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            job = self._job_from_row(row)
            if job.status in _TERMINAL_JOB_STATUSES:
                return job
            now = _iso()
            if job.status == "pending":
                connection.execute(
                    """
                    UPDATE dataset_jobs
                    SET status = 'cancelled', progress = 100, cancel_requested = 1,
                        message = ?, updated_at = ?, completed_at = ?
                    WHERE job_id = ?
                    """,
                    ("Dataset job cancelled", now, now, job_id),
                )
                self._append_event_connection(
                    connection,
                    job_id=job_id,
                    stage="cancelled",
                    percent=100,
                    message="Dataset job cancelled before execution",
                    created_at=now,
                )
            else:
                connection.execute(
                    """
                    UPDATE dataset_jobs
                    SET cancel_requested = 1, message = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    ("Cancellation requested", now, job_id),
                )
                self._append_event_connection(
                    connection,
                    job_id=job_id,
                    stage="cancellation_requested",
                    percent=job.progress or 0,
                    message="Cancellation will be applied at the next safe stage",
                    created_at=now,
                )
            updated = connection.execute(
                "SELECT * FROM dataset_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._remember(self._job_from_row(updated)) if updated else None

    def is_cancel_requested(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        return bool(job and job.cancel_requested)

    @contextmanager
    def canonical_mutation_lock(self) -> Iterator[None]:
        """Serialize final canonical-dataset promotion in this worker process."""
        with self._canonical_mutation_lock:
            yield

    def run_background(self, job: JobState, fn: Callable[[JobState], None]) -> None:
        """Submit work to the bounded executor and persist every terminal state."""

        def _wrapper() -> None:
            current = self.get_job(job.job_id)
            if current is None or current.status == "cancelled" or current.cancel_requested:
                return
            self.update_job(job.job_id, status="running", progress=0, message="Running")
            self.append_event(
                job.job_id,
                stage="started",
                percent=0,
                message="Dataset job started",
            )
            try:
                fn(job)
                current = self.get_job(job.job_id)
                if current is None:
                    return
                if current.cancel_requested:
                    self.update_job(
                        job.job_id,
                        status="cancelled",
                        progress=100,
                        message="Dataset job cancelled",
                    )
                    self.append_event(
                        job.job_id,
                        stage="cancelled",
                        percent=100,
                        message="Dataset job cancelled",
                    )
                elif current.status == "running":
                    self.update_job(
                        job.job_id,
                        status="completed",
                        progress=100,
                        message="Dataset job completed",
                    )
                    self.append_event(
                        job.job_id,
                        stage="completed",
                        percent=100,
                        message="Dataset job completed",
                    )
            except ApiError as exc:
                if exc.code == "DATASET_JOB_CANCELLED":
                    self.update_job(
                        job.job_id,
                        status="cancelled",
                        progress=100,
                        error_code=None,
                        error_message=None,
                        message="Dataset job cancelled",
                    )
                    self.append_event(
                        job.job_id,
                        stage="cancelled",
                        percent=100,
                        message="Dataset job cancelled",
                    )
                else:
                    self.update_job(
                        job.job_id,
                        status="failed",
                        progress=100,
                        error_code=exc.code,
                        error_message=exc.message,
                        message="Dataset job failed",
                    )
                    self.append_event(
                        job.job_id,
                        stage="failed",
                        percent=100,
                        message="Dataset job failed",
                    )
            except Exception:
                self.update_job(
                    job.job_id,
                    status="failed",
                    progress=100,
                    error_code="DATASET_JOB_FAILED",
                    error_message="Dataset job failed",
                    message="Dataset job failed",
                )
                self.append_event(
                    job.job_id,
                    stage="failed",
                    percent=100,
                    message="Dataset job failed",
                )
            finally:
                with self._lock:
                    self._futures.pop(job.job_id, None)

        with self._lock:
            self._futures[job.job_id] = self._executor.submit(_wrapper)

    def shutdown(self, *, wait: bool = True) -> None:
        """Release worker resources in explicit application/test shutdown paths."""
        self._executor.shutdown(wait=wait, cancel_futures=False)


job_service = JobService()
