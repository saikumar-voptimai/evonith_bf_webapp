"""Persistence, recovery, and retention tests for durable dataset jobs."""

from __future__ import annotations

import sqlite3

from apps.backend_api.app.services.job_service import JobService


def _shutdown(*services: JobService) -> None:
    for service in services:
        service.shutdown(wait=False)


def test_job_service_create_update_and_persist(tmp_path):
    db_path = tmp_path / "dataset_jobs.sqlite"
    jobs = JobService(db_path, max_workers=1)
    try:
        job = jobs.create_job("Queued", owner_user_id="operator-1")
        updated = jobs.update_job(job.job_id, status="failed", error_code="DATASET_JOB_FAILED")
        reopened = JobService(db_path, max_workers=1)
        try:
            persisted = reopened.get_job(job.job_id)
        finally:
            reopened.shutdown(wait=False)

        assert updated is not None
        assert updated.status == "failed"
        assert updated.error_code == "DATASET_JOB_FAILED"
        assert updated.completed_at is not None
        assert persisted is not None
        assert persisted.owner_user_id == "operator-1"
        assert persisted.status == "failed"
    finally:
        jobs.shutdown(wait=False)


def test_recovery_keeps_fresh_running_job_for_another_live_process(tmp_path):
    db_path = tmp_path / "dataset_jobs.sqlite"
    first = JobService(db_path, max_workers=1)
    second = None
    try:
        job = first.create_job("Queued", owner_user_id="operator-1")
        first.update_job(job.job_id, status="running", message="Running")

        second = JobService(db_path, max_workers=1)
        recovered = second.get_job(job.job_id)

        assert recovered is not None
        assert recovered.status == "running"
    finally:
        _shutdown(*(service for service in (first, second) if service is not None))


def test_recovery_marks_stale_pending_or_running_job_interrupted(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASET_JOB_RECOVERY_STALE_MINUTES", "1")
    db_path = tmp_path / "dataset_jobs.sqlite"
    first = JobService(db_path, max_workers=1)
    second = None
    try:
        job = first.create_job("Queued", owner_user_id="operator-1")
        first.update_job(job.job_id, status="running", message="Running")
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                "UPDATE dataset_jobs SET updated_at = ? WHERE job_id = ?",
                ("2000-01-01T00:00:00+00:00", job.job_id),
            )

        second = JobService(db_path, max_workers=1)
        recovered = second.get_job(job.job_id)
        events = second.get_events(job.job_id)

        assert recovered is not None
        assert recovered.status == "failed"
        assert recovered.error_code == "DATASET_JOB_INTERRUPTED"
        assert events[-1].stage == "interrupted"
    finally:
        _shutdown(*(service for service in (first, second) if service is not None))


def test_ttl_marks_job_expired_but_retains_download_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASET_JOB_TTL_HOURS", "1")
    db_path = tmp_path / "dataset_jobs.sqlite"
    first = JobService(db_path, max_workers=1)
    second = None
    try:
        job = first.create_job(
            "Queued",
            owner_user_id="operator-1",
            idempotency_key="old-key",
        )
        first.update_job(job.job_id, status="completed", artifact_id="a" * 32)
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                UPDATE dataset_jobs
                SET completed_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", job.job_id),
            )

        second = JobService(db_path, max_workers=1)
        replacement, replayed = second.create_or_get_dataset_job(
            operation="build_range",
            owner_user_id="operator-1",
            owner_username=None,
            idempotency_key="old-key",
            request_fingerprint="replacement-request",
            request_payload={"operation": "build_range"},
            requested_start=None,
            requested_end=None,
            expected_dataset_version=None,
        )
        expired = second.get_job(job.job_id)
        events = second.get_events(job.job_id)

        assert expired is not None
        assert expired.status == "expired"
        assert expired.artifact_id == "a" * 32
        assert events[-1].stage == "expired"
        assert replacement.job_id != job.job_id
        assert replayed is False
    finally:
        _shutdown(*(service for service in (first, second) if service is not None))