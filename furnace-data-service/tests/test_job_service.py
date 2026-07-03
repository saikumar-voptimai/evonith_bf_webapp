"""Tests for the Phase 4 in-process job registry."""

from app.services.job_service import JobService


def test_job_service_create_and_update_job():
    jobs = JobService()
    job = jobs.create_job("Queued")

    updated = jobs.update_job(job.job_id, status="failed", error_code="DATASET_JOB_FAILED")

    assert updated is not None
    assert updated.status == "failed"
    assert updated.error_code == "DATASET_JOB_FAILED"
    assert updated.completed_at is not None
