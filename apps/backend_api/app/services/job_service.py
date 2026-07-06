"""Small in-process job registry for Phase 4 dataset API jobs."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


JOB_STATUSES = {"pending", "running", "completed", "failed", "cancelled", "expired"}


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


class JobService:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create_job(self, message: str | None = None) -> JobState:
        now = datetime.now(timezone.utc)
        job = JobState(
            job_id=uuid.uuid4().hex,
            status="pending",
            message=message,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **updates) -> JobState | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)
        if job.status in {"completed", "failed", "cancelled", "expired"} and job.completed_at is None:
            job.completed_at = job.updated_at
        return job

    def run_background(self, job: JobState, fn: Callable[[JobState], None]) -> None:
        def _wrapper() -> None:
            self.update_job(job.job_id, status="running", progress=0.0, message="Running")
            try:
                fn(job)
                current = self.get_job(job.job_id)
                if current and current.status == "running":
                    self.update_job(job.job_id, status="completed", progress=1.0, message="Completed")
            except Exception as exc:
                self.update_job(
                    job.job_id,
                    status="failed",
                    error_code="DATASET_REFRESH_FAILED",
                    error_message=str(exc),
                    message="Dataset refresh failed",
                )

        threading.Thread(target=_wrapper, name=f"dataset-job-{job.job_id[:8]}", daemon=True).start()


job_service = JobService()
