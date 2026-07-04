"""In-process compute job registry for Phase 7 workflows."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class ComputeJobState:
    job_id: str
    workflow: str
    status: str = "pending"
    progress: float | None = None
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None
    artifact_id: str | None = None
    download_url: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class ComputeJobService:
    """Tiny, testable in-process job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, ComputeJobState] = {}
        self._lock = threading.Lock()

    def create_job(self, workflow: str, message: str | None = None) -> ComputeJobState:
        now = datetime.now(timezone.utc)
        job = ComputeJobState(
            job_id=uuid.uuid4().hex,
            workflow=workflow,
            status="pending",
            message=message,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> ComputeJobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, **updates: Any) -> ComputeJobState | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc)
            if job.status in {"completed", "failed", "cancelled", "expired"}:
                job.completed_at = job.completed_at or job.updated_at
            return job

    def run_inline(
        self,
        *,
        workflow: str,
        fn: Callable[[], dict[str, Any]],
        message: str | None = None,
    ) -> ComputeJobState:
        job = self.create_job(workflow, message=message)
        self.update_job(job.job_id, status="running", progress=0.0, message="Running")
        try:
            result = fn()
        except Exception as exc:
            self.update_job(
                job.job_id,
                status="failed",
                progress=1.0,
                error_code="COMPUTE_JOB_FAILED",
                error_message=str(exc),
                message="Compute job failed",
            )
        else:
            artifact_id = None
            download_url = None
            artifacts = result.get("artifacts") if isinstance(result, dict) else None
            if artifacts:
                first = artifacts[0]
                artifact_id = first.get("artifact_id")
                download_url = first.get("download_url")
            self.update_job(
                job.job_id,
                status="completed",
                progress=1.0,
                result=result,
                artifact_id=artifact_id,
                download_url=download_url,
                message="Completed",
            )
        current = self.get_job(job.job_id)
        return current or job

    @staticmethod
    def response(job: ComputeJobState) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "artifact_id": job.artifact_id,
            "download_url": job.download_url,
        }

    @staticmethod
    def status(job: ComputeJobState) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "workflow": job.workflow,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "result": job.result,
            "artifact_id": job.artifact_id,
            "download_url": job.download_url,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
        }


compute_job_service = ComputeJobService()
