"""Safe adapter over existing in-process job registries."""

from __future__ import annotations

from typing import Any

from app.core.errors import ApiError
from app.services.compute_job_service import compute_job_service
from app.services.job_service import job_service
from app.services.redaction_service import redact_value


class UnifiedJobService:
    """Expose Phase 4 and Phase 7-9 job metadata through one read API."""

    def list_jobs(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        items = [*self._dataset_jobs(), *self._compute_jobs()]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        start = max(0, int(offset))
        end = start + min(500, max(1, int(limit)))
        return {"items": items[start:end], "total": len(items), "limit": end - start, "offset": start}

    def get_job(self, job_id: str) -> dict[str, Any]:
        for item in [*self._dataset_jobs(), *self._compute_jobs()]:
            if item["job_id"] == job_id:
                return item
        raise ApiError("JOB_NOT_FOUND", "Job not found.", 404)

    @staticmethod
    def _dataset_jobs() -> list[dict[str, Any]]:
        with job_service._lock:  # noqa: SLF001 - explicit adapter over existing registry.
            jobs = list(job_service._jobs.values())  # noqa: SLF001
        return [
            {
                "job_id": job.job_id,
                "source": "datasets",
                "workflow": "dataset_refresh",
                "status": job.status,
                "progress": job.progress,
                "message": redact_value(job.message),
                "error_code": job.error_code,
                "error_message": redact_value(job.error_message),
                "artifact_id": job.artifact_id,
                "download_url": None,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "completed_at": job.completed_at,
            }
            for job in jobs
        ]

    @staticmethod
    def _compute_jobs() -> list[dict[str, Any]]:
        with compute_job_service._lock:  # noqa: SLF001 - explicit adapter over existing registry.
            jobs = list(compute_job_service._jobs.values())  # noqa: SLF001
        return [
            {
                "job_id": job.job_id,
                "source": "compute",
                "workflow": job.workflow,
                "status": job.status,
                "progress": job.progress,
                "message": redact_value(job.message),
                "error_code": job.error_code,
                "error_message": redact_value(job.error_message),
                "artifact_id": job.artifact_id,
                "download_url": job.download_url,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "completed_at": job.completed_at,
            }
            for job in jobs
        ]

