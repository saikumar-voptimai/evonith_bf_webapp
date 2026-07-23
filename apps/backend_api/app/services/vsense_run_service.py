"""Persistent V-Sense run orchestration using the common job worker."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.vsense_repository import VSenseRepository, fingerprint
from apps.backend_api.app.services.job_service import JobService, JobState, job_service
from apps.backend_api.app.services.vsense_context_service import VSenseContextService
from furnace_data.vsense.bounds import VSenseValidationError
from furnace_data.vsense.optimizer import run_legacy_optimization
from furnace_data.vsense.review import unavailable_review


_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


class VSenseRunService:
    """Create, monitor, and cancel V-Sense optimization runs."""

    def __init__(
        self,
        *,
        repository: VSenseRepository,
        context_service: VSenseContextService,
        settings: Any | None = None,
        jobs: JobService | None = None,
        audit_service: Any | None = None,
    ) -> None:
        self.repository = repository
        self.context_service = context_service
        self.settings = settings
        self.jobs = jobs or job_service
        self.audit_service = audit_service

    def create_run(
        self,
        request: dict[str, Any],
        *,
        current_user: dict[str, Any],
        idempotency_key: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        owner_user_id = _user_id(current_user)
        payload_fingerprint = fingerprint(request)
        replay = self.repository.get_idempotent_response(
            owner_user_id=owner_user_id,
            scope="run",
            idempotency_key=idempotency_key,
            request_fingerprint=payload_fingerprint,
        )
        if replay is not None:
            replay["idempotent_replay"] = True
            return replay

        context = self.context_service.get_context_for_run(
            str(request.get("context_id") or ""),
            current_user=current_user,
        )
        if str(request.get("optimization_type_id") or "") != str(
            context.get("optimization_type_id")
        ):
            raise ApiError(
                "VSENSE_INVALID_OPTIMIZATION_TYPE",
                "Run optimization_type_id must match the context.",
                status_code=400,
            )
        self._raise_if_active_run(owner_user_id)
        job = self.jobs.create_job(
            "V-Sense run queued",
            operation="vsense_run",
            owner_user_id=owner_user_id,
            owner_username=current_user.get("username"),
            idempotency_key=idempotency_key,
            request_fingerprint=payload_fingerprint,
            request_payload=request,
        )
        self.repository.store_run_metadata(
            run_id=job.job_id,
            owner_user_id=owner_user_id,
            owner_username=current_user.get("username"),
            optimization_type_id=str(request["optimization_type_id"]),
            context_id=str(request["context_id"]),
            request_payload=request,
            created_at=_iso(job.created_at),
        )
        accepted = self._accepted_response(job, idempotent_replay=False)
        self.repository.store_idempotent_response(
            owner_user_id=owner_user_id,
            scope="run",
            idempotency_key=idempotency_key,
            request_fingerprint=payload_fingerprint,
            response=accepted,
        )
        self._audit(
            {
                "request_id": request_id,
                "actor_user_id": owner_user_id,
                "actor_username": current_user.get("username"),
                "event_type": "vsense.run.created",
                "resource_type": "vsense.run",
                "resource_id": job.job_id,
                "action": "create",
                "result": "success",
                "status_code": 202,
                "metadata": {
                    "context_id": request.get("context_id"),
                    "optimization_type_id": request.get("optimization_type_id"),
                    "dataset_version": context.get("dataset", {}).get("version"),
                    "algorithm_version": context.get("algorithm_version"),
                    "control_profile_version": context.get("control_profile", {}).get("version"),
                    "input_override_parameter_ids": [
                        item.get("parameter_id") for item in request.get("input_overrides") or []
                    ],
                },
            }
        )
        self.jobs.run_background(
            job,
            lambda state: self._execute_run(
                state,
                context=context,
                request=request,
                current_user=current_user,
            ),
        )
        return accepted

    def get_run(
        self,
        run_id: str,
        *,
        current_user: dict[str, Any],
    ) -> dict[str, Any]:
        job = self._job(run_id)
        metadata = self._metadata(run_id)
        self._enforce_run_access(metadata, current_user)
        return self._status_response(job, metadata)

    def get_events(
        self,
        run_id: str,
        *,
        current_user: dict[str, Any],
        after: int = 0,
    ) -> dict[str, Any]:
        self._enforce_run_access(self._metadata(run_id), current_user)
        events = self.jobs.get_events(run_id, after=after)
        return {
            "run_id": run_id,
            "events": [
                {
                    "sequence": event.sequence,
                    "stage": event.stage,
                    "progress": event.percent,
                    "message": event.message,
                    "created_at": _iso(event.created_at),
                }
                for event in events
            ],
        }

    def cancel_run(
        self,
        run_id: str,
        *,
        current_user: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        metadata = self._metadata(run_id)
        self._enforce_run_cancel(metadata, current_user)
        job = self._job(run_id)
        if job.status in _TERMINAL_STATUSES:
            raise ApiError(
                "VSENSE_RUN_NOT_CANCELLABLE",
                "V-Sense run is not cancellable.",
                status_code=409,
            )
        updated = self.jobs.request_cancel(run_id)
        if updated is None:
            raise ApiError("VSENSE_RUN_NOT_FOUND", "V-Sense run not found.", 404)
        self._audit(
            {
                "request_id": request_id,
                "actor_user_id": _user_id(current_user),
                "actor_username": current_user.get("username"),
                "event_type": "vsense.run.cancelled",
                "resource_type": "vsense.run",
                "resource_id": run_id,
                "action": "cancel",
                "result": "success",
                "status_code": 200,
                "metadata": {
                    "optimization_type_id": metadata["optimization_type_id"],
                    "context_id": metadata["context_id"],
                },
            }
        )
        return self._status_response(updated, metadata)

    def _execute_run(
        self,
        state: JobState,
        *,
        context: dict[str, Any],
        request: dict[str, Any],
        current_user: dict[str, Any],
    ) -> None:
        run_id = state.job_id
        self.jobs.append_event(
            run_id,
            stage="context_loaded",
            percent=10,
            message="V-Sense immutable context loaded",
        )
        options = dict(request.get("options") or {})

        def progress(
            iteration: int,
            best_objective: float,
            best_feasible: float | None,
            evaluations: int,
            elapsed_s: float,
        ) -> bool:
            if self.jobs.is_cancel_requested(run_id):
                return True
            max_iterations = max(1, int(options.get("max_iterations") or 20))
            percent = min(95.0, 10.0 + (float(iteration) / max_iterations) * 80.0)
            self.jobs.append_event(
                run_id,
                stage="optimizing",
                percent=percent,
                message=f"Generation {iteration} complete",
            )
            return False

        try:
            result = run_legacy_optimization(
                context=context,
                control_plan=list(request.get("control_plan") or []),
                input_overrides=list(request.get("input_overrides") or []),
                lambda_reg=float(options.get("lambda_reg", 0.05)),
                iteration_budget={
                    "max_iterations": int(options.get("max_iterations") or 20),
                    "population": int(options.get("population") or 6),
                    "tolerance": float(options.get("tolerance") or 0.01),
                    "polish": bool(options.get("polish", False)),
                },
                seed=int(options.get("seed") or self._setting("vsense_default_seed", 42)),
                require_approved_bounds=bool(
                    self._setting("vsense_require_approved_bounds", True)
                ),
                progress_callback=progress,
            )
            if bool(options.get("request_llm_review")):
                if self._review_allowed(current_user):
                    result["review"] = unavailable_review(
                        "LLM provider integration is not enabled in this deployment."
                    )
                else:
                    result["review"] = unavailable_review(
                        "LLM review unavailable for this user or deployment."
                    )
                    result.setdefault("warnings", []).append("V-Sense LLM review unavailable.")
            if self.jobs.is_cancel_requested(run_id):
                return
            self.jobs.update_job(
                run_id,
                status="completed",
                progress=100,
                message="V-Sense run completed",
                result=result,
            )
            self.jobs.append_event(
                run_id,
                stage="completed",
                percent=100,
                message="V-Sense advisory result persisted",
            )
            self._audit(
                {
                    "actor_user_id": state.owner_user_id,
                    "actor_username": state.owner_username,
                    "event_type": "vsense.run.completed",
                    "resource_type": "vsense.run",
                    "resource_id": run_id,
                    "action": "complete",
                    "result": "success",
                    "status_code": 200,
                    "metadata": {
                        "context_id": request.get("context_id"),
                        "optimization_type_id": request.get("optimization_type_id"),
                        "warning_count": len(result.get("warnings") or []),
                        "violation_count": len(result.get("feasibility", {}).get("violations") or []),
                    },
                }
            )
        except VSenseValidationError as exc:
            self.jobs.update_job(
                run_id,
                status="failed",
                progress=100,
                message="V-Sense run failed",
                error_code=exc.code,
                error_message=str(exc),
            )
            self.jobs.append_event(
                run_id,
                stage="failed",
                percent=100,
                message="V-Sense run failed validation",
            )
        except Exception:
            self.jobs.update_job(
                run_id,
                status="failed",
                progress=100,
                message="V-Sense run failed",
                error_code="VSENSE_OPTIMIZATION_FAILED",
                error_message="V-Sense optimization failed.",
            )
            self.jobs.append_event(
                run_id,
                stage="failed",
                percent=100,
                message="V-Sense optimization failed",
            )

    def _raise_if_active_run(self, owner_user_id: str) -> None:
        for job in self.jobs.list_jobs(limit=500):
            if (
                job.operation == "vsense_run"
                and job.owner_user_id == owner_user_id
                and job.status in {"pending", "running"}
            ):
                raise ApiError(
                    "VSENSE_RUN_ALREADY_ACTIVE",
                    "Another V-Sense run is already active for this user.",
                    status_code=409,
                    details={"run_id": job.job_id},
                )

    def _job(self, run_id: str) -> JobState:
        job = self.jobs.get_job(run_id)
        if job is None or job.operation != "vsense_run":
            raise ApiError("VSENSE_RUN_NOT_FOUND", "V-Sense run not found.", 404)
        return job

    def _metadata(self, run_id: str) -> dict[str, Any]:
        metadata = self.repository.get_run_metadata(run_id)
        if metadata is None:
            raise ApiError("VSENSE_RUN_NOT_FOUND", "V-Sense run not found.", 404)
        return metadata

    @staticmethod
    def _enforce_run_access(
        metadata: dict[str, Any],
        current_user: dict[str, Any],
    ) -> None:
        permissions = set(current_user.get("permissions") or [])
        if metadata["owner_user_id"] != _user_id(current_user) and "vsense:runs:read:any" not in permissions:
            raise ApiError("FORBIDDEN", "Insufficient permissions.", 403)

    @staticmethod
    def _enforce_run_cancel(
        metadata: dict[str, Any],
        current_user: dict[str, Any],
    ) -> None:
        permissions = set(current_user.get("permissions") or [])
        if metadata["owner_user_id"] != _user_id(current_user) and "vsense:runs:cancel:any" not in permissions:
            raise ApiError("FORBIDDEN", "Insufficient permissions.", 403)

    def _status_response(self, job: JobState, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": job.job_id,
            "context_id": metadata["context_id"],
            "optimization_type_id": metadata["optimization_type_id"],
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "cancellable": job.status not in _TERMINAL_STATUSES,
            "created_at": _iso(job.created_at),
            "updated_at": _iso(job.updated_at),
            "completed_at": _iso(job.completed_at) if job.completed_at else None,
            "result": job.result,
        }

    @staticmethod
    def _accepted_response(job: JobState, *, idempotent_replay: bool) -> dict[str, Any]:
        return {
            "run_id": job.job_id,
            "status": job.status,
            "created_at": _iso(job.created_at),
            "status_url": f"/vsense/runs/{job.job_id}",
            "events_url": f"/vsense/runs/{job.job_id}/events",
            "cancellable": True,
            "idempotent_replay": idempotent_replay,
        }

    def _review_allowed(self, current_user: dict[str, Any]) -> bool:
        return bool(self._setting("vsense_llm_enabled", False)) and "vsense:llm" in set(
            current_user.get("permissions") or []
        )

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default)

    def _audit(self, payload: dict[str, Any]) -> None:
        if self.audit_service is not None:
            self.audit_service.record_event(payload)


def _user_id(current_user: dict[str, Any]) -> str:
    return str(current_user.get("id") or current_user.get("username") or "unknown")


def _iso(value: datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
