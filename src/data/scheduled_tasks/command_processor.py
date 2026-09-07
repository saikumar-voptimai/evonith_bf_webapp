"""One-shot Jetson processor for queued scheduled-job control requests.

The processor claims a bounded number of commands, delegates lifecycle changes
to the existing provisioning coordinator, records outcomes, and exits. It is
designed for a systemd timer and is not a continuously running scheduler daemon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from furnace_data.relational import ScheduledJobStatus

from .control_service import (
    ScheduledJobCommandService,
    ScheduledJobCommandView,
)
from .execution_service import safe_scheduled_job_error_message
from .provisioning_service import ScheduledJobProvisioningService
from .service import (
    ScheduledJobService,
    ScheduledJobView,
)

LOGGER = logging.getLogger("furnacemind.scheduled_job_command_processor")


class ScheduledJobCommandStateError(RuntimeError):
    """Raised when a queued command is stale or no longer applicable."""


@dataclass(frozen=True, slots=True)
class ScheduledJobCommandBatchResult:
    """Bounded processing summary suitable for logs and CLI JSON output."""

    target_device_id: str
    claimed: int
    succeeded: int
    retrying: int
    failed: int

    @property
    def clean(self) -> bool:
        """Return whether every claimed command completed successfully."""

        return self.retrying == 0 and self.failed == 0


class ScheduledJobCommandProcessor:
    """Execute allow-listed queued commands through the provisioning service."""

    def __init__(
        self,
        *,
        command_service: ScheduledJobCommandService,
        job_service: ScheduledJobService,
        provisioning_service: ScheduledJobProvisioningService,
        target_device_id: str,
        worker_id: str,
        lease_seconds: int = 300,
        retry_delay_seconds: int = 30,
    ) -> None:
        """Create a bounded command processor for one deployment target."""

        target = target_device_id.strip()
        worker = worker_id.strip()
        if not target or len(target) > 128:
            raise ValueError("target_device_id must be 1 to 128 characters.")
        if not worker or len(worker) > 128:
            raise ValueError("worker_id must be 1 to 128 characters.")
        if not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 5 and 3600.")
        if not 1 <= retry_delay_seconds <= 3600:
            raise ValueError("retry_delay_seconds must be between 1 and 3600.")
        self._commands = command_service
        self._jobs = job_service
        self._provisioner = provisioning_service
        self._target_device_id = target
        self._worker_id = worker
        self._lease_seconds = lease_seconds
        self._retry_delay_seconds = retry_delay_seconds

    def process_batch(
        self, *, max_commands: int = 25
    ) -> ScheduledJobCommandBatchResult:
        """Process at most ``max_commands`` due requests and then return."""

        if not 1 <= max_commands <= 100:
            raise ValueError("max_commands must be between 1 and 100.")
        claimed = succeeded = retrying = failed = 0
        for _ in range(max_commands):
            command = self._commands.claim_next(
                target_device_id=self._target_device_id,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            if command is None:
                break
            claimed += 1
            try:
                status = self._execute(command)
                self._commands.mark_succeeded(
                    command,
                    resulting_job_status=status,
                )
            except ScheduledJobCommandStateError as exc:
                failed += 1
                self._record_failure(command, exc, retry=False)
            except Exception as exc:
                settled = self._record_failure(command, exc, retry=True)
                if settled == "pending":
                    retrying += 1
                else:
                    failed += 1
            else:
                succeeded += 1
        return ScheduledJobCommandBatchResult(
            target_device_id=self._target_device_id,
            claimed=claimed,
            succeeded=succeeded,
            retrying=retrying,
            failed=failed,
        )

    def _record_failure(
        self,
        command: ScheduledJobCommandView,
        error: BaseException,
        *,
        retry: bool,
    ) -> str:
        """Persist a safe failure and return its resulting command status."""

        safe_error = safe_scheduled_job_error_message(error)
        LOGGER.error(
            "Scheduled-job command %s (%s) failed: %s",
            command.command_id,
            command.action,
            safe_error,
        )
        settled = self._commands.mark_failed_or_retry(
            command,
            error=safe_error,
            retry_delay_seconds=self._retry_delay_seconds,
            retry=retry,
        )
        return settled.status

    def _execute(self, command: ScheduledJobCommandView) -> str:
        """Execute one claimed command and return the resulting job status."""

        job = self._jobs.get_job(command.job_id)
        if job is None:
            raise ScheduledJobCommandStateError("The scheduled job no longer exists.")
        if job.target_device_id != self._target_device_id:
            raise ScheduledJobCommandStateError(
                "The command does not belong to this target device."
            )
        if job.created_by_user_id != command.requested_by_user_id:
            raise ScheduledJobCommandStateError(
                "The command requester no longer owns the scheduled job."
            )

        desired_status = {
            "provision": ScheduledJobStatus.ACTIVE.value,
            "pause": ScheduledJobStatus.PAUSED.value,
            "resume": ScheduledJobStatus.ACTIVE.value,
            "archive": ScheduledJobStatus.DELETED.value,
        }.get(command.action)
        if desired_status is not None and job.status == desired_status:
            return job.status

        if command.action == "update":
            return self._apply_update(command, job)
        if job.updated_at != command.expected_job_updated_at:
            raise ScheduledJobCommandStateError(
                "The scheduled job changed after this command was requested."
            )

        if command.action == "provision":
            return self._provisioner.provision(job.job_id).job.status
        if command.action == "pause":
            return self._provisioner.pause(job.job_id).job.status
        if command.action == "resume":
            return self._provisioner.resume(job.job_id).job.status
        if command.action == "archive":
            return self._provisioner.archive(job.job_id).job.status
        raise ScheduledJobCommandStateError(
            f"Unsupported scheduled-job command action: {command.action}"
        )

    def _apply_update(
        self,
        command: ScheduledJobCommandView,
        original: ScheduledJobView,
    ) -> str:
        """Safely update an inactive job or pause/update/resume an active job."""

        definition = command.definition
        if definition is None:
            raise ScheduledJobCommandStateError(
                "An update command is missing its proposed definition."
            )
        target = definition.get("target_device")
        proposed_target = target.get("device_id") if isinstance(target, dict) else None
        if proposed_target != original.target_device_id:
            raise ScheduledJobCommandStateError(
                "A scheduled job cannot be moved to another device by editing it."
            )

        was_active = command.requested_job_status == ScheduledJobStatus.ACTIVE.value
        if original.definition == definition:
            if not was_active or original.status == ScheduledJobStatus.ACTIVE.value:
                return original.status
            if original.status == ScheduledJobStatus.PAUSED.value:
                return self._provisioner.resume(original.job_id).job.status
            if original.status == ScheduledJobStatus.PROVISIONING_FAILED.value:
                return self._provisioner.provision(original.job_id).job.status
            raise ScheduledJobCommandStateError(
                "The edited definition was stored but its active state cannot "
                "be restored."
            )

        continuing_interrupted_pause = was_active and original.status in {
            ScheduledJobStatus.PAUSED.value,
            ScheduledJobStatus.PROVISIONING_FAILED.value,
        }
        if (
            original.updated_at != command.expected_job_updated_at
            and not continuing_interrupted_pause
        ):
            raise ScheduledJobCommandStateError(
                "The scheduled job changed after this edit was requested."
            )
        paused_by_update = False
        revision_applied = False
        try:
            current = original
            if was_active and original.status == ScheduledJobStatus.ACTIVE.value:
                current = self._provisioner.pause(original.job_id).job
                paused_by_update = True
            applied = self._commands.apply_definition_revision(
                command=command,
                expected_job_updated_at=current.updated_at,
                definition=definition,
                change_kind="edited",
            )
            if applied is None:
                raise ScheduledJobCommandStateError(
                    "The scheduled job changed while its edit was being applied."
                )
            revision_applied = True
            if was_active:
                if applied.status == ScheduledJobStatus.PAUSED.value:
                    return self._provisioner.resume(original.job_id).job.status
                if applied.status == ScheduledJobStatus.PROVISIONING_FAILED.value:
                    return self._provisioner.provision(original.job_id).job.status
                raise ScheduledJobCommandStateError(
                    "The edited task entered an unexpected inactive state."
                )
            return applied.status
        except Exception as original_error:
            rollback_errors: list[str] = []
            try:
                current = self._jobs.get_job(original.job_id)
                if current is not None and revision_applied:
                    restored = self._commands.apply_definition_revision(
                        command=command,
                        expected_job_updated_at=current.updated_at,
                        definition=original.definition,
                        change_kind="rollback",
                    )
                    if restored is None:
                        rollback_errors.append("definition rollback conflicted")
                if was_active and (paused_by_update or continuing_interrupted_pause):
                    current = self._jobs.get_job(original.job_id)
                    if current is not None:
                        if current.status == ScheduledJobStatus.PAUSED.value:
                            self._provisioner.resume(original.job_id)
                        elif (
                            current.status
                            == ScheduledJobStatus.PROVISIONING_FAILED.value
                        ):
                            self._provisioner.provision(original.job_id)
            except Exception as rollback_error:
                rollback_errors.append(safe_scheduled_job_error_message(rollback_error))
            if rollback_errors:
                raise RuntimeError(
                    f"{safe_scheduled_job_error_message(original_error)}; "
                    f"rollback needs attention: {'; '.join(rollback_errors)}"
                ) from original_error
            raise ScheduledJobCommandStateError(
                "The edit could not be applied; the previous task definition "
                "and active state were restored."
            ) from original_error


__all__ = [
    "ScheduledJobCommandBatchResult",
    "ScheduledJobCommandProcessor",
    "ScheduledJobCommandStateError",
]
