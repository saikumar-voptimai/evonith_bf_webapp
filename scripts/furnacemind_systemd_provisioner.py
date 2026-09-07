"""Privileged command-line boundary for FurnaceMind scheduled-job timers.

Operators create definitions in Streamlit, while this narrowly scoped command
installs the shared service template and performs fixed per-job provisioning,
pause/resume, archive, or finite reconciliation actions. Mutations are opt-in,
and production activation remains separately gated by explicit deployment
readiness flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.scheduled_tasks.provisioning_service import (  # noqa: E402
    ControlSyncUnitsResult,
    FileJobOperationLock,
    ScheduledJobExecutorNotReadyError,
    ScheduledJobProvisioningDisabledError,
    ScheduledJobProvisioningError,
    ScheduledJobProvisioningPolicy,
    ScheduledJobProvisioningResult,
    ScheduledJobProvisioningService,
    ScheduledJobProvisioningStateError,
    ScheduledJobReconciliationBatchResult,
    ScheduledJobUnsupportedHostError,
    SharedServiceTemplateResult,
)
from data.scheduled_tasks.service import (  # noqa: E402
    ScheduledJobPersistenceError,
    ScheduledJobValidationError,
)
from utils.scheduled_tasks.furnacemind_executor import (  # noqa: E402
    PRODUCTION_EXECUTOR_CAPABILITY_READY,
)
from utils.scheduled_tasks.systemd_controller import (  # noqa: E402
    SystemdController,
    UnitFileStore,
)
from utils.scheduled_tasks.systemd_units import (  # noqa: E402
    SystemdProvisioningSettings,
)

LOGGER = logging.getLogger("furnacemind.scheduled_job_provisioner")


def _argument_parser() -> argparse.ArgumentParser:
    """Build the stable administrative command-line contract."""

    parser = argparse.ArgumentParser(
        prog="furnacemind-systemd-provisioner",
        description="Manage FurnaceMind systemd scheduling units.",
    )
    parser.add_argument(
        "action",
        choices=(
            "install-service",
            "install-control-sync",
            "plan",
            "provision",
            "pause",
            "resume",
            "archive",
            "reconcile",
            "reconcile-all",
        ),
        help="Lifecycle action to perform",
    )
    parser.add_argument("--job-id", help="Scheduled job UUID")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the action without filesystem, systemctl, or database writes",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Internal reconcile-all page size (1-500)",
    )
    return parser


def _settings_from_environment(
    environment: Mapping[str, str],
) -> SystemdProvisioningSettings:
    """Build trusted systemd paths and service identity from deployment values."""

    return SystemdProvisioningSettings(
        unit_directory=environment.get(
            "SCHEDULED_TASKS_SYSTEMD_UNIT_DIRECTORY",
            "/etc/systemd/system",
        ),
        working_directory=environment.get(
            "SCHEDULED_TASKS_WORKING_DIRECTORY",
            "/opt/furnacemind",
        ),
        python_executable=environment.get(
            "SCHEDULED_TASKS_PYTHON_EXECUTABLE",
            "/opt/furnacemind/.venv/bin/python",
        ),
        runner_script=environment.get(
            "SCHEDULED_TASKS_RUNNER_SCRIPT",
            "/opt/furnacemind/scripts/furnacemind_job_runner.py",
        ),
        control_sync_script=environment.get(
            "SCHEDULED_TASKS_CONTROL_SYNC_SCRIPT",
            "/opt/furnacemind/scripts/furnacemind_control_sync.py",
        ),
        operation_lock_directory=environment.get(
            "SCHEDULED_TASKS_LOCK_DIRECTORY",
            "/run/lock/furnacemind-scheduled-tasks",
        ),
        environment_file=environment.get(
            "SCHEDULED_TASKS_ENVIRONMENT_FILE",
            "/etc/furnacemind/furnacemind.env",
        ),
        systemctl_path=environment.get(
            "SCHEDULED_TASKS_SYSTEMCTL_PATH",
            "/usr/bin/systemctl",
        ),
        systemd_analyze_path=environment.get(
            "SCHEDULED_TASKS_SYSTEMD_ANALYZE_PATH",
            "/usr/bin/systemd-analyze",
        ),
        service_user=environment.get(
            "SCHEDULED_TASKS_SERVICE_USER",
            "furnacemind",
        ),
        service_group=environment.get(
            "SCHEDULED_TASKS_SERVICE_GROUP",
            "furnacemind",
        ),
        control_sync_interval_seconds=int(
            environment.get("SCHEDULED_TASKS_CONTROL_SYNC_INTERVAL_SECONDS", "30")
        ),
    )


def database_url_from_environment(environment: Mapping[str, str]) -> str | None:
    """Return the first configured relational URL without exposing its value."""

    for name in ("DATABASE_URL", "NEON_DATABASE_URL", "NEON_STR"):
        value = environment.get(name, "").strip()
        if value:
            return value
    return None


def build_provisioning_service_from_environment(
    environment: Mapping[str, str],
) -> ScheduledJobProvisioningService:
    """Construct the production provisioning coordinator from trusted settings."""

    settings = _settings_from_environment(environment)
    controller = SystemdController(
        systemctl_path=settings.systemctl_path,
        systemd_analyze_path=settings.systemd_analyze_path,
    )
    unit_store = UnitFileStore(
        str(settings.unit_directory),
        validate_on_init=False,
        require_root_trust=True,
    )
    operation_lock = FileJobOperationLock(
        environment.get(
            "SCHEDULED_TASKS_LOCK_DIRECTORY",
            "/run/lock/furnacemind-scheduled-tasks",
        )
    )
    return ScheduledJobProvisioningService(
        settings=settings,
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy.from_environment(environment),
        operation_lock=operation_lock,
        db_url=database_url_from_environment(environment),
        executor_capability_ready=PRODUCTION_EXECUTOR_CAPABILITY_READY,
    )


def _result_document(
    result: (
        ScheduledJobProvisioningResult
        | ScheduledJobReconciliationBatchResult
        | SharedServiceTemplateResult
        | ControlSyncUnitsResult
    ),
) -> dict[str, object]:
    """Convert a lifecycle result into safe JSON for deployment automation."""

    if isinstance(result, ScheduledJobReconciliationBatchResult):
        return {
            "action": result.action,
            "target_device_id": result.target_device_id,
            "scanned": result.scanned,
            "succeeded": result.succeeded,
            "warnings": result.warnings,
            "failed": result.failed,
            "omitted_issues": result.omitted_issues,
            "clean": result.clean,
            "dry_run": result.dry_run,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "issues": [
                {
                    "job_id": issue.job_id,
                    "outcome": issue.outcome,
                    "status": issue.status,
                    "changed": issue.changed,
                    "message": issue.message,
                }
                for issue in result.issues
            ],
        }
    if isinstance(result, ControlSyncUnitsResult):
        document = {
            "action": result.action,
            "changed": result.changed,
            "dry_run": result.dry_run,
            "units": [
                {
                    "unit_name": artifact.unit_name,
                    "destination_path": str(artifact.destination_path),
                }
                for artifact in result.artifacts
            ],
        }
        if result.warning:
            document["warning"] = result.warning
        if result.dry_run:
            for item, artifact in zip(document["units"], result.artifacts):
                item["content"] = artifact.content
        return document
    if isinstance(result, SharedServiceTemplateResult):
        document: dict[str, object] = {
            "action": result.action,
            "changed": result.changed,
            "dry_run": result.dry_run,
            "unit_name": result.artifact.unit_name,
            "destination_path": str(result.artifact.destination_path),
        }
        if result.warning:
            document["warning"] = result.warning
        if result.dry_run:
            document["content"] = result.artifact.content
        return document

    document: dict[str, object] = {
        "action": result.action,
        "job_id": result.job.job_id,
        "status": result.job.status,
        "changed": result.changed,
        "dry_run": result.dry_run,
        "timer_unit": result.plan.timer_unit,
        "service_instance": result.plan.service_instance,
        "on_calendar": list(result.plan.on_calendar),
    }
    if result.warning:
        document["warning"] = result.warning
    if result.dry_run:
        document["unit_files"] = [
            {
                "unit_name": artifact.unit_name,
                "destination_path": str(artifact.destination_path),
                "content": artifact.content,
            }
            for artifact in result.plan.artifacts
        ]
    return document


def _run_action(
    service: ScheduledJobProvisioningService,
    *,
    action: str,
    job_id: str | None,
    dry_run: bool,
    page_size: int,
) -> (
    ScheduledJobProvisioningResult
    | ScheduledJobReconciliationBatchResult
    | SharedServiceTemplateResult
    | ControlSyncUnitsResult
):
    """Dispatch one allow-listed lifecycle action without dynamic method lookup."""

    if action == "install-service":
        if job_id is not None:
            raise ValueError("install-service does not accept --job-id.")
        return service.install_service_template(dry_run=dry_run)
    if action == "install-control-sync":
        if job_id is not None:
            raise ValueError("install-control-sync does not accept --job-id.")
        return service.install_control_sync_units(dry_run=dry_run)
    if action == "reconcile-all":
        if job_id is not None:
            raise ValueError("reconcile-all does not accept --job-id.")
        return service.reconcile_all(dry_run=dry_run, page_size=page_size)
    if job_id is None:
        raise ValueError(f"{action} requires --job-id.")
    if action == "plan":
        return service.plan(job_id)
    if action == "provision":
        return service.provision(job_id, dry_run=dry_run)
    if action == "pause":
        return service.pause(job_id, dry_run=dry_run)
    if action == "resume":
        return service.resume(job_id, dry_run=dry_run)
    if action == "archive":
        return service.archive(job_id, dry_run=dry_run)
    if action == "reconcile":
        return service.reconcile(job_id, dry_run=dry_run)
    raise ValueError(f"Unsupported provisioning action: {action}")


def main(
    argv: Sequence[str] | None = None,
    *,
    service: ScheduledJobProvisioningService | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    """Execute one management action and return a stable process exit status."""

    args = _argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    owns_service = service is None
    try:
        if service is None:
            service = build_provisioning_service_from_environment(
                os.environ if environment is None else environment
            )
        result = _run_action(
            service,
            action=args.action,
            job_id=args.job_id,
            dry_run=args.dry_run,
            page_size=args.page_size,
        )
    except (
        ScheduledJobExecutorNotReadyError,
        ScheduledJobProvisioningDisabledError,
        ScheduledJobProvisioningStateError,
        ScheduledJobUnsupportedHostError,
        ScheduledJobValidationError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Scheduled-job request was rejected: %s", exc)
        return 2
    except (ScheduledJobPersistenceError, ScheduledJobProvisioningError) as exc:
        LOGGER.error("Scheduled-job operation failed: %s", exc)
        return 1
    finally:
        if owns_service and service is not None:
            service.dispose()

    print(json.dumps(_result_document(result), indent=2, sort_keys=True))
    if isinstance(result, ScheduledJobReconciliationBatchResult) and not result.clean:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
