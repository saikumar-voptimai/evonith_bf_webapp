"""Safely reconcile database-backed scheduled jobs with Linux systemd.

The Streamlit process only creates pending database rows.  This module is the
separate privileged boundary that renders fixed unit files, validates and
installs them, invokes systemd without a shell, and performs guarded database
lifecycle transitions.  External operations never run inside a database
transaction, and every activation failure is compensated toward an inactive
state.
"""

from __future__ import annotations

import os
import platform
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import ContextManager, Protocol
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobRepository,
    ScheduledJobRunLeaseBusyError,
    ScheduledJobStatus,
    build_relational_engine,
    build_relational_session_factory,
)
from utils.scheduled_tasks.scheduled_task_definition import validate_task_definition
from utils.scheduled_tasks.systemd_units import (
    CONTROL_SYNC_TIMER_UNIT_NAME,
    SERVICE_TEMPLATE_UNIT_NAME,
    ProvisioningPlan,
    SystemdProvisioningSettings,
    SystemdUnitArtifact,
    build_control_sync_artifacts,
    build_provisioning_plan,
    build_service_template_artifact,
    service_instance_name,
    timer_unit_name,
)

from .execution_service import safe_scheduled_job_error_message
from .service import (
    ScheduledJobPersistenceError,
    ScheduledJobService,
    ScheduledJobValidationError,
    ScheduledJobView,
)


class ScheduledJobProvisioningError(RuntimeError):
    """Raised when a systemd or lifecycle provisioning operation fails."""


class ScheduledJobProvisioningConflictError(ScheduledJobProvisioningError):
    """Raised when another actor changes a job during a guarded operation."""


class ScheduledJobProvisioningStateError(ScheduledJobProvisioningError):
    """Raised when an action is invalid for the job's current lifecycle state."""


class ScheduledJobProvisioningDisabledError(ScheduledJobProvisioningError):
    """Raised when deployment policy has not enabled systemd mutations."""


class ScheduledJobExecutorNotReadyError(ScheduledJobProvisioningError):
    """Raised when activation is attempted before a real executor is connected."""


class ScheduledJobUnsupportedHostError(ScheduledJobProvisioningError):
    """Raised when a mutating operation is attempted outside a systemd host."""


class SystemdControllerProtocol(Protocol):
    """Systemd command surface required by the provisioning coordinator."""

    def validate_calendar(self, expression: str) -> str:
        """Validate one calendar expression and return normalized tool output."""

    def verify(self, paths: tuple[Path, ...]) -> None:
        """Validate installed service and timer unit files."""

    def daemon_reload(self) -> None:
        """Reload systemd's unit-file view."""

    def enable(self, unit_name: str) -> None:
        """Enable a timer without starting it."""

    def start(self, unit_name: str) -> None:
        """Start a timer."""

    def stop(self, unit_name: str) -> None:
        """Stop a timer."""

    def disable(self, unit_name: str) -> None:
        """Disable a timer."""

    def clean_state(self, unit_name: str) -> None:
        """Remove persistent timer state."""

    def is_enabled(self, unit_name: str) -> bool:
        """Return whether a timer is enabled."""

    def is_active(self, unit_name: str) -> bool:
        """Return whether a timer is active."""


class UnitFileSnapshotProtocol(Protocol):
    """Prior unit-file state that can be restored after a failed operation."""

    unit_name: str
    content: bytes | str | None
    existed: bool


class UnitFileStoreProtocol(Protocol):
    """Atomic managed-unit storage used by the provisioning coordinator."""

    def snapshot(self, unit_name: str) -> UnitFileSnapshotProtocol:
        """Capture a managed unit's current state."""

    def install(self, unit_name: str, content: str) -> UnitFileSnapshotProtocol:
        """Atomically install content and return the prior state."""

    def restore(self, snapshot: UnitFileSnapshotProtocol) -> None:
        """Restore an earlier unit-file state."""

    def remove(self, unit_name: str) -> UnitFileSnapshotProtocol:
        """Remove an exact managed unit and return its prior state."""


class SystemdHostProbeProtocol(Protocol):
    """Deployment-host checks separated by cleanup and activation capability."""

    def ensure_control_supported(self, settings: SystemdProvisioningSettings) -> None:
        """Raise when existing systemd timers cannot be controlled safely."""

    def ensure_activation_supported(
        self,
        settings: SystemdProvisioningSettings,
    ) -> None:
        """Raise when this host cannot validate and activate scheduled jobs."""


class JobOperationLockProtocol(Protocol):
    """Per-job serialization surface for filesystem and systemd operations."""

    def hold(self, job_id: str) -> ContextManager[None]:
        """Return a context manager holding the lock for one canonical job id."""


@dataclass(frozen=True, slots=True)
class ScheduledJobProvisioningPolicy:
    """Deployment-owned gates for privileged scheduling operations."""

    systemd_mutations_enabled: bool = False
    production_executor_ready: bool = False
    expected_target_device_id: str = "bf2-jetson-01"

    def __post_init__(self) -> None:
        """Validate the deployment target identifier."""

        target = self.expected_target_device_id.strip()
        if not target or len(target) > 128:
            raise ValueError("expected_target_device_id must be 1 to 128 characters.")
        if any(character in target for character in ("\x00", "\r", "\n")):
            raise ValueError("expected_target_device_id contains unsafe characters.")
        object.__setattr__(self, "expected_target_device_id", target)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> ScheduledJobProvisioningPolicy:
        """Build conservative policy gates from deployment environment values."""

        values = os.environ if environment is None else environment
        return cls(
            systemd_mutations_enabled=_environment_flag(
                values,
                "SCHEDULED_TASKS_SYSTEMD_ENABLED",
            ),
            production_executor_ready=_environment_flag(
                values,
                "SCHEDULED_TASKS_EXECUTOR_READY",
            ),
            expected_target_device_id=values.get(
                "SCHEDULED_TASKS_TARGET_DEVICE_ID",
                "bf2-jetson-01",
            ),
        )


@dataclass(frozen=True, slots=True)
class ScheduledJobCleanupPlan:
    """UUID-derived timer target usable even when stored JSON is corrupt."""

    job_id: str
    service_instance: str
    timer_unit: str
    on_calendar: tuple[str, ...] = ()
    artifacts: tuple[SystemdUnitArtifact, ...] = ()

    def __post_init__(self) -> None:
        """Ensure all cleanup identifiers agree with one canonical job UUID."""

        canonical = _canonical_job_id(self.job_id)
        if self.service_instance != service_instance_name(canonical):
            raise ValueError("service_instance does not match cleanup job_id.")
        if self.timer_unit != timer_unit_name(canonical):
            raise ValueError("timer_unit does not match cleanup job_id.")
        if self.on_calendar or self.artifacts:
            raise ValueError("Cleanup plans must not contain rendered schedule files.")


@dataclass(frozen=True, slots=True)
class ScheduledJobProvisioningResult:
    """Stable result returned by a provisioning lifecycle action."""

    action: str
    job: ScheduledJobView
    plan: ProvisioningPlan | ScheduledJobCleanupPlan
    changed: bool
    dry_run: bool = False
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class SharedServiceTemplateResult:
    """Result of installing or inspecting the deployment-wide service template."""

    action: str
    artifact: SystemdUnitArtifact
    changed: bool
    dry_run: bool = False
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ControlSyncUnitsResult:
    """Result of installing the Jetson command-sync service and timer."""

    action: str
    artifacts: tuple[SystemdUnitArtifact, SystemdUnitArtifact]
    changed: bool
    dry_run: bool = False
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduledJobReconciliationIssue:
    """One warning or failure retained from a bounded reconciliation scan."""

    job_id: str
    outcome: str
    status: str | None
    changed: bool | None
    message: str


@dataclass(frozen=True, slots=True)
class ScheduledJobReconciliationBatchResult:
    """Aggregate result of an explicit, finite all-job reconciliation pass."""

    action: str
    target_device_id: str
    scanned: int
    succeeded: int
    warnings: int
    failed: int
    omitted_issues: int
    dry_run: bool
    started_at: datetime
    completed_at: datetime
    issues: tuple[ScheduledJobReconciliationIssue, ...]

    @property
    def clean(self) -> bool:
        """Return whether every scanned job converged without a warning."""

        return self.warnings == 0 and self.failed == 0


def _environment_flag(environment: Mapping[str, str], name: str) -> bool:
    """Return a strict opt-in boolean for one deployment environment value."""

    return environment.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC instant."""

    return datetime.now(timezone.utc)


def _canonical_job_id(job_id: object) -> str:
    """Normalize a UUID for database lookup, locks, and derived unit names."""

    try:
        return str(UUID(str(job_id).strip()))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ScheduledJobValidationError(("job_id: valid UUID is required",)) from exc


_SHARED_SERVICE_LOCK_ID = "00000000-0000-0000-0000-000000000000"
_MAX_RECONCILIATION_ISSUES = 1_000


def _one_time_run_at(definition: object) -> datetime | None:
    """Return a validated one-time trigger instant, otherwise ``None``."""

    if not isinstance(definition, dict):
        return None
    schedule = definition.get("schedule")
    trigger = schedule.get("trigger") if isinstance(schedule, dict) else None
    if not isinstance(trigger, dict) or trigger.get("type") != "once":
        return None
    raw_run_at = trigger.get("run_at")
    if not isinstance(raw_run_at, str):
        raise ScheduledJobProvisioningError(
            "The one-time schedule is missing its run timestamp."
        )
    try:
        run_at = datetime.fromisoformat(raw_run_at)
    except ValueError as exc:
        raise ScheduledJobProvisioningError(
            "The one-time schedule has an invalid run timestamp."
        ) from exc
    if run_at.tzinfo is None or run_at.utcoffset() is None:
        raise ScheduledJobProvisioningError(
            "The one-time schedule timestamp must include a timezone offset."
        )
    return run_at.astimezone(timezone.utc)


class LinuxSystemdHostProbe:
    """Verify that a real Linux systemd manager and configured tools are present."""

    def __init__(self, manager_directory: Path = Path("/run/systemd/system")) -> None:
        """Create a probe for the system manager's runtime directory."""

        self._manager_directory = manager_directory

    def ensure_control_supported(self, settings: SystemdProvisioningSettings) -> None:
        """Require only the host capabilities needed to stop or remove timers."""

        try:
            self._ensure_control_supported(settings)
        except OSError as exc:
            raise ScheduledJobUnsupportedHostError(
                "The systemd control plane could not be inspected safely."
            ) from exc

    def _ensure_control_supported(
        self,
        settings: SystemdProvisioningSettings,
    ) -> None:
        """Inspect the minimal host paths used by cleanup operations."""

        if platform.system() != "Linux" or not self._manager_directory.is_dir():
            raise ScheduledJobUnsupportedHostError(
                "Systemd mutations require a Linux host with a running system manager."
            )
        systemctl_path = Path(str(settings.systemctl_path))
        if not systemctl_path.is_file() or not os.access(systemctl_path, os.X_OK):
            raise ScheduledJobUnsupportedHostError(
                f"Required systemd tool is unavailable: {systemctl_path}"
            )

    def ensure_activation_supported(
        self,
        settings: SystemdProvisioningSettings,
    ) -> None:
        """Require every runtime dependency used to validate and execute jobs."""

        try:
            self._ensure_activation_supported(settings)
        except OSError as exc:
            raise ScheduledJobUnsupportedHostError(
                "The scheduled-job runtime could not be inspected safely."
            ) from exc

    def _ensure_activation_supported(
        self,
        settings: SystemdProvisioningSettings,
    ) -> None:
        """Inspect all application and identity dependencies used for activation."""

        self.ensure_control_supported(settings)
        analyze_path = Path(str(settings.systemd_analyze_path))
        if not analyze_path.is_file() or not os.access(analyze_path, os.X_OK):
            raise ScheduledJobUnsupportedHostError(
                f"Required systemd tool is unavailable: {analyze_path}"
            )
        working_directory = Path(str(settings.working_directory))
        if not working_directory.is_dir():
            raise ScheduledJobUnsupportedHostError(
                f"Configured FurnaceMind working directory is unavailable: "
                f"{working_directory}"
            )
        python_executable = Path(str(settings.python_executable))
        if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
            raise ScheduledJobUnsupportedHostError(
                f"Configured FurnaceMind Python is not executable: {python_executable}"
            )
        runner_script = Path(str(settings.runner_script))
        if not runner_script.is_file():
            raise ScheduledJobUnsupportedHostError(
                f"Configured scheduled-job runner is unavailable: {runner_script}"
            )
        environment_file = Path(str(settings.environment_file))
        if environment_file.is_symlink() or not environment_file.is_file():
            raise ScheduledJobUnsupportedHostError(
                f"Configured environment file is unavailable or unsafe: "
                f"{environment_file}"
            )
        environment_status = environment_file.stat()
        unsafe_environment_bits = stat.S_IWGRP | stat.S_IXGRP | stat.S_IRWXO
        if (
            environment_status.st_uid != 0
            or environment_status.st_mode & unsafe_environment_bits
        ):
            raise ScheduledJobUnsupportedHostError(
                "The scheduled-job environment file must be root-owned, group-readable "
                "at most, and inaccessible to other users."
            )
        try:
            import grp
            import pwd

            service_user = pwd.getpwnam(settings.service_user)
            service_group = grp.getgrnam(settings.service_group)
        except (ImportError, KeyError) as exc:
            raise ScheduledJobUnsupportedHostError(
                "The configured FurnaceMind service user or group does not exist."
            ) from exc
        if (
            service_user.pw_uid == 0
            or service_user.pw_gid == 0
            or service_group.gr_gid == 0
        ):
            raise ScheduledJobUnsupportedHostError(
                "The scheduled-job service must use an unprivileged user and group."
            )


class FileJobOperationLock:
    """Serialize privileged lifecycle work with global and per-job Linux locks."""

    def __init__(
        self,
        lock_directory: (
            str | os.PathLike[str]
        ) = "/run/lock/furnacemind-scheduled-tasks",
    ) -> None:
        """Create a lock manager rooted at an explicit absolute directory."""

        raw_directory = os.fspath(lock_directory)
        posix_directory = PurePosixPath(raw_directory)
        if not posix_directory.is_absolute() or posix_directory == PurePosixPath("/"):
            raise ValueError("lock_directory must be a specific absolute path.")
        self._lock_directory = Path(raw_directory)

    @staticmethod
    def _validate_directory_status(directory_status: os.stat_result) -> None:
        """Require a root-owned directory that unprivileged users cannot replace."""

        unsafe_write_bits = stat.S_IWGRP | stat.S_IWOTH
        if (
            not stat.S_ISDIR(directory_status.st_mode)
            or directory_status.st_uid != 0
            or directory_status.st_mode & unsafe_write_bits
        ):
            raise ScheduledJobUnsupportedHostError(
                "The scheduled-task lock directory is not trusted."
            )

    @staticmethod
    def _validate_lock_file_status(file_status: os.stat_result) -> None:
        """Require each opened lock descriptor to reference a root-owned file."""

        unsafe_write_bits = stat.S_IWGRP | stat.S_IWOTH
        if (
            not stat.S_ISREG(file_status.st_mode)
            or file_status.st_uid != 0
            or file_status.st_nlink != 1
            or file_status.st_mode & unsafe_write_bits
        ):
            raise ScheduledJobUnsupportedHostError(
                "A scheduled-task lock file is not trusted."
            )

    @contextmanager
    def hold(self, job_id: str) -> Iterator[None]:
        """Hold shared-unit and job locks in a stable global-to-local order."""

        canonical = _canonical_job_id(job_id)
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - host probe blocks Windows
            raise ScheduledJobUnsupportedHostError(
                "Per-job file locking requires a POSIX host."
            ) from exc

        descriptors: list[int] = []
        try:
            self._lock_directory.mkdir(mode=0o750, parents=True, exist_ok=True)
            directory_stat = self._lock_directory.lstat()
            if self._lock_directory.is_symlink():
                raise ScheduledJobUnsupportedHostError(
                    "The scheduled-task lock directory is not trusted."
                )
            self._validate_directory_status(directory_stat)
            for lock_name in (
                "furnacemind-systemd-global.lock",
                f"furnacemind-job-{canonical}.lock",
            ):
                lock_path = self._lock_directory / lock_name
                flags = os.O_CREAT | os.O_RDWR
                flags |= getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(lock_path, flags, 0o600)
                descriptors.append(descriptor)
                self._validate_lock_file_status(os.fstat(descriptor))
                os.fchmod(descriptor, 0o600)
                self._validate_lock_file_status(os.fstat(descriptor))
                fcntl.flock(descriptor, fcntl.LOCK_EX)
        except (OSError, ScheduledJobProvisioningError) as exc:
            for descriptor in reversed(descriptors):
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if isinstance(exc, ScheduledJobProvisioningError):
                raise
            raise ScheduledJobProvisioningError(
                "Scheduled-task operation locks could not be acquired."
            ) from exc

        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            release_error: OSError | None = None
            for descriptor in reversed(descriptors):
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError as exc:
                    release_error = release_error or exc
                try:
                    os.close(descriptor)
                except OSError as exc:
                    release_error = release_error or exc
            if release_error is not None and not body_failed:
                raise ScheduledJobProvisioningError(
                    "Scheduled-task operation locks could not be released."
                ) from release_error


class ScheduledJobProvisioningService:
    """Coordinate safe systemd files, commands, and database lifecycle state."""

    def __init__(
        self,
        *,
        settings: SystemdProvisioningSettings,
        controller: SystemdControllerProtocol,
        unit_store: UnitFileStoreProtocol,
        policy: ScheduledJobProvisioningPolicy | None = None,
        repository: ScheduledJobRepository | None = None,
        db_url: str | None = None,
        host_probe: SystemdHostProbeProtocol | None = None,
        operation_lock: JobOperationLockProtocol | None = None,
        clock: Callable[[], datetime] | None = None,
        executor_capability_ready: bool = False,
    ) -> None:
        """Create a coordinator with explicit privileged-operation dependencies."""

        self._engine: Engine | None = None
        self._db_url = db_url
        self._repository_instance = repository
        self._job_service_instance = (
            ScheduledJobService(repository=repository)
            if repository is not None
            else None
        )
        self._settings = settings
        self._controller = controller
        self._unit_store = unit_store
        self._policy = policy or ScheduledJobProvisioningPolicy()
        self._host_probe = host_probe or LinuxSystemdHostProbe()
        self._operation_lock = operation_lock or FileJobOperationLock()
        self._clock = clock or _utc_now
        self._executor_capability_ready = executor_capability_ready

    def dispose(self) -> None:
        """Dispose the database engine created by this service, if any."""

        if self._engine is not None:
            self._engine.dispose()

    @property
    def _repository(self) -> ScheduledJobRepository:
        """Return the repository, creating database resources only when needed."""

        if self._repository_instance is None:
            self._engine = build_relational_engine(db_url=self._db_url)
            session_factory = build_relational_session_factory(self._engine)
            self._repository_instance = ScheduledJobRepository(session_factory)
        return self._repository_instance

    @property
    def _job_service(self) -> ScheduledJobService:
        """Return the read service backed by the lazily created repository."""

        if self._job_service_instance is None:
            self._job_service_instance = ScheduledJobService(
                repository=self._repository
            )
        return self._job_service_instance

    def install_service_template(
        self,
        *,
        dry_run: bool = False,
    ) -> SharedServiceTemplateResult:
        """Install the shared service once as an explicit deployment action."""

        artifact = build_service_template_artifact(self._settings)
        if dry_run:
            return SharedServiceTemplateResult(
                action="install-service",
                artifact=artifact,
                changed=False,
                dry_run=True,
            )

        self._ensure_activation_allowed()
        with self._operation_lock.hold(_SHARED_SERVICE_LOCK_ID):
            snapshot = self._unit_store.snapshot(artifact.unit_name)
            changed = not self._snapshot_matches(snapshot, artifact.content)
            installed_snapshot: UnitFileSnapshotProtocol | None = None
            try:
                if changed:
                    installed_snapshot = self._unit_store.install(
                        artifact.unit_name,
                        artifact.content,
                    )
                self._controller.verify((Path(str(artifact.destination_path)),))
                # Reload even when the bytes already match. This repairs a crash
                # after file replacement but before the manager observed it.
                self._controller.daemon_reload()
            except Exception as exc:
                compensation_warning = None
                if installed_snapshot is not None:
                    failures: list[str] = []
                    try:
                        self._unit_store.restore(installed_snapshot)
                    except Exception as rollback_exc:
                        failures.append(safe_scheduled_job_error_message(rollback_exc))
                    try:
                        self._controller.daemon_reload()
                    except Exception as reload_exc:
                        failures.append(safe_scheduled_job_error_message(reload_exc))
                    compensation_warning = "; ".join(failures) or None
                warning = self._combined_warning(
                    safe_scheduled_job_error_message(exc),
                    compensation_warning,
                )
                raise ScheduledJobProvisioningError(
                    "Shared scheduled-job service installation failed: " + warning
                ) from exc
            return SharedServiceTemplateResult(
                action="install-service",
                artifact=artifact,
                changed=changed,
            )

    def install_control_sync_units(
        self,
        *,
        dry_run: bool = False,
    ) -> ControlSyncUnitsResult:
        """Install and start the short-lived PostgreSQL command bridge timer."""

        artifacts = build_control_sync_artifacts(self._settings)
        if dry_run:
            return ControlSyncUnitsResult(
                action="install-control-sync",
                artifacts=artifacts,
                changed=False,
                dry_run=True,
            )

        self._ensure_activation_allowed()
        with self._operation_lock.hold(_SHARED_SERVICE_LOCK_ID):
            snapshots = tuple(
                self._unit_store.snapshot(artifact.unit_name) for artifact in artifacts
            )
            changed = any(
                not self._snapshot_matches(snapshot, artifact.content)
                for snapshot, artifact in zip(snapshots, artifacts)
            )
            installed: list[UnitFileSnapshotProtocol] = []
            try:
                for snapshot, artifact in zip(snapshots, artifacts):
                    if not self._snapshot_matches(snapshot, artifact.content):
                        installed.append(
                            self._unit_store.install(
                                artifact.unit_name,
                                artifact.content,
                            )
                        )
                self._controller.verify(
                    tuple(
                        Path(str(artifact.destination_path)) for artifact in artifacts
                    )
                )
                self._controller.daemon_reload()
                self._controller.enable(CONTROL_SYNC_TIMER_UNIT_NAME)
                self._controller.start(CONTROL_SYNC_TIMER_UNIT_NAME)
            except Exception as exc:
                failures: list[str] = []
                try:
                    self._controller.stop(CONTROL_SYNC_TIMER_UNIT_NAME)
                except Exception as cleanup_exc:
                    failures.append(safe_scheduled_job_error_message(cleanup_exc))
                try:
                    self._controller.disable(CONTROL_SYNC_TIMER_UNIT_NAME)
                except Exception as cleanup_exc:
                    failures.append(safe_scheduled_job_error_message(cleanup_exc))
                for snapshot in reversed(installed):
                    try:
                        self._unit_store.restore(snapshot)
                    except Exception as rollback_exc:
                        failures.append(safe_scheduled_job_error_message(rollback_exc))
                try:
                    self._controller.daemon_reload()
                except Exception as reload_exc:
                    failures.append(safe_scheduled_job_error_message(reload_exc))
                detail = self._combined_warning(
                    safe_scheduled_job_error_message(exc),
                    "; ".join(failures) or None,
                )
                raise ScheduledJobProvisioningError(
                    "Scheduled-task control synchronization installation failed: "
                    + detail
                ) from exc
            return ControlSyncUnitsResult(
                action="install-control-sync",
                artifacts=artifacts,
                changed=changed,
            )

    def reconcile_all(
        self,
        *,
        dry_run: bool = False,
        page_size: int = 100,
    ) -> ScheduledJobReconciliationBatchResult:
        """Run one bounded, explicit reconciliation pass for this target host."""

        if not 1 <= page_size <= 500:
            raise ValueError("page_size must be between 1 and 500.")
        started_at = self._normalized_clock("reconciliation start")
        try:
            created_through = self._repository.reconciliation_cutoff()
        except (SQLAlchemyError, ValueError) as exc:
            raise ScheduledJobPersistenceError(
                "The reconciliation database cutoff could not be loaded."
            ) from exc
        if not dry_run:
            self._ensure_mutation_policy_enabled()

        scanned = succeeded = warnings = failed = omitted_issues = 0
        issues: list[ScheduledJobReconciliationIssue] = []
        after_created_at: datetime | None = None
        after_job_id: str | None = None
        while True:
            try:
                page = self._repository.list_reconciliation_job_ids(
                    target_device_id=self._policy.expected_target_device_id,
                    created_through=created_through,
                    after_created_at=after_created_at,
                    after_job_id=after_job_id,
                    limit=page_size,
                )
            except (SQLAlchemyError, ValueError) as exc:
                raise ScheduledJobPersistenceError(
                    "A scheduled-job reconciliation page could not be loaded."
                ) from exc
            if not page:
                break
            for candidate_job_id, _ in page:
                scanned += 1
                try:
                    result = self.reconcile(candidate_job_id, dry_run=dry_run)
                except (
                    ScheduledJobPersistenceError,
                    ScheduledJobProvisioningError,
                    ScheduledJobValidationError,
                    ValueError,
                ) as exc:
                    failed += 1
                    issue = ScheduledJobReconciliationIssue(
                        job_id=candidate_job_id,
                        outcome="failed",
                        status=None,
                        changed=None,
                        message=safe_scheduled_job_error_message(exc),
                    )
                    omitted_issues += self._append_reconciliation_issue(issues, issue)
                    continue
                if result.warning:
                    warnings += 1
                    issue = ScheduledJobReconciliationIssue(
                        job_id=candidate_job_id,
                        outcome="warning",
                        status=result.job.status,
                        changed=result.changed,
                        message=safe_scheduled_job_error_message(result.warning),
                    )
                    omitted_issues += self._append_reconciliation_issue(issues, issue)
                else:
                    succeeded += 1
            if len(page) < page_size:
                break
            after_job_id, after_created_at = page[-1]

        return ScheduledJobReconciliationBatchResult(
            action="reconcile-all",
            target_device_id=self._policy.expected_target_device_id,
            scanned=scanned,
            succeeded=succeeded,
            warnings=warnings,
            failed=failed,
            omitted_issues=omitted_issues,
            dry_run=dry_run,
            started_at=started_at,
            completed_at=self._normalized_clock("reconciliation completion"),
            issues=tuple(issues),
        )

    def plan(self, job_id: str) -> ScheduledJobProvisioningResult:
        """Build a read-only unit plan without filesystem, systemctl, or DB writes."""

        job, plan = self._load_and_plan(job_id)
        return ScheduledJobProvisioningResult(
            action="plan",
            job=job,
            plan=plan,
            changed=False,
            dry_run=True,
        )

    def provision(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Install and activate a pending or previously failed scheduled job."""

        if dry_run:
            job, plan = self._load_and_plan(job_id)
            if job.status not in {
                ScheduledJobStatus.PENDING_PROVISIONING.value,
                ScheduledJobStatus.PROVISIONING_FAILED.value,
                ScheduledJobStatus.ACTIVE.value,
            }:
                raise ScheduledJobProvisioningStateError(
                    f"Cannot provision a job with status {job.status}."
                )
            return ScheduledJobProvisioningResult(
                action="provision",
                job=job,
                plan=plan,
                changed=False,
                dry_run=True,
            )
        self._ensure_activation_allowed()
        canonical = _canonical_job_id(job_id)
        with self._operation_lock.hold(canonical):
            job, plan = self._load_and_plan(canonical)
            if job.status == ScheduledJobStatus.ACTIVE.value:
                if job.timer_unit_name != plan.timer_unit:
                    raise ScheduledJobProvisioningStateError(
                        "Active job has an unexpected timer receipt; reconcile it first."
                    )
                return self._reconcile_active_job(
                    job=job,
                    plan=plan,
                    action="provision",
                )
            allowed = {
                ScheduledJobStatus.PENDING_PROVISIONING.value,
                ScheduledJobStatus.PROVISIONING_FAILED.value,
            }
            if job.status not in allowed:
                raise ScheduledJobProvisioningStateError(
                    f"Cannot provision a job with status {job.status}."
                )
            return self._activate(job=job, plan=plan, action="provision")

    def pause(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Make a job non-runnable before stopping and disabling its timer."""

        if dry_run:
            job, plan = self._load_cleanup_plan(job_id)
            if job.status not in {
                ScheduledJobStatus.ACTIVE.value,
                ScheduledJobStatus.PAUSED.value,
            }:
                raise ScheduledJobProvisioningStateError(
                    f"Cannot pause a job with status {job.status}."
                )
            return ScheduledJobProvisioningResult(
                action="pause",
                job=job,
                plan=plan,
                changed=False,
                dry_run=True,
            )
        self._ensure_mutation_policy_enabled()
        canonical = _canonical_job_id(job_id)
        with self._operation_lock.hold(canonical):
            job, plan = self._load_cleanup_plan(canonical)
            changed = False
            if job.status == ScheduledJobStatus.ACTIVE.value:
                try:
                    transitioned = self._repository_call(
                        "pause scheduled job",
                        self._repository.pause_job,
                        job_id=job.job_id,
                        expected_status=job.status,
                        expected_updated_at=job.updated_at,
                    )
                except ScheduledJobPersistenceError:
                    current = self._require_job(job.job_id)
                    if current.status != ScheduledJobStatus.PAUSED.value:
                        raise
                    transitioned = None
                    job = current
                if transitioned is None:
                    current = self._require_job(job.job_id)
                    if current.status != ScheduledJobStatus.PAUSED.value:
                        raise ScheduledJobProvisioningConflictError(
                            "The job changed while it was being paused."
                        )
                    job = current
                else:
                    job = self._require_job(job.job_id)
                changed = True
            elif job.status != ScheduledJobStatus.PAUSED.value:
                raise ScheduledJobProvisioningStateError(
                    f"Cannot pause a job with status {job.status}."
                )

            warning = self._deactivate_timer(job=job, timer_unit=plan.timer_unit)
            job = self._require_job(job.job_id)
            return ScheduledJobProvisioningResult(
                action="pause",
                job=job,
                plan=plan,
                changed=changed,
                warning=warning,
            )

    def resume(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Reinstall and activate a paused scheduled job with a fresh boundary."""

        if dry_run:
            job, plan = self._load_and_plan(job_id)
            if job.status not in {
                ScheduledJobStatus.PAUSED.value,
                ScheduledJobStatus.ACTIVE.value,
            }:
                raise ScheduledJobProvisioningStateError(
                    f"Cannot resume a job with status {job.status}."
                )
            return ScheduledJobProvisioningResult(
                action="resume",
                job=job,
                plan=plan,
                changed=False,
                dry_run=True,
            )
        self._ensure_activation_allowed()
        canonical = _canonical_job_id(job_id)
        with self._operation_lock.hold(canonical):
            job, plan = self._load_and_plan(canonical)
            if job.status == ScheduledJobStatus.ACTIVE.value:
                if job.timer_unit_name != plan.timer_unit:
                    raise ScheduledJobProvisioningStateError(
                        "Active job has an unexpected timer receipt; reconcile it first."
                    )
                return self._reconcile_active_job(
                    job=job,
                    plan=plan,
                    action="resume",
                )
            if job.status != ScheduledJobStatus.PAUSED.value:
                raise ScheduledJobProvisioningStateError(
                    f"Cannot resume a job with status {job.status}."
                )
            return self._activate(job=job, plan=plan, action="resume")

    def archive(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Soft-delete a job, remove its timer, and retain definition/run history."""

        if dry_run:
            job, plan = self._load_cleanup_plan(job_id)
            return ScheduledJobProvisioningResult(
                action="archive",
                job=job,
                plan=plan,
                changed=False,
                dry_run=True,
            )
        self._ensure_mutation_policy_enabled()
        canonical = _canonical_job_id(job_id)
        with self._operation_lock.hold(canonical):
            job, plan = self._load_cleanup_plan(canonical)
            changed = False
            if job.status != ScheduledJobStatus.DELETED.value:
                try:
                    transitioned = self._repository_call(
                        "archive scheduled job",
                        self._repository.mark_deleted,
                        job_id=job.job_id,
                        expected_status=job.status,
                        expected_updated_at=job.updated_at,
                    )
                except ScheduledJobPersistenceError:
                    current = self._require_job(job.job_id)
                    if current.status != ScheduledJobStatus.DELETED.value:
                        raise
                    transitioned = None
                    job = current
                if transitioned is None:
                    current = self._require_job(job.job_id)
                    if current.status != ScheduledJobStatus.DELETED.value:
                        raise ScheduledJobProvisioningConflictError(
                            "The job changed while it was being archived."
                        )
                    job = current
                else:
                    job = self._require_job(job.job_id)
                changed = True

            warning = self._remove_timer(job=job, timer_unit=plan.timer_unit)
            job = self._require_job(job.job_id)
            return ScheduledJobProvisioningResult(
                action="archive",
                job=job,
                plan=plan,
                changed=changed,
                warning=warning,
            )

    def reconcile(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Converge external timer state toward the database lifecycle state."""

        if dry_run:
            job, cleanup_plan = self._load_cleanup_plan(job_id)
            warning = None
            plan: ProvisioningPlan | ScheduledJobCleanupPlan = cleanup_plan
            if job.status == ScheduledJobStatus.ACTIVE.value:
                if not self._executor_ready:
                    warning = (
                        "Active job would be paused because the production "
                        "scheduled-task executor is not ready."
                    )
                else:
                    try:
                        _, plan = self._load_and_plan(job.job_id)
                    except ScheduledJobValidationError as exc:
                        warning = safe_scheduled_job_error_message(exc)
            return ScheduledJobProvisioningResult(
                action="reconcile",
                job=job,
                plan=plan,
                changed=False,
                dry_run=True,
                warning=warning,
            )
        self._ensure_mutation_policy_enabled()
        canonical = _canonical_job_id(job_id)
        with self._operation_lock.hold(canonical):
            job, cleanup_plan = self._load_cleanup_plan(canonical)
            if job.status == ScheduledJobStatus.ACTIVE.value:
                if not self._executor_ready:
                    return self._pause_unsafe_active_job(
                        job=job,
                        plan=cleanup_plan,
                        reason="The production scheduled-task executor is not ready.",
                    )
                try:
                    _, plan = self._load_and_plan(canonical)
                except ScheduledJobValidationError as exc:
                    return self._pause_unsafe_active_job(
                        job=job,
                        plan=cleanup_plan,
                        reason=safe_scheduled_job_error_message(exc),
                    )
                try:
                    self._host_probe.ensure_activation_supported(self._settings)
                except Exception as exc:
                    return self._pause_unsafe_active_job(
                        job=job,
                        plan=cleanup_plan,
                        reason=safe_scheduled_job_error_message(exc),
                    )
                return self._reconcile_active_job(
                    job=job,
                    plan=plan,
                    action="reconcile",
                )
            elif job.status in {
                ScheduledJobStatus.PENDING_PROVISIONING.value,
                ScheduledJobStatus.PROVISIONING_FAILED.value,
                ScheduledJobStatus.PAUSED.value,
            }:
                self._deactivate_timer(
                    job=job,
                    timer_unit=cleanup_plan.timer_unit,
                )
            elif job.status in {
                ScheduledJobStatus.COMPLETED.value,
                ScheduledJobStatus.EXECUTION_FAILED.value,
                ScheduledJobStatus.DELETED.value,
            }:
                self._remove_timer(
                    job=job,
                    timer_unit=cleanup_plan.timer_unit,
                )
            else:  # pragma: no cover - database constraint rejects unknown statuses
                raise ScheduledJobProvisioningStateError(
                    f"Unsupported scheduled-job status: {job.status}"
                )
            current = self._require_job(job.job_id)
            return ScheduledJobProvisioningResult(
                action="reconcile",
                job=current,
                plan=cleanup_plan,
                changed=True,
                warning=current.provisioning_error,
            )

    def _load_and_plan(
        self,
        job_id: str,
    ) -> tuple[ScheduledJobView, ProvisioningPlan]:
        """Load, revalidate, target-check, and render one stored job definition."""

        job = self._require_job(_canonical_job_id(job_id))
        if job.status in {
            ScheduledJobStatus.DELETED.value,
            ScheduledJobStatus.COMPLETED.value,
            ScheduledJobStatus.EXECUTION_FAILED.value,
        }:
            raise ScheduledJobProvisioningStateError(
                f"Jobs with status {job.status} cannot be activated."
            )
        errors = validate_task_definition(job.definition)
        if errors:
            raise ScheduledJobValidationError(tuple(errors))
        # Import lazily to avoid a package-initialization cycle: the executor's
        # persistence types are exported through ``data.scheduled_tasks``.
        from utils.scheduled_tasks.furnacemind_executor import (
            validate_production_definition,
        )

        production_errors = validate_production_definition(job.definition)
        if production_errors:
            raise ScheduledJobValidationError(production_errors)
        if job.target_device_id != self._policy.expected_target_device_id:
            raise ScheduledJobValidationError(
                ("target_device: stored job does not target this provisioning host",)
            )
        try:
            plan = build_provisioning_plan(job.job_id, job.definition, self._settings)
        except (KeyError, TypeError, ValueError) as exc:
            raise ScheduledJobValidationError((str(exc),)) from exc
        return job, plan

    def _load_cleanup_plan(
        self,
        job_id: str,
    ) -> tuple[ScheduledJobView, ScheduledJobCleanupPlan]:
        """Load a job and derive its cleanup target without trusting definition JSON."""

        canonical = _canonical_job_id(job_id)
        job = self._require_job(canonical)
        return job, ScheduledJobCleanupPlan(
            job_id=canonical,
            service_instance=service_instance_name(canonical),
            timer_unit=timer_unit_name(canonical),
        )

    def _require_job(self, job_id: str) -> ScheduledJobView:
        """Load one job or raise a stable not-found validation error."""

        job = self._job_service.get_job(job_id)
        if job is None:
            raise ScheduledJobValidationError(("job_id: scheduled job was not found",))
        return job

    def _ensure_mutation_policy_enabled(self) -> None:
        """Require the deployment-owned opt-in before any systemd mutation."""

        if not self._policy.systemd_mutations_enabled:
            raise ScheduledJobProvisioningDisabledError(
                "Systemd mutations are disabled by deployment policy."
            )

    def _ensure_activation_allowed(self) -> None:
        """Require policy, executor, and complete host activation capability."""

        self._ensure_mutation_policy_enabled()
        if not self._executor_ready:
            raise ScheduledJobExecutorNotReadyError(
                "Timer activation is blocked until the production executor is ready."
            )
        self._host_probe.ensure_activation_supported(self._settings)

    @property
    def _executor_ready(self) -> bool:
        """Require both deployment approval and an in-code executor capability."""

        return (
            self._policy.production_executor_ready and self._executor_capability_ready
        )

    def _pause_unsafe_active_job(
        self,
        *,
        job: ScheduledJobView,
        plan: ScheduledJobCleanupPlan,
        reason: str,
    ) -> ScheduledJobProvisioningResult:
        """Fail closed when an active job cannot be safely rendered or executed."""

        try:
            transitioned = self._repository_call(
                "pause unsafe active scheduled job",
                self._repository.pause_job,
                job_id=job.job_id,
                expected_status=job.status,
                expected_updated_at=job.updated_at,
            )
        except ScheduledJobPersistenceError:
            try:
                current = self._require_job(job.job_id)
            except (ScheduledJobPersistenceError, ScheduledJobValidationError):
                self._best_effort_timer_shutdown(plan.timer_unit)
                raise
            if current.status != ScheduledJobStatus.PAUSED.value:
                self._best_effort_timer_shutdown(plan.timer_unit)
                raise
            transitioned = None
        if transitioned is None:
            current = self._require_job(job.job_id)
            if current.status != ScheduledJobStatus.PAUSED.value:
                self._best_effort_timer_shutdown(plan.timer_unit)
                raise ScheduledJobProvisioningConflictError(
                    "The unsafe active job changed before it could be paused."
                )
        try:
            paused = self._require_job(job.job_id)
        except (ScheduledJobPersistenceError, ScheduledJobValidationError):
            self._best_effort_timer_shutdown(plan.timer_unit)
            raise
        cleanup_warning = self._deactivate_timer(
            job=paused,
            timer_unit=plan.timer_unit,
        )
        warning = safe_scheduled_job_error_message(
            f"Active job was paused during reconciliation: {reason}"
        )
        if cleanup_warning:
            warning = safe_scheduled_job_error_message(
                f"{warning}; timer cleanup failed: {cleanup_warning}"
            )
        self._record_external_error(job.job_id, warning)
        current = self._require_job(job.job_id)
        return ScheduledJobProvisioningResult(
            action="reconcile",
            job=current,
            plan=plan,
            changed=True,
            warning=warning,
        )

    def _reconcile_active_job(
        self,
        *,
        job: ScheduledJobView,
        plan: ProvisioningPlan,
        action: str,
    ) -> ScheduledJobProvisioningResult:
        """Ensure an active DB job has installed, enabled, and started units."""

        job = self._repair_active_timer_receipt(
            job_id=job.job_id,
            timer_unit=plan.timer_unit,
        )
        snapshots: list[UnitFileSnapshotProtocol] = []
        try:
            self._install_and_enable(
                plan,
                snapshots,
                reset_existing_timer=False,
            )
            self._controller.start(plan.timer_unit)
        except Exception as exc:
            warning = safe_scheduled_job_error_message(exc)
            compensation_warning = self._compensate_external(
                plan,
                tuple(snapshots),
            )
            warning = self._combined_warning(warning, compensation_warning)
            current = self._require_job(job.job_id)
            if current.status == ScheduledJobStatus.ACTIVE.value:
                transitioned = self._repository_call(
                    "fail closed during active timer reconciliation",
                    self._repository.pause_job,
                    job_id=current.job_id,
                    expected_status=current.status,
                    expected_updated_at=current.updated_at,
                )
                if transitioned is None:
                    raise ScheduledJobProvisioningConflictError(
                        "The job changed while failed reconciliation was recorded."
                    ) from exc
            self._record_external_error(job.job_id, warning)
            raise ScheduledJobProvisioningError(
                "Active timer reconciliation failed; the job was made inactive."
            ) from exc
        self._clear_external_error(job.job_id)
        current = self._require_job(job.job_id)
        return ScheduledJobProvisioningResult(
            action=action,
            job=current,
            plan=plan,
            changed=True,
        )

    def _repair_active_timer_receipt(
        self,
        *,
        job_id: str,
        timer_unit: str,
    ) -> ScheduledJobView:
        """CAS-repair and confirm the active receipt before systemd mutation."""

        for _ in range(2):
            current = self._require_job(job_id)
            if current.status != ScheduledJobStatus.ACTIVE.value:
                raise ScheduledJobProvisioningConflictError(
                    "The job became inactive while its timer was reconciled."
                )
            if current.timer_unit_name == timer_unit:
                return current
            try:
                repaired = self._repository_call(
                    "repair the active job timer receipt",
                    self._repository.repair_timer_unit_name,
                    job_id=current.job_id,
                    expected_status=current.status,
                    expected_updated_at=current.updated_at,
                    timer_unit_name=timer_unit,
                )
            except ScheduledJobPersistenceError:
                confirmed = self._require_job(job_id)
                if (
                    confirmed.status == ScheduledJobStatus.ACTIVE.value
                    and confirmed.timer_unit_name == timer_unit
                ):
                    return confirmed
                raise
            if repaired is not None:
                confirmed = self._require_job(job_id)
                if (
                    confirmed.status == ScheduledJobStatus.ACTIVE.value
                    and confirmed.timer_unit_name == timer_unit
                ):
                    return confirmed
                if confirmed.status != ScheduledJobStatus.ACTIVE.value:
                    raise ScheduledJobProvisioningConflictError(
                        "The job became inactive while its timer receipt was repaired."
                    )
        raise ScheduledJobProvisioningConflictError(
            "The active job changed while its timer receipt was repaired."
        )

    def _activate(
        self,
        *,
        job: ScheduledJobView,
        plan: ProvisioningPlan,
        action: str,
    ) -> ScheduledJobProvisioningResult:
        """Install, enable, CAS-activate, and start a job with compensation."""

        activated_at = job.created_at if action == "provision" else self._clock()
        if activated_at.tzinfo is None or activated_at.utcoffset() is None:
            raise ScheduledJobProvisioningError(
                "The activation boundary must be a timezone-aware timestamp."
            )
        activated_at = activated_at.astimezone(timezone.utc)
        one_time_run_at = _one_time_run_at(job.definition)
        if action == "provision" and one_time_run_at is not None:
            activated_at = min(activated_at, one_time_run_at)
        elif (
            action == "resume"
            and one_time_run_at is not None
            and one_time_run_at < activated_at
        ):
            raise ScheduledJobProvisioningStateError(
                "Cannot resume a one-time job after its scheduled time has elapsed."
            )
        snapshots: list[UnitFileSnapshotProtocol] = []
        try:
            self._install_and_enable(
                plan,
                snapshots,
                reset_existing_timer=True,
            )
        except Exception as exc:
            warning = safe_scheduled_job_error_message(exc)
            compensation_warning = self._compensate_external(
                plan,
                tuple(snapshots),
            )
            persisted_warning = self._combined_warning(warning, compensation_warning)
            if action == "provision":
                self._mark_provisioning_failed(job, persisted_warning)
            else:
                self._record_external_error(job.job_id, persisted_warning)
            detail = (
                " Compensation requires reconciliation." if compensation_warning else ""
            )
            raise ScheduledJobProvisioningError(
                f"Scheduled job {action} failed before activation.{detail}"
            ) from exc

        try:
            activation_operation = (
                self._repository.settle_and_activate_job
                if action == "resume"
                else self._repository.activate_job
            )
            activated = self._repository_call(
                "activate scheduled job",
                activation_operation,
                job_id=job.job_id,
                expected_status=job.status,
                expected_updated_at=job.updated_at,
                timer_unit_name=plan.timer_unit,
                activated_at=activated_at,
            )
        except ScheduledJobRunLeaseBusyError as exc:
            compensation_warning = self._compensate_external(
                plan,
                tuple(snapshots),
            )
            if compensation_warning:
                self._record_external_error(
                    job.job_id,
                    self._combined_warning(
                        "Resume was blocked by an in-flight execution lease.",
                        compensation_warning,
                    ),
                )
            detail = (
                " External compensation requires reconciliation."
                if compensation_warning
                else ""
            )
            raise ScheduledJobProvisioningStateError(
                "Cannot resume while the previous scheduled execution still "
                f"owns a live lease.{detail}"
            ) from exc
        except ScheduledJobPersistenceError as exc:
            try:
                current = self._require_job(job.job_id)
            except (ScheduledJobPersistenceError, ScheduledJobValidationError):
                compensation_warning = self._compensate_external(
                    plan,
                    tuple(snapshots),
                )
                detail = (
                    " Compensation requires reconciliation."
                    if compensation_warning
                    else ""
                )
                raise ScheduledJobProvisioningError(
                    f"Database activation could not be confirmed.{detail}"
                ) from exc
            if (
                current.status == ScheduledJobStatus.ACTIVE.value
                and current.timer_unit_name == plan.timer_unit
            ):
                activated = None
            else:
                compensation_warning = self._compensate_external(
                    plan,
                    tuple(snapshots),
                )
                if current.status == job.status:
                    warning = self._combined_warning(
                        safe_scheduled_job_error_message(exc),
                        compensation_warning,
                    )
                    if action == "provision":
                        self._mark_provisioning_failed(current, warning)
                    else:
                        self._record_external_error(current.job_id, warning)
                raise ScheduledJobProvisioningError(
                    "Database activation failed; the timer was compensated."
                ) from exc
        if activated is None:
            current = self._require_job(job.job_id)
            if not (
                current.status == ScheduledJobStatus.ACTIVE.value
                and current.timer_unit_name == plan.timer_unit
            ):
                compensation_warning = self._compensate_external(
                    plan,
                    tuple(snapshots),
                )
                detail = (
                    " Compensation requires reconciliation."
                    if compensation_warning
                    else ""
                )
                raise ScheduledJobProvisioningConflictError(
                    "The job changed while systemd units were being installed."
                    f"{detail}"
                )

        try:
            self._controller.start(plan.timer_unit)
        except Exception as exc:
            warning = safe_scheduled_job_error_message(exc)
            compensation_warning = self._compensate_external(
                plan,
                tuple(snapshots),
            )
            persisted_warning = self._combined_warning(warning, compensation_warning)
            current = self._require_job(job.job_id)
            if current.status == ScheduledJobStatus.ACTIVE.value:
                if action == "resume":
                    transitioned = self._repository_call(
                        "pause job after failed resume",
                        self._repository.pause_job,
                        job_id=current.job_id,
                        expected_status=current.status,
                        expected_updated_at=current.updated_at,
                    )
                    if transitioned is not None:
                        self._record_external_error(current.job_id, persisted_warning)
                else:
                    transitioned = self._repository_call(
                        "mark failed job after timer start failure",
                        self._repository.mark_provisioning_failed,
                        job_id=current.job_id,
                        expected_status=current.status,
                        expected_updated_at=current.updated_at,
                        error=persisted_warning,
                    )
                if transitioned is None:
                    raise ScheduledJobProvisioningConflictError(
                        "Timer start failed and the job changed during compensation."
                    ) from exc
            detail = (
                " Compensation requires reconciliation." if compensation_warning else ""
            )
            raise ScheduledJobProvisioningError(
                f"Scheduled job {action} failed while starting its timer.{detail}"
            ) from exc

        current = self._require_job(job.job_id)
        return ScheduledJobProvisioningResult(
            action=action,
            job=current,
            plan=plan,
            changed=True,
        )

    def _install_and_enable(
        self,
        plan: ProvisioningPlan,
        snapshots: list[UnitFileSnapshotProtocol],
        *,
        reset_existing_timer: bool,
    ) -> None:
        """Validate, install, optionally reset, and enable one managed timer."""

        for expression in plan.on_calendar:
            self._controller.validate_calendar(expression)
        service_artifact, timer_artifact = self._plan_artifacts(plan)
        self._require_shared_service_current(service_artifact)
        timer_snapshot = self._unit_store.install(
            timer_artifact.unit_name,
            timer_artifact.content,
        )
        snapshots.append(timer_snapshot)
        self._controller.verify(
            tuple(Path(str(artifact.destination_path)) for artifact in plan.artifacts)
        )
        self._controller.daemon_reload()
        if reset_existing_timer:
            failures = self._stop_and_disable_timer(plan.timer_unit)
            try:
                self._controller.clean_state(plan.timer_unit)
            except Exception as exc:
                failures.append(safe_scheduled_job_error_message(exc))
            if failures:
                raise ScheduledJobProvisioningError(
                    "Existing timer state could not be reset safely: "
                    + "; ".join(failures)
                )
        self._controller.enable(plan.timer_unit)

    def _require_shared_service_current(
        self,
        artifact: SystemdUnitArtifact,
    ) -> None:
        """Require the deployed shared service to exactly match this release."""

        shared_snapshot = self._unit_store.snapshot(artifact.unit_name)
        if not self._snapshot_matches(shared_snapshot, artifact.content):
            raise ScheduledJobProvisioningStateError(
                "The shared service template is missing or outdated; run the "
                "install-service deployment action first."
            )

    def _normalized_clock(self, field_name: str) -> datetime:
        """Return the injected clock as an aware UTC lifecycle timestamp."""

        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ScheduledJobProvisioningError(
                f"The {field_name} timestamp must include a timezone."
            )
        return value.astimezone(timezone.utc)

    @staticmethod
    def _append_reconciliation_issue(
        issues: list[ScheduledJobReconciliationIssue],
        issue: ScheduledJobReconciliationIssue,
    ) -> int:
        """Append one bounded batch issue and return one when it was omitted."""

        if len(issues) >= _MAX_RECONCILIATION_ISSUES:
            return 1
        issues.append(issue)
        return 0

    @staticmethod
    def _snapshot_matches(
        snapshot: UnitFileSnapshotProtocol,
        expected_content: str,
    ) -> bool:
        """Return whether a managed snapshot exactly matches rendered text."""

        content = snapshot.content
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                return False
        return content == expected_content

    @staticmethod
    def _plan_artifacts(
        plan: ProvisioningPlan,
    ) -> tuple[SystemdUnitArtifact, SystemdUnitArtifact]:
        """Return the exact shared service and per-job timer artifacts."""

        services = tuple(
            artifact
            for artifact in plan.artifacts
            if artifact.unit_name == SERVICE_TEMPLATE_UNIT_NAME
        )
        timers = tuple(
            artifact
            for artifact in plan.artifacts
            if artifact.unit_name == plan.timer_unit
        )
        if len(services) != 1 or len(timers) != 1 or len(plan.artifacts) != 2:
            raise ScheduledJobProvisioningError(
                "The provisioning plan does not contain the expected unit artifacts."
            )
        return services[0], timers[0]

    def _compensate_external(
        self,
        plan: ProvisioningPlan,
        snapshots: tuple[UnitFileSnapshotProtocol, ...],
    ) -> str | None:
        """Best-effort stop/disable/restore after an activation-path failure."""

        failures = self._stop_and_disable_timer(plan.timer_unit)
        if snapshots:
            try:
                self._restore_snapshots(snapshots)
            except Exception as exc:  # compensation error is returned to the caller
                failures.append(safe_scheduled_job_error_message(exc))
        try:
            self._controller.daemon_reload()
        except Exception as exc:
            failures.append(safe_scheduled_job_error_message(exc))
        return "; ".join(failures) or None

    @staticmethod
    def _combined_warning(primary: str, compensation: str | None) -> str:
        """Combine a primary failure with bounded compensation diagnostics."""

        if compensation is None:
            return primary
        return safe_scheduled_job_error_message(
            f"{primary}; compensation failed: {compensation}"
        )

    def _restore_snapshots(
        self,
        snapshots: tuple[UnitFileSnapshotProtocol, ...],
    ) -> None:
        """Restore installed unit files in reverse order."""

        first_error: Exception | None = None
        for snapshot in reversed(snapshots):
            try:
                self._unit_store.restore(snapshot)
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def _deactivate_timer(
        self,
        *,
        job: ScheduledJobView,
        timer_unit: str,
    ) -> str | None:
        """Stop and disable a timer, retaining an error while DB stays inactive."""

        control_error = self._control_support_error()
        if control_error is not None:
            warning = safe_scheduled_job_error_message(
                f"Timer cleanup is unavailable: {control_error}"
            )
            if job.provisioning_error:
                warning = safe_scheduled_job_error_message(
                    f"{job.provisioning_error}; timer cleanup failed: {warning}"
                )
            self._record_external_error(job.job_id, warning)
            return warning
        failures = self._stop_and_disable_timer(timer_unit)
        if failures:
            warning = safe_scheduled_job_error_message("; ".join(failures))
            if job.provisioning_error:
                warning = safe_scheduled_job_error_message(
                    f"{job.provisioning_error}; timer cleanup failed: {warning}"
                )
            self._record_external_error(job.job_id, warning)
            return warning
        if job.status != ScheduledJobStatus.PROVISIONING_FAILED.value:
            self._clear_external_error(job.job_id)
        return None

    def _remove_timer(
        self,
        *,
        job: ScheduledJobView,
        timer_unit: str,
    ) -> str | None:
        """Disable, clean, and remove only the exact UUID-derived managed timer."""

        control_error = self._control_support_error()
        if control_error is not None:
            warning = safe_scheduled_job_error_message(
                f"Timer cleanup is unavailable: {control_error}"
            )
            self._record_external_error(job.job_id, warning)
            return warning
        try:
            snapshot = self._unit_store.snapshot(timer_unit)
        except Exception as exc:
            warning = safe_scheduled_job_error_message(exc)
            self._record_external_error(job.job_id, warning)
            return warning

        failures = self._stop_and_disable_timer(timer_unit)
        clean_succeeded = True
        if snapshot.existed:
            try:
                self._controller.clean_state(timer_unit)
            except Exception as exc:
                failures.append(safe_scheduled_job_error_message(exc))
                clean_succeeded = False
            if clean_succeeded:
                try:
                    self._unit_store.remove(timer_unit)
                except Exception as exc:
                    failures.append(safe_scheduled_job_error_message(exc))
        try:
            self._controller.daemon_reload()
        except Exception as exc:
            failures.append(safe_scheduled_job_error_message(exc))

        if failures:
            warning = safe_scheduled_job_error_message("; ".join(failures))
            self._record_external_error(job.job_id, warning)
            return warning
        self._clear_external_error(job.job_id)
        return None

    def _control_support_error(self) -> str | None:
        """Return a safe host-control error while allowing DB-first cleanup."""

        try:
            self._host_probe.ensure_control_supported(self._settings)
        except Exception as exc:
            return safe_scheduled_job_error_message(exc)
        return None

    def _best_effort_timer_shutdown(self, timer_unit: str) -> str | None:
        """Quiesce a timer without requiring a follow-up database write."""

        control_error = self._control_support_error()
        if control_error is not None:
            return control_error
        failures = self._stop_and_disable_timer(timer_unit)
        return (
            safe_scheduled_job_error_message("; ".join(failures)) if failures else None
        )

    def _stop_and_disable_timer(self, timer_unit: str) -> list[str]:
        """Best-effort stop and disable, attempting disable even when stop fails."""

        failures: list[str] = []
        try:
            should_stop = self._controller.is_active(timer_unit)
        except Exception:
            should_stop = True
        if should_stop:
            try:
                self._controller.stop(timer_unit)
            except Exception as exc:
                failures.append(safe_scheduled_job_error_message(exc))

        try:
            should_disable = self._controller.is_enabled(timer_unit)
        except Exception:
            should_disable = True
        if should_disable:
            try:
                self._controller.disable(timer_unit)
            except Exception as exc:
                failures.append(safe_scheduled_job_error_message(exc))
        return failures

    def _mark_provisioning_failed(
        self,
        job: ScheduledJobView,
        warning: str,
    ) -> None:
        """CAS a provisioning attempt into the inactive failed state."""

        transitioned = self._repository_call(
            "mark scheduled job provisioning failed",
            self._repository.mark_provisioning_failed,
            job_id=job.job_id,
            expected_status=job.status,
            expected_updated_at=job.updated_at,
            error=warning,
        )
        if transitioned is None:
            raise ScheduledJobProvisioningConflictError(
                "The job changed while its provisioning failure was recorded."
            )

    def _record_external_error(self, job_id: str, warning: str) -> None:
        """CAS a sanitized warning, retrying once after a concurrent update."""

        for _ in range(2):
            current = self._require_job(job_id)
            if current.provisioning_error == warning:
                return
            try:
                transitioned = self._repository_call(
                    "record scheduled job external error",
                    self._repository.record_external_error,
                    job_id=current.job_id,
                    expected_status=current.status,
                    expected_updated_at=current.updated_at,
                    error=warning,
                )
            except ScheduledJobPersistenceError:
                confirmed = self._require_job(job_id)
                if confirmed.provisioning_error == warning:
                    return
                raise
            if transitioned is not None:
                return
        raise ScheduledJobProvisioningConflictError(
            "The job changed while its external scheduler warning was recorded."
        )

    def _clear_external_error(self, job_id: str) -> None:
        """Clear a stale warning, retrying once after a concurrent update."""

        for _ in range(2):
            current = self._require_job(job_id)
            if current.provisioning_error is None:
                return
            try:
                transitioned = self._repository_call(
                    "clear scheduled job external error",
                    self._repository.clear_external_error,
                    job_id=current.job_id,
                    expected_status=current.status,
                    expected_updated_at=current.updated_at,
                )
            except ScheduledJobPersistenceError:
                confirmed = self._require_job(job_id)
                if confirmed.provisioning_error is None:
                    return
                raise
            if transitioned is not None:
                return
        raise ScheduledJobProvisioningConflictError(
            "The job changed while its external scheduler warning was cleared."
        )

    @staticmethod
    def _repository_call(
        description: str,
        operation: Callable[..., ScheduledJob | None],
        **kwargs: object,
    ) -> ScheduledJob | None:
        """Run one short repository transition and normalize database failures."""

        try:
            return operation(**kwargs)
        except SQLAlchemyError as exc:
            raise ScheduledJobPersistenceError(
                f"The database could not {description}."
            ) from exc


__all__ = [
    "ControlSyncUnitsResult",
    "FileJobOperationLock",
    "LinuxSystemdHostProbe",
    "ScheduledJobCleanupPlan",
    "ScheduledJobExecutorNotReadyError",
    "ScheduledJobProvisioningConflictError",
    "ScheduledJobProvisioningDisabledError",
    "ScheduledJobProvisioningError",
    "ScheduledJobProvisioningPolicy",
    "ScheduledJobReconciliationBatchResult",
    "ScheduledJobReconciliationIssue",
    "ScheduledJobProvisioningResult",
    "ScheduledJobProvisioningService",
    "ScheduledJobProvisioningStateError",
    "ScheduledJobUnsupportedHostError",
    "SharedServiceTemplateResult",
]
