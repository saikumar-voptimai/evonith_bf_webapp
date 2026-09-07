"""Tests for safe Phase 4 scheduled-job provisioning orchestration."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from data.scheduled_tasks import (
    FileJobOperationLock,
    LinuxSystemdHostProbe,
    ScheduledJobCreateRequest,
    ScheduledJobExecutorNotReadyError,
    ScheduledJobPersistenceError,
    ScheduledJobProvisioningConflictError,
    ScheduledJobProvisioningDisabledError,
    ScheduledJobProvisioningError,
    ScheduledJobProvisioningPolicy,
    ScheduledJobProvisioningService,
    ScheduledJobProvisioningStateError,
    ScheduledJobService,
    ScheduledJobUnsupportedHostError,
    ScheduledJobValidationError,
)
from furnace_data.relational import (
    ScheduledJob,
    ScheduledJobOutput,
    ScheduledJobRepository,
    ScheduledJobRevision,
    ScheduledJobRun,
    ScheduledJobRunLog,
    ScheduledJobRunStatus,
    ScheduledJobStatus,
)
from utils.scheduled_tasks.schedule_occurrence import latest_scheduled_occurrence
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)
from utils.scheduled_tasks.systemd_units import (
    CONTROL_SYNC_SERVICE_UNIT_NAME,
    CONTROL_SYNC_TIMER_UNIT_NAME,
    SERVICE_TEMPLATE_UNIT_NAME,
    SystemdProvisioningSettings,
    render_service_template,
    timer_unit_name,
)

_USER_ID = "00000000-0000-0000-0000-000000000001"
_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


def _valid_definition() -> dict[str, object]:
    """Build one representative definition through the production contract."""

    return build_task_definition(
        ScheduledTaskInput(
            name="Hourly BF2 review",
            instructions="Review BF2 operation and summarize important changes.",
            furnace="BF2",
            data_period="last_24_hours",
            output_format="operator_summary",
            schedule_kind="hourly",
            delivery_channel="in_app",
            job_type="furnace_summary",
            hourly_minute=5,
        ),
        generated_at=_NOW,
    )


def _past_one_time_definition() -> dict[str, object]:
    """Build a one-time definition due before the provisioning test clock."""

    return build_task_definition(
        ScheduledTaskInput(
            name="One-time BF2 review",
            instructions="Review BF2 once at the requested time.",
            furnace="BF2",
            data_period="last_24_hours",
            output_format="operator_summary",
            schedule_kind="once",
            delivery_channel="in_app",
            job_type="furnace_summary",
            run_date=date(2026, 9, 5),
            run_time=time(10, 0),
        ),
        generated_at=datetime(2026, 9, 5, 3, 30, tzinfo=timezone.utc),
    )


def _external_delivery_definition(channel: str) -> dict[str, object]:
    """Build a portable definition using an intentionally unsupported channel."""

    definition = _valid_definition()
    if channel == "email":
        definition["delivery"] = {
            "channel": "email",
            "notify_on_failure": True,
            "recipients": ["operator@example.com"],
            "subject": "BF2 report",
            "attachments": [],
        }
    elif channel == "whatsapp":
        definition["delivery"] = {
            "channel": "whatsapp",
            "notify_on_failure": True,
            "destination": "+919876543210",
        }
    else:
        raise ValueError(f"Unsupported test delivery channel: {channel}")
    return definition


def _fixed_clock() -> datetime:
    """Return the deterministic activation timestamp used by lifecycle tests."""

    return _NOW


def _windows_system() -> str:
    """Return the platform name used by the unsupported-host test."""

    return "Windows"


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """In-memory prior unit state used by the fake file store."""

    unit_name: str
    content: str | None

    @property
    def existed(self) -> bool:
        """Return whether the fake unit existed before the operation."""

        return self.content is not None


class _FakeUnitStore:
    """Record reversible unit changes without touching the filesystem."""

    def __init__(self, events: list[str]) -> None:
        """Create an empty fake store sharing an ordered event log."""

        self.events = events
        self.files: dict[str, str] = {}
        self.fail_operation: str | None = None

    def _maybe_fail(self, operation: str) -> None:
        """Raise a deterministic credential-bearing error for redaction tests."""

        if self.fail_operation == operation:
            raise OSError(f"{operation} failed password=hunter2")

    def snapshot(self, unit_name: str) -> _Snapshot:
        """Return the current in-memory state for one unit."""

        self.events.append(f"store:snapshot:{unit_name}")
        self._maybe_fail("snapshot")
        return _Snapshot(unit_name, self.files.get(unit_name))

    def install(self, unit_name: str, content: str) -> _Snapshot:
        """Install one fake unit and return its prior state."""

        self.events.append(f"store:install:{unit_name}")
        self._maybe_fail("install")
        previous = _Snapshot(unit_name, self.files.get(unit_name))
        self.files[unit_name] = content
        return previous

    def restore(self, snapshot: _Snapshot) -> None:
        """Restore or remove a fake unit from its snapshot."""

        self.events.append(f"store:restore:{snapshot.unit_name}")
        self._maybe_fail("restore")
        if snapshot.content is None:
            self.files.pop(snapshot.unit_name, None)
        else:
            self.files[snapshot.unit_name] = snapshot.content

    def remove(self, unit_name: str) -> _Snapshot:
        """Remove one fake unit and return its prior state."""

        self.events.append(f"store:remove:{unit_name}")
        self._maybe_fail("remove")
        previous = _Snapshot(unit_name, self.files.get(unit_name))
        self.files.pop(unit_name, None)
        return previous


class _FakeController:
    """Record fixed systemd operations and support fault injection."""

    def __init__(self, events: list[str]) -> None:
        """Create a controller with empty enabled and active sets."""

        self.events = events
        self.enabled: set[str] = set()
        self.active: set[str] = set()
        self.persistent_state: set[str] = set()
        self.fail_operation: str | None = None
        self.fail_once_operation: str | None = None

    def _record(self, operation: str) -> None:
        """Record an operation and raise when it is selected for failure."""

        self.events.append(f"systemd:{operation}")
        if self.fail_once_operation == operation:
            self.fail_once_operation = None
            raise RuntimeError(f"{operation} failed token=super-secret-token")
        if self.fail_operation == operation:
            raise RuntimeError(f"{operation} failed token=super-secret-token")

    def validate_calendar(self, expression: str) -> str:
        """Record one calendar validation."""

        self._record(f"calendar:{expression}")
        return expression

    def verify(self, paths: tuple[Path, ...]) -> None:
        """Record unit verification without requiring real files."""

        self._record("verify")
        assert len(paths) in {1, 2}

    def daemon_reload(self) -> None:
        """Record a systemd daemon reload."""

        self._record("reload")

    def enable(self, unit_name: str) -> None:
        """Record and apply fake enablement."""

        self._record("enable")
        self.enabled.add(unit_name)

    def start(self, unit_name: str) -> None:
        """Record and apply fake activation."""

        self._record("start")
        self.active.add(unit_name)

    def stop(self, unit_name: str) -> None:
        """Record and apply fake deactivation."""

        self._record("stop")
        self.active.discard(unit_name)

    def disable(self, unit_name: str) -> None:
        """Record and apply fake disablement."""

        self._record("disable")
        self.enabled.discard(unit_name)

    def clean_state(self, unit_name: str) -> None:
        """Record removal of fake persistent timer state."""

        self._record("clean")
        self.persistent_state.discard(unit_name)

    def is_enabled(self, unit_name: str) -> bool:
        """Return the fake enabled state after recording its query."""

        self._record("is-enabled")
        return unit_name in self.enabled

    def is_active(self, unit_name: str) -> bool:
        """Return the fake active state after recording its query."""

        self._record("is-active")
        return unit_name in self.active


class _FakeHostProbe:
    """Record that the real-host mutation gate was evaluated."""

    def __init__(self, events: list[str]) -> None:
        """Create a probe sharing an ordered event log."""

        self.events = events
        self.fail_control = False
        self.fail_activation = False
        self.activation_error: Exception | None = None

    def ensure_control_supported(self, settings: SystemdProvisioningSettings) -> None:
        """Accept cleanup operations without inspecting the Windows host."""

        assert isinstance(settings, SystemdProvisioningSettings)
        self.events.append("host:validated")
        if self.fail_control:
            raise ScheduledJobUnsupportedHostError("systemctl is unavailable")

    def ensure_activation_supported(
        self,
        settings: SystemdProvisioningSettings,
    ) -> None:
        """Accept activation operations without inspecting the Windows host."""

        assert isinstance(settings, SystemdProvisioningSettings)
        self.events.append("host:validated")
        if self.fail_control:
            raise ScheduledJobUnsupportedHostError("systemctl is unavailable")
        if self.activation_error is not None:
            raise self.activation_error
        if self.fail_activation:
            raise ScheduledJobUnsupportedHostError("executor runtime is unavailable")


class _FakeOperationLock:
    """Record a per-job critical section without using platform file locks."""

    def __init__(self, events: list[str]) -> None:
        """Create a lock sharing an ordered event log."""

        self.events = events

    @contextmanager
    def hold(self, job_id: str) -> Iterator[None]:
        """Enter and exit one recorded critical section."""

        self.events.append(f"lock:enter:{job_id}")
        try:
            yield
        finally:
            self.events.append(f"lock:exit:{job_id}")


class _RecordingRepository:
    """Delegate persistence while recording lifecycle transition ordering."""

    def __init__(self, repository: ScheduledJobRepository, events: list[str]) -> None:
        """Wrap a real repository and share the orchestration event log."""

        self._repository = repository
        self._events = events
        self.raise_after_activation = False
        self.raise_before_receipt_repair = False
        self.raise_after_receipt_repair = False
        self.reject_receipt_repair = False
        self.fail_reconciliation_page = False
        self.raise_on_failure_transition = False
        self.raise_before_pause = False
        self.raise_after_pause = False

    def create_job(self, **kwargs: Any) -> ScheduledJob:
        """Delegate scheduled-job creation."""

        return self._repository.create_job(**kwargs)

    def get_job(self, job_id: str) -> ScheduledJob | None:
        """Delegate scheduled-job lookup."""

        return self._repository.get_job(job_id)

    def reconciliation_cutoff(self) -> datetime:
        """Delegate the database-clock scan cutoff."""

        return self._repository.reconciliation_cutoff()

    def list_reconciliation_job_ids(self, **kwargs: Any) -> list[tuple[str, datetime]]:
        """Delegate one keyset reconciliation page."""

        if self.fail_reconciliation_page:
            raise SQLAlchemyError("reconciliation page unavailable")
        return self._repository.list_reconciliation_job_ids(**kwargs)

    def activate_job(self, **kwargs: Any) -> ScheduledJob | None:
        """Record and delegate job activation."""

        self._events.append("db:activate")
        result = self._repository.activate_job(**kwargs)
        if self.raise_after_activation:
            raise SQLAlchemyError("database response was lost after commit")
        return result

    def settle_and_activate_job(self, **kwargs: Any) -> object | None:
        """Record and delegate atomic expired-run settlement during resume."""

        self._events.append("db:settle-and-activate")
        return self._repository.settle_and_activate_job(**kwargs)

    def repair_timer_unit_name(self, **kwargs: Any) -> ScheduledJob | None:
        """Record and delegate active timer receipt repair."""

        self._events.append("db:repair-timer-receipt")
        if self.raise_before_receipt_repair:
            raise SQLAlchemyError("database rejected receipt update")
        if self.reject_receipt_repair:
            return None
        result = self._repository.repair_timer_unit_name(**kwargs)
        if self.raise_after_receipt_repair:
            raise SQLAlchemyError("database response was lost after receipt commit")
        return result

    def mark_provisioning_failed(self, **kwargs: Any) -> ScheduledJob | None:
        """Record and delegate the provisioning-failed transition."""

        self._events.append("db:failed")
        if self.raise_on_failure_transition:
            raise SQLAlchemyError("failure transition unavailable")
        return self._repository.mark_provisioning_failed(**kwargs)

    def pause_job(self, **kwargs: Any) -> ScheduledJob | None:
        """Record and delegate job pause."""

        self._events.append("db:pause")
        if self.raise_before_pause:
            raise SQLAlchemyError("database rejected pause before commit")
        result = self._repository.pause_job(**kwargs)
        if self.raise_after_pause:
            raise SQLAlchemyError("database response was lost after pause commit")
        return result

    def mark_deleted(self, **kwargs: Any) -> ScheduledJob | None:
        """Record and delegate soft deletion."""

        self._events.append("db:delete")
        return self._repository.mark_deleted(**kwargs)

    def record_external_error(self, **kwargs: Any) -> ScheduledJob | None:
        """Record and delegate an external warning update."""

        self._events.append("db:error")
        return self._repository.record_external_error(**kwargs)

    def clear_external_error(self, **kwargs: Any) -> ScheduledJob | None:
        """Record and delegate external warning cleanup."""

        self._events.append("db:clear-error")
        return self._repository.clear_external_error(**kwargs)


@pytest.fixture
def provisioning_stack():
    """Create a real lifecycle database with fake systemd side effects."""

    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        execution_options={
            "schema_translate_map": {"automation": None, "identity": None}
        },
    )
    ScheduledJob.__table__.create(engine)
    ScheduledJobRevision.__table__.create(engine)
    ScheduledJobRun.__table__.create(engine)
    ScheduledJobRunLog.__table__.create(engine)
    ScheduledJobOutput.__table__.create(engine)
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
        future=True,
    )
    events: list[str] = []
    repository = _RecordingRepository(ScheduledJobRepository(factory), events)
    job_service = ScheduledJobService(repository=repository)  # type: ignore[arg-type]
    controller = _FakeController(events)
    unit_store = _FakeUnitStore(events)
    settings = SystemdProvisioningSettings()
    unit_store.files[SERVICE_TEMPLATE_UNIT_NAME] = render_service_template(settings)
    provisioner = ScheduledJobProvisioningService(
        settings=settings,
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy(
            systemd_mutations_enabled=True,
            production_executor_ready=True,
        ),
        repository=repository,  # type: ignore[arg-type]
        host_probe=_FakeHostProbe(events),
        operation_lock=_FakeOperationLock(events),
        clock=_fixed_clock,
        executor_capability_ready=True,
    )
    try:
        yield provisioner, job_service, controller, unit_store, events, factory
    finally:
        provisioner.dispose()
        engine.dispose()


def _create_job(job_service: ScheduledJobService):
    """Persist one valid pending scheduled job for a lifecycle test."""

    return job_service.create_job(
        ScheduledJobCreateRequest(
            definition=_valid_definition(),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )


def _create_past_one_time_job(job_service: ScheduledJobService):
    """Persist a one-time task whose 10:00 IST occurrence is already due."""

    return job_service.create_job(
        ScheduledJobCreateRequest(
            definition=_past_one_time_definition(),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )


def _insert_running_run(
    factory: sessionmaker[Session],
    *,
    job_id: str,
    lease_expires_at: datetime,
) -> UUID:
    """Insert one constraint-valid running occurrence for resume fencing tests."""

    with factory() as session:
        job = session.get(ScheduledJob, job_id)
        assert job is not None
        run = ScheduledJobRun(
            job_id=job_id,
            scheduled_for=_NOW - timedelta(hours=1),
            status=ScheduledJobRunStatus.RUNNING.value,
            attempt_number=1,
            activation_generation=job.activation_generation,
            lease_token=uuid4(),
            lease_expires_at=lease_expires_at,
            attempt_deadline_at=datetime(2100, 1, 1, tzinfo=timezone.utc),
            started_at=_NOW - timedelta(minutes=5),
            triggered_by="systemd",
            created_at=_NOW - timedelta(minutes=5),
        )
        session.add(run)
        session.commit()
        return run.run_id


def test_plan_is_cross_platform_and_has_no_mutating_side_effects(
    provisioning_stack,
) -> None:
    """Planning should only read, validate, and render a pending definition."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    events.clear()

    result = provisioner.plan(job.job_id)

    assert result.dry_run is True
    assert result.changed is False
    assert result.job.status == ScheduledJobStatus.PENDING_PROVISIONING.value
    assert result.plan.timer_unit == timer_unit_name(job.job_id)
    assert events == []
    assert controller.enabled == set()
    assert set(unit_store.files) == {SERVICE_TEMPLATE_UNIT_NAME}


@pytest.mark.parametrize("channel", ("email", "whatsapp"))
def test_external_delivery_is_rejected_before_any_systemd_mutation(
    provisioning_stack,
    channel: str,
) -> None:
    """Phase 5 must fail closed for schema-valid unimplemented delivery paths."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = job_service.create_job(
        ScheduledJobCreateRequest(
            definition=_external_delivery_definition(channel),
            created_by_user_id=_USER_ID,
            created_by_username="operator.test",
        )
    )
    original_files = dict(unit_store.files)
    events.clear()

    with pytest.raises(ScheduledJobValidationError, match="only in_app"):
        provisioner.provision(job.job_id)

    assert controller.enabled == set()
    assert controller.active == set()
    assert unit_store.files == original_files
    assert not any(event.startswith("systemd:") for event in events)
    assert not any(event.startswith("store:") for event in events)
    assert job_service.get_job(job.job_id).status == (
        ScheduledJobStatus.PENDING_PROVISIONING.value
    )


def test_shared_service_template_is_an_explicit_idempotent_deployment_action(
    provisioning_stack,
) -> None:
    """The global service must be installed separately from per-job timers."""

    provisioner, _, _, unit_store, events, _ = provisioning_stack
    unit_store.files.clear()
    events.clear()

    preview = provisioner.install_service_template(dry_run=True)
    assert preview.dry_run is True
    assert preview.artifact.unit_name == SERVICE_TEMPLATE_UNIT_NAME
    assert unit_store.files == {}
    assert events == []

    installed = provisioner.install_service_template()
    assert installed.changed is True
    assert SERVICE_TEMPLATE_UNIT_NAME in unit_store.files
    assert "systemd:verify" in events
    assert "systemd:reload" in events
    assert "systemd:enable" not in events
    assert "systemd:start" not in events

    events.clear()
    repeated = provisioner.install_service_template()
    assert repeated.changed is False
    assert not any(event.startswith("store:install") for event in events)
    assert "systemd:verify" in events
    assert "systemd:reload" in events


def test_control_sync_units_install_and_start_as_one_deployment_action(
    provisioning_stack,
) -> None:
    """Deployment should atomically install and start the queue bridge timer."""

    provisioner, _, controller, unit_store, events, _ = provisioning_stack
    events.clear()

    preview = provisioner.install_control_sync_units(dry_run=True)
    assert preview.dry_run is True
    assert [artifact.unit_name for artifact in preview.artifacts] == [
        CONTROL_SYNC_SERVICE_UNIT_NAME,
        CONTROL_SYNC_TIMER_UNIT_NAME,
    ]
    assert CONTROL_SYNC_SERVICE_UNIT_NAME not in unit_store.files

    installed = provisioner.install_control_sync_units()
    assert installed.changed is True
    assert CONTROL_SYNC_SERVICE_UNIT_NAME in unit_store.files
    assert CONTROL_SYNC_TIMER_UNIT_NAME in unit_store.files
    assert CONTROL_SYNC_TIMER_UNIT_NAME in controller.enabled
    assert CONTROL_SYNC_TIMER_UNIT_NAME in controller.active
    assert "systemd:verify" in events
    assert "systemd:reload" in events
    assert "systemd:enable" in events
    assert "systemd:start" in events


@pytest.mark.parametrize("shared_content", [None, "outdated managed service"])
def test_job_provisioning_requires_the_current_shared_service_template(
    provisioning_stack,
    shared_content: str | None,
) -> None:
    """Per-job work must fail before timer installation if deployment is stale."""

    provisioner, job_service, _, unit_store, events, _ = provisioning_stack
    if shared_content is None:
        unit_store.files.pop(SERVICE_TEMPLATE_UNIT_NAME)
    else:
        unit_store.files[SERVICE_TEMPLATE_UNIT_NAME] = shared_content
    job = _create_job(job_service)
    events.clear()

    with pytest.raises(ScheduledJobProvisioningError, match="before activation"):
        provisioner.provision(job.job_id)

    current = job_service.get_job(job.job_id)
    assert current.status == ScheduledJobStatus.PROVISIONING_FAILED.value
    assert timer_unit_name(job.job_id) not in unit_store.files
    assert f"store:install:{SERVICE_TEMPLATE_UNIT_NAME}" not in events


def test_default_policy_blocks_mutation_before_host_or_systemd_calls(
    provisioning_stack,
) -> None:
    """Deployment must explicitly enable both mutation and the real executor."""

    _, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    blocked = ScheduledJobProvisioningService(
        settings=SystemdProvisioningSettings(),
        controller=controller,
        unit_store=unit_store,
        repository=job_service._repository,  # noqa: SLF001
        host_probe=_FakeHostProbe(events),
        operation_lock=_FakeOperationLock(events),
    )
    events.clear()

    with pytest.raises(ScheduledJobProvisioningDisabledError):
        blocked.provision(job.job_id)

    assert events == []
    assert job_service.get_job(job.job_id).status == "pending_provisioning"


def test_real_host_probe_rejects_windows_before_inspecting_linux_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-systemd development host must never be mistaken for the target."""

    monkeypatch.setattr("platform.system", _windows_system)
    probe = LinuxSystemdHostProbe(manager_directory=tmp_path)

    with pytest.raises(ScheduledJobUnsupportedHostError, match="Linux host"):
        probe.ensure_control_supported(SystemdProvisioningSettings())


@pytest.mark.parametrize(
    ("user_id", "primary_group_id", "configured_group_id"),
    [(0, 1000, 1000), (1000, 0, 1000), (1000, 1000, 0)],
)
def test_activation_probe_rejects_accounts_resolving_to_root_identity(
    monkeypatch: pytest.MonkeyPatch,
    user_id: int,
    primary_group_id: int,
    configured_group_id: int,
) -> None:
    """Aliases resolving to root IDs must not bypass account-name validation."""

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda _path: SimpleNamespace(st_uid=0, st_mode=0o640),
    )
    monkeypatch.setattr("os.access", lambda _path, _mode: True)
    monkeypatch.setitem(
        sys.modules,
        "pwd",
        SimpleNamespace(
            getpwnam=lambda _name: SimpleNamespace(
                pw_uid=user_id,
                pw_gid=primary_group_id,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "grp",
        SimpleNamespace(
            getgrnam=lambda _name: SimpleNamespace(gr_gid=configured_group_id)
        ),
    )
    probe = LinuxSystemdHostProbe(manager_directory=Path("/run/systemd/system"))
    settings = SystemdProvisioningSettings(
        service_user="furnacemind-alias",
        service_group="furnacemind-alias",
    )

    with pytest.raises(ScheduledJobUnsupportedHostError, match="unprivileged"):
        probe.ensure_activation_supported(settings)


def test_activation_probe_normalizes_host_filesystem_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host inspection failures must remain stable provisioning errors."""

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr("os.access", lambda _path, _mode: True)

    def _raise_stat_error(_path: Path) -> os.stat_result:
        """Simulate an environment-file metadata lookup failure."""

        raise OSError("metadata unavailable")

    monkeypatch.setattr(Path, "stat", _raise_stat_error)
    probe = LinuxSystemdHostProbe(manager_directory=Path("/run/systemd/system"))

    with pytest.raises(ScheduledJobUnsupportedHostError, match="inspected safely"):
        probe.ensure_activation_supported(SystemdProvisioningSettings())


@pytest.mark.parametrize(
    ("owner_id", "mode"),
    [(1000, stat.S_IFDIR | 0o750), (0, stat.S_IFDIR | 0o770)],
)
def test_file_operation_lock_rejects_untrusted_directory_status(
    owner_id: int,
    mode: int,
) -> None:
    """Lock serialization must not rely on a replaceable directory."""

    with pytest.raises(ScheduledJobUnsupportedHostError, match="not trusted"):
        FileJobOperationLock._validate_directory_status(  # noqa: SLF001
            SimpleNamespace(st_uid=owner_id, st_mode=mode)
        )


@pytest.mark.parametrize(
    ("owner_id", "mode", "link_count"),
    [
        (1000, stat.S_IFREG | 0o600, 1),
        (0, stat.S_IFREG | 0o620, 1),
        (0, stat.S_IFREG | 0o600, 2),
    ],
)
def test_file_operation_lock_rejects_untrusted_open_file_status(
    owner_id: int,
    mode: int,
    link_count: int,
) -> None:
    """An opened lock inode must remain a root-owned non-writable regular file."""

    with pytest.raises(ScheduledJobUnsupportedHostError, match="not trusted"):
        FileJobOperationLock._validate_lock_file_status(  # noqa: SLF001
            SimpleNamespace(st_uid=owner_id, st_mode=mode, st_nlink=link_count)
        )


def test_file_operation_lock_normalizes_acquisition_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock filesystem failures should reach the CLI as stable domain errors."""

    monkeypatch.setitem(
        sys.modules,
        "fcntl",
        SimpleNamespace(LOCK_EX=2, LOCK_UN=8, flock=lambda _fd, _operation: None),
    )

    def _raise_permission_error(*_args: object, **_kwargs: object) -> None:
        """Simulate a privileged lock-directory creation failure."""

        raise PermissionError("lock path denied")

    monkeypatch.setattr(Path, "mkdir", _raise_permission_error)
    operation_lock = FileJobOperationLock("/run/lock/furnacemind-test")

    with pytest.raises(ScheduledJobProvisioningError, match="could not be acquired"):
        with operation_lock.hold("00000000-0000-0000-0000-000000000000"):
            pytest.fail("The operation body must not run without its lock.")


def test_environment_policy_requires_explicit_activation_opt_ins() -> None:
    """Missing or ambiguous environment values must keep production disabled."""

    defaults = ScheduledJobProvisioningPolicy.from_environment({})
    enabled = ScheduledJobProvisioningPolicy.from_environment(
        {
            "SCHEDULED_TASKS_SYSTEMD_ENABLED": "true",
            "SCHEDULED_TASKS_EXECUTOR_READY": "1",
            "SCHEDULED_TASKS_TARGET_DEVICE_ID": "bf2-jetson-01",
        }
    )

    assert defaults.systemd_mutations_enabled is False
    assert defaults.production_executor_ready is False
    assert enabled.systemd_mutations_enabled is True
    assert enabled.production_executor_ready is True


def test_executor_gate_blocks_activation_but_allows_dry_run(
    provisioning_stack,
) -> None:
    """A validation-only runner must never receive an activated timer."""

    _, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    blocked = ScheduledJobProvisioningService(
        settings=SystemdProvisioningSettings(),
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy(systemd_mutations_enabled=True),
        repository=job_service._repository,  # noqa: SLF001
        host_probe=_FakeHostProbe(events),
        operation_lock=_FakeOperationLock(events),
    )

    dry_run = blocked.provision(job.job_id, dry_run=True)
    with pytest.raises(ScheduledJobExecutorNotReadyError):
        blocked.provision(job.job_id)

    assert dry_run.dry_run is True
    assert events == []
    assert job_service.get_job(job.job_id).status == "pending_provisioning"


def test_environment_flag_cannot_claim_a_missing_executor_capability(
    provisioning_stack,
) -> None:
    """Administrative opt-in alone must not make validation-only code runnable."""

    _, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    blocked = ScheduledJobProvisioningService(
        settings=SystemdProvisioningSettings(),
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy(
            systemd_mutations_enabled=True,
            production_executor_ready=True,
        ),
        repository=job_service._repository,  # noqa: SLF001
        host_probe=_FakeHostProbe(events),
        operation_lock=_FakeOperationLock(events),
    )
    events.clear()

    with pytest.raises(ScheduledJobExecutorNotReadyError):
        blocked.provision(job.job_id)

    assert events == []
    assert job_service.get_job(job.job_id).status == "pending_provisioning"


def test_reconcile_dry_run_reports_fail_closed_executor_policy(
    provisioning_stack,
) -> None:
    """Dry-run must show that an unsafe active job would be paused."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    blocked = ScheduledJobProvisioningService(
        settings=SystemdProvisioningSettings(),
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy(systemd_mutations_enabled=True),
        repository=job_service._repository,  # noqa: SLF001
        host_probe=_FakeHostProbe(events),
        operation_lock=_FakeOperationLock(events),
    )
    events.clear()

    result = blocked.reconcile(job.job_id, dry_run=True)

    assert result.dry_run is True
    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert result.warning is not None
    assert "executor is not ready" in result.warning
    assert result.plan.artifacts == ()
    assert events == []


def test_provision_installs_enables_activates_then_starts(
    provisioning_stack,
) -> None:
    """Successful provisioning must activate DB state before starting the timer."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    events.clear()

    result = provisioner.provision(job.job_id)

    timer = timer_unit_name(job.job_id)
    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert result.job.is_active is True
    assert result.job.activated_at == job.created_at
    assert result.job.timer_unit_name == timer
    assert timer in controller.enabled
    assert timer in controller.active
    assert set(unit_store.files) == {SERVICE_TEMPLATE_UNIT_NAME, timer}
    assert events.index("systemd:enable") < events.index("db:activate")
    assert events.index("db:activate") < events.index("systemd:start")
    assert f"store:install:{SERVICE_TEMPLATE_UNIT_NAME}" not in events


def test_delayed_one_time_provision_keeps_the_due_occurrence_eligible(
    provisioning_stack,
) -> None:
    """Initial provisioning must not lose a one-time run during setup delay."""

    provisioner, job_service, _, _, _, factory = provisioning_stack
    job = _create_past_one_time_job(job_service)
    created_after_run = datetime(2026, 9, 5, 4, 31, tzinfo=timezone.utc)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(created_at=created_after_run, updated_at=created_after_run)
        )
        session.commit()

    result = provisioner.provision(job.job_id)
    occurrence = latest_scheduled_occurrence(result.job.definition, at=_NOW)

    assert occurrence == datetime(2026, 9, 5, 4, 30, tzinfo=timezone.utc)
    assert result.job.activated_at == occurrence


def test_failed_one_time_provision_retry_keeps_due_occurrence_eligible(
    provisioning_stack,
) -> None:
    """Retrying provisioning after run_at must retain fire-once misfire behavior."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    job = _create_past_one_time_job(job_service)
    controller.fail_operation = "verify"
    with pytest.raises(ScheduledJobProvisioningError):
        provisioner.provision(job.job_id)
    assert (
        job_service.get_job(job.job_id).status
        == ScheduledJobStatus.PROVISIONING_FAILED.value
    )

    controller.fail_operation = None
    result = provisioner.provision(job.job_id)
    occurrence = latest_scheduled_occurrence(result.job.definition, at=_NOW)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert result.job.activated_at == occurrence


def test_one_time_retry_resets_a_stray_elapsed_timer_before_activation(
    provisioning_stack,
) -> None:
    """Retry must clear a pre-DB-crash timer stamp before activating the row."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_past_one_time_job(job_service)
    plan = provisioner.plan(job.job_id).plan
    timer = plan.timer_unit
    timer_artifact = next(
        artifact for artifact in plan.artifacts if artifact.unit_name == timer
    )
    unit_store.files[timer] = timer_artifact.content
    controller.enabled.add(timer)
    controller.active.add(timer)
    events.clear()

    result = provisioner.provision(job.job_id)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert timer in controller.enabled
    assert timer in controller.active
    assert events.index("systemd:stop") < events.index("systemd:clean")
    assert events.index("systemd:clean") < events.index("db:activate")
    assert events.index("db:activate") < events.index("systemd:start")


def test_one_time_retry_cleans_persistent_state_when_unit_file_is_absent(
    provisioning_stack,
) -> None:
    """An ambiguous first start must not let a stale stamp suppress its retry."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_past_one_time_job(job_service)
    timer = timer_unit_name(job.job_id)
    controller.persistent_state.add(timer)
    assert timer not in unit_store.files
    events.clear()

    result = provisioner.provision(job.job_id)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert timer not in controller.persistent_state
    assert "systemd:clean" in events
    assert events.index("systemd:clean") < events.index("db:activate")


def test_elapsed_one_time_job_cannot_be_resumed_as_if_it_were_pending(
    provisioning_stack,
) -> None:
    """Resume must not leave an elapsed one-time task misleadingly active."""

    provisioner, job_service, _, _, events, _ = provisioning_stack
    job = _create_past_one_time_job(job_service)
    provisioner.provision(job.job_id)
    provisioner.pause(job.job_id)
    events.clear()

    with pytest.raises(ScheduledJobProvisioningStateError, match="elapsed"):
        provisioner.resume(job.job_id)

    assert job_service.get_job(job.job_id).status == ScheduledJobStatus.PAUSED.value
    assert not any(event.startswith("store:install") for event in events)


def test_timer_start_failure_restores_files_and_marks_job_failed(
    provisioning_stack,
) -> None:
    """A start failure after activation must compensate to an inactive DB state."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    controller.fail_operation = "start"

    with pytest.raises(ScheduledJobProvisioningError, match="starting its timer"):
        provisioner.provision(job.job_id)

    current = job_service.get_job(job.job_id)
    assert current.status == ScheduledJobStatus.PROVISIONING_FAILED.value
    assert current.is_active is False
    assert current.activated_at is None
    assert "super-secret-token" not in (current.provisioning_error or "")
    assert "[REDACTED]" in (current.provisioning_error or "")
    assert set(unit_store.files) == {SERVICE_TEMPLATE_UNIT_NAME}
    assert timer_unit_name(job.job_id) not in controller.enabled
    assert "db:failed" in events


def test_start_failure_with_database_outage_stays_externally_disabled(
    provisioning_stack,
) -> None:
    """A failed compensation write must still leave the timer unable to fire."""

    provisioner, job_service, controller, unit_store, _, _ = provisioning_stack
    job = _create_job(job_service)
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.raise_on_failure_transition = True
    controller.fail_operation = "start"

    with pytest.raises(ScheduledJobPersistenceError, match="mark failed"):
        provisioner.provision(job.job_id)

    current = job_service.get_job(job.job_id)
    timer = timer_unit_name(job.job_id)
    assert current.status == ScheduledJobStatus.ACTIVE.value
    assert timer not in controller.enabled
    assert timer not in unit_store.files

    repository.raise_on_failure_transition = False
    controller.fail_operation = None
    repaired = provisioner.reconcile(job.job_id)
    assert repaired.job.status == ScheduledJobStatus.ACTIVE.value
    assert timer in controller.enabled
    assert timer in controller.active


def test_unit_verification_failure_never_activates_the_database(
    provisioning_stack,
) -> None:
    """Invalid installed units must be restored before a job is marked failed."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    controller.fail_operation = "verify"

    with pytest.raises(ScheduledJobProvisioningError, match="before activation"):
        provisioner.provision(job.job_id)

    current = job_service.get_job(job.job_id)
    assert current.status == ScheduledJobStatus.PROVISIONING_FAILED.value
    assert current.is_active is False
    assert set(unit_store.files) == {SERVICE_TEMPLATE_UNIT_NAME}
    assert "db:activate" not in events
    assert "systemd:start" not in events


def test_reconcile_calendar_failure_disables_an_existing_active_timer(
    provisioning_stack,
) -> None:
    """A pre-install failure must still fail closed around an old active timer."""

    provisioner, job_service, controller, _, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    plan_result = provisioner.plan(job.job_id)
    timer = timer_unit_name(job.job_id)
    controller.fail_operation = f"calendar:{plan_result.plan.on_calendar[0]}"
    events.clear()

    with pytest.raises(ScheduledJobProvisioningError, match="made inactive"):
        provisioner.reconcile(job.job_id)

    current = job_service.get_job(job.job_id)
    assert current.status == ScheduledJobStatus.PAUSED.value
    assert timer not in controller.enabled
    assert timer not in controller.active
    assert "systemd:stop" in events
    assert "systemd:disable" in events
    assert "systemd:reload" in events


def test_reconcile_raw_host_probe_error_pauses_and_disables_active_job(
    provisioning_stack,
) -> None:
    """Unexpected host inspection errors must still follow fail-closed cleanup."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    host_probe = _FakeHostProbe(events)
    host_probe.activation_error = OSError("NSS lookup failed")
    reconciliation_service = ScheduledJobProvisioningService(
        settings=SystemdProvisioningSettings(),
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy(
            systemd_mutations_enabled=True,
            production_executor_ready=True,
        ),
        repository=provisioner._repository,  # noqa: SLF001
        host_probe=host_probe,
        operation_lock=_FakeOperationLock(events),
        executor_capability_ready=True,
    )

    result = reconciliation_service.reconcile(job.job_id)

    assert result.job.status == ScheduledJobStatus.PAUSED.value
    assert result.warning is not None
    assert timer_unit_name(job.job_id) not in controller.enabled
    assert timer_unit_name(job.job_id) not in controller.active


def test_repeated_provision_repairs_an_active_timer_crash_window(
    provisioning_stack,
) -> None:
    """A rerun must converge active DB state instead of assuming systemd is healthy."""

    provisioner, job_service, controller, _, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    timer = timer_unit_name(job.job_id)
    controller.enabled.clear()
    controller.active.clear()
    events.clear()

    result = provisioner.provision(job.job_id)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert timer in controller.enabled
    assert timer in controller.active
    assert "systemd:enable" in events
    assert "systemd:start" in events


@pytest.mark.parametrize("stored_receipt", [None, "wrong-but-stored.timer"])
def test_reconcile_repairs_an_active_job_timer_receipt(
    provisioning_stack,
    stored_receipt: str | None,
) -> None:
    """Reconciliation must restore the canonical receipt on an active row."""

    provisioner, job_service, controller, _, events, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    canonical_timer = timer_unit_name(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(timer_unit_name=stored_receipt)
        )
        session.commit()
    events.clear()

    result = provisioner.reconcile(job.job_id)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert result.job.timer_unit_name == canonical_timer
    assert canonical_timer in controller.enabled
    assert canonical_timer in controller.active
    assert "db:repair-timer-receipt" in events
    first_external_event = next(
        index
        for index, event in enumerate(events)
        if event.startswith(("store:", "systemd:"))
    )
    assert events.index("db:repair-timer-receipt") < first_external_event

    events.clear()
    repeated = provisioner.provision(job.job_id)
    assert repeated.job.timer_unit_name == canonical_timer
    assert "db:repair-timer-receipt" not in events


def test_ambiguous_timer_receipt_repair_is_confirmed_by_readback(
    provisioning_stack,
) -> None:
    """A lost receipt-commit response must accept a confirmed canonical row."""

    provisioner, job_service, _, _, _, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(timer_unit_name=None)
        )
        session.commit()
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.raise_after_receipt_repair = True

    result = provisioner.reconcile(job.job_id)

    assert result.job.timer_unit_name == timer_unit_name(job.job_id)


def test_timer_receipt_cas_conflict_prevents_systemd_mutation(
    provisioning_stack,
) -> None:
    """Receipt conflicts must stop reconciliation before changing systemd."""

    provisioner, job_service, _, _, events, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(timer_unit_name=None)
        )
        session.commit()
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.reject_receipt_repair = True
    events.clear()

    with pytest.raises(ScheduledJobProvisioningConflictError, match="changed"):
        provisioner.reconcile(job.job_id)

    assert not any(event.startswith("systemd:") for event in events)
    assert not any(event.startswith("store:") for event in events)


def test_timer_receipt_database_failure_prevents_systemd_mutation(
    provisioning_stack,
) -> None:
    """An uncommitted repair failure must leave external state untouched."""

    provisioner, job_service, _, _, events, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(timer_unit_name=None)
        )
        session.commit()
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.raise_before_receipt_repair = True
    events.clear()

    with pytest.raises(ScheduledJobPersistenceError, match="repair"):
        provisioner.reconcile(job.job_id)

    assert not any(event.startswith("systemd:") for event in events)
    assert not any(event.startswith("store:") for event in events)


def test_ambiguous_activation_commit_is_confirmed_by_database_readback(
    provisioning_stack,
) -> None:
    """A lost commit response must not disable a job proven active on readback."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    job = _create_job(job_service)
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.raise_after_activation = True

    result = provisioner.provision(job.job_id)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert result.plan.timer_unit in controller.enabled
    assert result.plan.timer_unit in controller.active


def test_pause_fails_closed_before_attempting_timer_stop(
    provisioning_stack,
) -> None:
    """Pause must make the runner reject the job before external cleanup begins."""

    provisioner, job_service, controller, _, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    events.clear()

    result = provisioner.pause(job.job_id)

    assert result.job.status == ScheduledJobStatus.PAUSED.value
    assert result.job.is_active is False
    assert result.job.activated_at is None
    assert events.index("db:pause") < events.index("systemd:stop")
    assert timer_unit_name(job.job_id) not in controller.enabled
    assert timer_unit_name(job.job_id) not in controller.active


def test_pause_stays_database_inactive_when_systemd_control_is_unavailable(
    provisioning_stack,
) -> None:
    """A broken systemctl path must not prevent the DB-first safety transition."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    host_probe = _FakeHostProbe(events)
    host_probe.fail_control = True
    cleanup_service = ScheduledJobProvisioningService(
        settings=SystemdProvisioningSettings(),
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy(systemd_mutations_enabled=True),
        repository=provisioner._repository,  # noqa: SLF001
        host_probe=host_probe,
        operation_lock=_FakeOperationLock(events),
    )

    result = cleanup_service.pause(job.job_id)

    assert result.job.status == ScheduledJobStatus.PAUSED.value
    assert result.job.is_active is False
    assert result.warning is not None
    assert "systemctl is unavailable" in result.warning
    assert timer_unit_name(job.job_id) in controller.active


def test_pause_does_not_require_activation_runtime_dependencies(
    provisioning_stack,
) -> None:
    """Emergency cleanup should need systemd control but not the agent runtime."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    host_probe = _FakeHostProbe(events)
    host_probe.fail_activation = True
    cleanup_service = ScheduledJobProvisioningService(
        settings=SystemdProvisioningSettings(),
        controller=controller,
        unit_store=unit_store,
        policy=ScheduledJobProvisioningPolicy(systemd_mutations_enabled=True),
        repository=provisioner._repository,  # noqa: SLF001
        host_probe=host_probe,
        operation_lock=_FakeOperationLock(events),
    )

    result = cleanup_service.pause(job.job_id)

    assert result.warning is None
    assert result.job.status == ScheduledJobStatus.PAUSED.value
    assert timer_unit_name(job.job_id) not in controller.active


def test_pause_cleanup_failure_remains_inactive_and_records_redacted_warning(
    provisioning_stack,
) -> None:
    """A systemd cleanup error must not reopen a safely paused job."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    controller.fail_operation = "stop"

    result = provisioner.pause(job.job_id)

    assert result.warning is not None
    assert "super-secret-token" not in result.warning
    assert result.job.status == ScheduledJobStatus.PAUSED.value
    assert result.job.is_active is False
    assert "[REDACTED]" in (result.job.provisioning_error or "")
    assert result.plan.timer_unit not in controller.enabled


def test_pause_and_archive_cleanup_do_not_depend_on_valid_definition_json(
    provisioning_stack,
) -> None:
    """Operators must be able to make a corrupt stored job safely inactive."""

    provisioner, job_service, controller, unit_store, _, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(
                definition_json=["malformed"],
                target_device_id="obsolete-device",
            )
        )
        session.commit()

    paused = provisioner.pause(job.job_id)
    archived = provisioner.archive(job.job_id)

    assert paused.job.status == ScheduledJobStatus.PAUSED.value
    assert archived.job.status == ScheduledJobStatus.DELETED.value
    assert archived.plan.timer_unit not in controller.enabled
    assert archived.plan.timer_unit not in controller.active
    assert archived.plan.timer_unit not in unit_store.files


def test_resume_sets_a_new_activation_boundary(
    provisioning_stack,
) -> None:
    """Resume must reactivate a paused job through the guarded installation path."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    provisioner.pause(job.job_id)

    result = provisioner.resume(job.job_id)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert result.job.activated_at == _NOW
    assert result.plan.timer_unit in controller.enabled
    assert result.plan.timer_unit in controller.active


def test_resume_rejects_a_still_live_execution_lease_and_stays_paused(
    provisioning_stack,
) -> None:
    """A paused job must not rotate ownership while its old worker lease is live."""

    provisioner, job_service, controller, _, events, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    with factory() as session:
        active_row = session.get(ScheduledJob, job.job_id)
        assert active_row is not None
        initial_generation = active_row.activation_generation
    run_id = _insert_running_run(
        factory,
        job_id=job.job_id,
        lease_expires_at=datetime(2100, 1, 1, tzinfo=timezone.utc),
    )
    provisioner.pause(job.job_id)
    events.clear()

    with pytest.raises(ScheduledJobProvisioningStateError, match="live lease"):
        provisioner.resume(job.job_id)

    current = job_service.get_job(job.job_id)
    assert current.status == ScheduledJobStatus.PAUSED.value
    assert timer_unit_name(job.job_id) not in controller.enabled
    assert timer_unit_name(job.job_id) not in controller.active
    assert "db:settle-and-activate" in events
    assert "systemd:start" not in events
    with factory() as session:
        run = session.get(ScheduledJobRun, run_id)
        assert run is not None
        assert run.status == ScheduledJobRunStatus.RUNNING.value
        assert run.lease_token is not None
        current_row = session.get(ScheduledJob, job.job_id)
        assert current_row is not None
        assert current_row.activation_generation == initial_generation


def test_resume_atomically_cancels_an_expired_lease_before_reactivation(
    provisioning_stack,
) -> None:
    """Expired work must be fenced and cancelled in the resume transaction."""

    provisioner, job_service, controller, _, events, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    with factory() as session:
        active_row = session.get(ScheduledJob, job.job_id)
        assert active_row is not None
        initial_generation = active_row.activation_generation
    run_id = _insert_running_run(
        factory,
        job_id=job.job_id,
        lease_expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )
    provisioner.pause(job.job_id)
    events.clear()

    result = provisioner.resume(job.job_id)

    assert result.job.status == ScheduledJobStatus.ACTIVE.value
    assert result.plan.timer_unit in controller.enabled
    assert result.plan.timer_unit in controller.active
    assert events.index("systemd:enable") < events.index("db:settle-and-activate")
    assert events.index("db:settle-and-activate") < events.index("systemd:start")
    with factory() as session:
        run = session.execute(
            select(ScheduledJobRun).where(ScheduledJobRun.run_id == run_id)
        ).scalar_one()
        assert run.status == ScheduledJobRunStatus.CANCELLED.value
        assert run.completed_at is not None
        assert run.lease_token is None
        assert run.lease_expires_at is None
        assert run.attempt_deadline_at is None
        assert run.error_message == "Expired execution cancelled before job resume."
        current_row = session.get(ScheduledJob, job.job_id)
        assert current_row is not None
        assert current_row.activation_generation == initial_generation + 1


def test_resume_start_failure_returns_to_paused_with_a_warning(
    provisioning_stack,
) -> None:
    """A failed resume must compensate to paused rather than leave an active row."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    provisioner.pause(job.job_id)
    controller.fail_operation = "start"

    with pytest.raises(ScheduledJobProvisioningError, match="starting its timer"):
        provisioner.resume(job.job_id)

    current = job_service.get_job(job.job_id)
    assert current.status == ScheduledJobStatus.PAUSED.value
    assert current.is_active is False
    assert current.activated_at is None
    assert "[REDACTED]" in (current.provisioning_error or "")


def test_archive_soft_deletes_before_removing_only_the_job_timer(
    provisioning_stack,
) -> None:
    """Archive must retain history/shared service while removing the exact timer."""

    provisioner, job_service, _, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    events.clear()

    result = provisioner.archive(job.job_id)

    assert result.job.status == ScheduledJobStatus.DELETED.value
    assert result.job.is_active is False
    assert events.index("db:delete") < events.index("systemd:stop")
    assert "systemd:clean" in events
    assert result.plan.timer_unit not in unit_store.files
    assert SERVICE_TEMPLATE_UNIT_NAME in unit_store.files


def test_archive_cleanup_failure_stays_deleted_and_is_reconcilable(
    provisioning_stack,
) -> None:
    """External archive failure must retain the terminal inactive DB state."""

    provisioner, job_service, controller, unit_store, _, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    controller.fail_operation = "clean"

    result = provisioner.archive(job.job_id)

    assert result.job.status == ScheduledJobStatus.DELETED.value
    assert result.job.is_active is False
    assert result.warning is not None
    assert result.plan.timer_unit in unit_store.files
    assert result.job.provisioning_error is not None


def test_archive_retry_reloads_systemd_after_a_prior_reload_failure(
    provisioning_stack,
) -> None:
    """Retry must reload even when the timer file was removed previously."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    timer = timer_unit_name(job.job_id)
    controller.fail_once_operation = "reload"
    events.clear()

    first = provisioner.archive(job.job_id)
    assert first.warning is not None
    assert timer not in unit_store.files

    events.clear()
    retried = provisioner.archive(job.job_id)

    assert retried.job.status == ScheduledJobStatus.DELETED.value
    assert retried.warning is None
    assert retried.job.provisioning_error is None
    assert "systemd:reload" in events


def test_dry_run_rejects_an_action_invalid_for_current_state(
    provisioning_stack,
) -> None:
    """A dry-run must model the same lifecycle rules as the real operation."""

    provisioner, job_service, _, _, events, _ = provisioning_stack
    job = _create_job(job_service)
    events.clear()

    with pytest.raises(ScheduledJobProvisioningStateError, match="Cannot resume"):
        provisioner.resume(job.job_id, dry_run=True)

    assert events == []
    assert job_service.get_job(job.job_id).status == "pending_provisioning"


def test_plan_ignores_untrusted_stored_timer_receipt(
    provisioning_stack,
) -> None:
    """Filesystem names must be recomputed from UUID rather than trusted from DB."""

    provisioner, job_service, _, _, _, factory = provisioning_stack
    job = _create_job(job_service)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(timer_unit_name="../../operator-input.timer")
        )
        session.commit()

    result = provisioner.plan(job.job_id)

    assert result.plan.timer_unit == timer_unit_name(job.job_id)
    assert "operator-input" not in result.plan.timer_unit


def test_plan_rejects_a_job_for_a_different_device(
    provisioning_stack,
) -> None:
    """A provisioning host must not install a task targeting another device."""

    provisioner, job_service, _, _, _, factory = provisioning_stack
    job = _create_job(job_service)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(target_device_id="another-device")
        )
        session.commit()

    with pytest.raises(ScheduledJobValidationError, match="does not target"):
        provisioner.plan(job.job_id)


def test_reconcile_disables_a_stray_timer_for_a_pending_job(
    provisioning_stack,
) -> None:
    """Explicit reconciliation must close the enable-before-DB crash window."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    job = _create_job(job_service)
    timer = timer_unit_name(job.job_id)
    controller.enabled.add(timer)
    controller.active.add(timer)

    result = provisioner.reconcile(job.job_id)

    assert result.job.status == ScheduledJobStatus.PENDING_PROVISIONING.value
    assert timer not in controller.enabled
    assert timer not in controller.active


@pytest.mark.parametrize(
    "terminal_status",
    (
        ScheduledJobStatus.COMPLETED.value,
        ScheduledJobStatus.EXECUTION_FAILED.value,
    ),
)
def test_reconcile_quiesces_terminal_execution_timers_without_reactivation(
    provisioning_stack,
    terminal_status: str,
) -> None:
    """Terminal one-time outcomes must keep DB history and remove timer files."""

    provisioner, job_service, controller, unit_store, events, factory = (
        provisioning_stack
    )
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    timer = timer_unit_name(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(
                status=terminal_status,
                is_active=False,
                activated_at=None,
            )
        )
        session.commit()
    events.clear()

    result = provisioner.reconcile(job.job_id)

    assert result.job.status == terminal_status
    assert result.job.is_active is False
    assert result.job.activated_at is None
    assert timer not in controller.enabled
    assert timer not in controller.active
    assert timer not in unit_store.files
    assert "systemd:enable" not in events
    assert "systemd:start" not in events


def test_reconcile_pauses_an_active_job_with_corrupt_definition_json(
    provisioning_stack,
) -> None:
    """Invalid active definitions must fail closed using only the UUID timer name."""

    provisioner, job_service, controller, _, _, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    timer = timer_unit_name(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(definition_json=["malformed"])
        )
        session.commit()

    result = provisioner.reconcile(job.job_id)

    assert result.job.status == ScheduledJobStatus.PAUSED.value
    assert result.job.is_active is False
    assert result.warning is not None
    assert timer not in controller.enabled
    assert timer not in controller.active


def test_unsafe_pause_confirms_an_ambiguous_committed_database_transition(
    provisioning_stack,
) -> None:
    """Lost pause responses must still be read back and externally quiesced."""

    provisioner, job_service, controller, _, _, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    timer = timer_unit_name(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(definition_json=["malformed"])
        )
        session.commit()
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.raise_after_pause = True

    result = provisioner.reconcile(job.job_id)

    assert result.job.status == ScheduledJobStatus.PAUSED.value
    assert timer not in controller.enabled
    assert timer not in controller.active


def test_unsafe_pause_quiesces_timer_when_database_transition_did_not_commit(
    provisioning_stack,
) -> None:
    """A rejected safety transition must still stop the invalid active timer."""

    provisioner, job_service, controller, _, _, factory = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    timer = timer_unit_name(job.job_id)
    with factory() as session:
        session.execute(
            update(ScheduledJob)
            .where(ScheduledJob.job_id == job.job_id)
            .values(definition_json=["malformed"])
        )
        session.commit()
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.raise_before_pause = True

    with pytest.raises(ScheduledJobPersistenceError, match="pause unsafe"):
        provisioner.reconcile(job.job_id)

    assert job_service.get_job(job.job_id).status == ScheduledJobStatus.ACTIVE.value
    assert timer not in controller.enabled
    assert timer not in controller.active


def test_reconcile_all_scans_keyset_pages_without_a_polling_daemon(
    provisioning_stack,
) -> None:
    """One explicit batch should converge every target job exactly once."""

    provisioner, job_service, _, _, events, _ = provisioning_stack
    active_job = _create_job(job_service)
    pending_job = _create_job(job_service)
    provisioner.provision(active_job.job_id)
    events.clear()

    result = provisioner.reconcile_all(page_size=1)

    assert result.clean is True
    assert result.scanned == 2
    assert result.succeeded == 2
    assert result.warnings == 0
    assert result.failed == 0
    assert result.issues == ()
    entered_jobs = {
        event.removeprefix("lock:enter:")
        for event in events
        if event.startswith("lock:enter:")
    }
    assert entered_jobs == {active_job.job_id, pending_job.job_id}


def test_reconcile_all_continues_after_one_job_failure(
    provisioning_stack,
) -> None:
    """A broken job must not prevent later jobs from being reconciled."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    active_job = _create_job(job_service)
    _create_job(job_service)
    provisioner.provision(active_job.job_id)
    controller.fail_operation = "start"

    result = provisioner.reconcile_all(page_size=1)

    assert result.scanned == 2
    assert result.failed == 1
    assert result.succeeded == 1
    assert len(result.issues) == 1
    assert result.issues[0].job_id == active_job.job_id
    assert result.issues[0].outcome == "failed"


def test_reconcile_all_warning_is_reported_as_partial_convergence(
    provisioning_stack,
) -> None:
    """Cleanup warnings must make the aggregate result non-clean."""

    provisioner, job_service, controller, _, _, _ = provisioning_stack
    job = _create_job(job_service)
    timer = timer_unit_name(job.job_id)
    controller.enabled.add(timer)
    controller.active.add(timer)
    controller.fail_operation = "stop"

    result = provisioner.reconcile_all()

    assert result.clean is False
    assert result.scanned == 1
    assert result.warnings == 1
    assert result.failed == 0
    assert result.issues[0].outcome == "warning"
    assert result.issues[0].status == ScheduledJobStatus.PENDING_PROVISIONING.value


def test_reconcile_all_pauses_active_job_when_shared_service_is_missing(
    provisioning_stack,
) -> None:
    """A stale global template must fail closed instead of blocking cleanup."""

    provisioner, job_service, controller, unit_store, _, _ = provisioning_stack
    job = _create_job(job_service)
    provisioner.provision(job.job_id)
    timer = timer_unit_name(job.job_id)
    unit_store.files.pop(SERVICE_TEMPLATE_UNIT_NAME)

    result = provisioner.reconcile_all()

    assert result.failed == 1
    assert job_service.get_job(job.job_id).status == ScheduledJobStatus.PAUSED.value
    assert timer not in controller.enabled
    assert timer not in controller.active


def test_reconcile_all_dry_run_has_no_external_or_database_side_effects(
    provisioning_stack,
) -> None:
    """Batch preview must remain usable without privileged host access."""

    provisioner, job_service, controller, unit_store, events, _ = provisioning_stack
    job = _create_job(job_service)
    original_files = dict(unit_store.files)
    events.clear()

    result = provisioner.reconcile_all(dry_run=True, page_size=1)

    assert result.dry_run is True
    assert result.scanned == 1
    assert result.clean is True
    assert job_service.get_job(job.job_id).status == "pending_provisioning"
    assert unit_store.files == original_files
    assert controller.enabled == set()
    assert controller.active == set()
    assert events == []


def test_reconcile_all_aborts_cleanly_when_a_candidate_page_cannot_load(
    provisioning_stack,
) -> None:
    """A page-query failure must not produce a misleading partial summary."""

    provisioner, job_service, _, _, events, _ = provisioning_stack
    _create_job(job_service)
    repository = provisioner._repository  # noqa: SLF001
    assert isinstance(repository, _RecordingRepository)
    repository.fail_reconciliation_page = True
    events.clear()

    with pytest.raises(ScheduledJobPersistenceError, match="page"):
        provisioner.reconcile_all()

    assert not any(event.startswith("db:") for event in events)
    assert not any(event.startswith("systemd:") for event in events)
