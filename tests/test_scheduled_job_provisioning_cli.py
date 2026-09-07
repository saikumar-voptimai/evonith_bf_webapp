"""Tests for the narrow scheduled-job systemd management CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from data.scheduled_tasks import (
    ControlSyncUnitsResult,
    ScheduledJobExecutorNotReadyError,
    ScheduledJobProvisioningResult,
    ScheduledJobReconciliationBatchResult,
    ScheduledJobReconciliationIssue,
    ScheduledJobView,
    SharedServiceTemplateResult,
)
from scripts.furnacemind_systemd_provisioner import (
    build_provisioning_service_from_environment,
    main,
)
from utils.scheduled_tasks.scheduled_task_definition import (
    ScheduledTaskInput,
    build_task_definition,
)
from utils.scheduled_tasks.systemd_units import (
    SystemdProvisioningSettings,
    build_control_sync_artifacts,
    build_provisioning_plan,
    build_service_template_artifact,
)

_JOB_ID = "11111111-1111-4111-8111-111111111111"
_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


def _definition() -> dict[str, object]:
    """Build a safe scheduled-job definition for CLI result fixtures."""

    return build_task_definition(
        ScheduledTaskInput(
            name="Hourly furnace review",
            instructions="Review recent BF2 operation.",
            furnace="BF2",
            data_period="last_24_hours",
            output_format="operator_summary",
            schedule_kind="hourly",
            delivery_channel="in_app",
            job_type="furnace_summary",
            hourly_minute=10,
        ),
        generated_at=_NOW,
    )


def _result(action: str, *, dry_run: bool) -> ScheduledJobProvisioningResult:
    """Build a complete CLI result without requiring a database."""

    definition = _definition()
    job = ScheduledJobView(
        job_id=_JOB_ID,
        job_name="Hourly furnace review",
        schema_version="scheduled-job/v1",
        definition=definition,
        status="pending_provisioning",
        is_active=False,
        created_by_user_id="00000000-0000-0000-0000-000000000001",
        created_by_username="operator.test",
        target_device_id="bf2-jetson-01",
        timer_unit_name=None,
        provisioning_error=None,
        activated_at=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    plan = build_provisioning_plan(
        _JOB_ID,
        definition,
        SystemdProvisioningSettings(),
    )
    return ScheduledJobProvisioningResult(
        action=action,
        job=job,
        plan=plan,
        changed=False,
        dry_run=dry_run,
    )


def _shared_result(*, dry_run: bool) -> SharedServiceTemplateResult:
    """Build one deterministic shared-service CLI result."""

    return SharedServiceTemplateResult(
        action="install-service",
        artifact=build_service_template_artifact(SystemdProvisioningSettings()),
        changed=not dry_run,
        dry_run=dry_run,
    )


def _control_sync_result(*, dry_run: bool) -> ControlSyncUnitsResult:
    """Build one deterministic control-sync installation result."""

    return ControlSyncUnitsResult(
        action="install-control-sync",
        artifacts=build_control_sync_artifacts(SystemdProvisioningSettings()),
        changed=not dry_run,
        dry_run=dry_run,
    )


def _batch_result(
    *,
    dry_run: bool,
    warning: bool = False,
) -> ScheduledJobReconciliationBatchResult:
    """Build one deterministic all-job reconciliation result."""

    issues = (
        (
            ScheduledJobReconciliationIssue(
                job_id=_JOB_ID,
                outcome="warning",
                status="paused",
                changed=True,
                message="timer cleanup needs attention",
            ),
        )
        if warning
        else ()
    )
    return ScheduledJobReconciliationBatchResult(
        action="reconcile-all",
        target_device_id="bf2-jetson-01",
        scanned=1,
        succeeded=0 if warning else 1,
        warnings=1 if warning else 0,
        failed=0,
        omitted_issues=0,
        dry_run=dry_run,
        started_at=_NOW,
        completed_at=_NOW,
        issues=issues,
    )


class _FakeProvisioningService:
    """Return deterministic results while recording CLI dispatch."""

    def __init__(self, *, reject_activation: bool = False) -> None:
        """Create a fake that can emulate the executor-readiness gate."""

        self.reject_activation = reject_activation
        self.batch_warning = False
        self.calls: list[tuple[str, str, bool]] = []
        self.disposed = False

    def plan(self, job_id: str) -> ScheduledJobProvisioningResult:
        """Record and return a read-only plan."""

        self.calls.append(("plan", job_id, True))
        return _result("plan", dry_run=True)

    def install_service_template(
        self,
        *,
        dry_run: bool = False,
    ) -> SharedServiceTemplateResult:
        """Record and return shared service-template installation."""

        self.calls.append(("install-service", "", dry_run))
        return _shared_result(dry_run=dry_run)

    def install_control_sync_units(
        self,
        *,
        dry_run: bool = False,
    ) -> ControlSyncUnitsResult:
        """Record and return command-bridge unit installation."""

        self.calls.append(("install-control-sync", "", dry_run))
        return _control_sync_result(dry_run=dry_run)

    def provision(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Record provisioning or emulate a blocked production executor."""

        self.calls.append(("provision", job_id, dry_run))
        if self.reject_activation and not dry_run:
            raise ScheduledJobExecutorNotReadyError("executor is not ready")
        return _result("provision", dry_run=dry_run)

    def pause(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Record and return a pause result."""

        self.calls.append(("pause", job_id, dry_run))
        return _result("pause", dry_run=dry_run)

    def resume(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Record and return a resume result."""

        self.calls.append(("resume", job_id, dry_run))
        return _result("resume", dry_run=dry_run)

    def archive(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Record and return an archive result."""

        self.calls.append(("archive", job_id, dry_run))
        return _result("archive", dry_run=dry_run)

    def reconcile(
        self,
        job_id: str,
        *,
        dry_run: bool = False,
    ) -> ScheduledJobProvisioningResult:
        """Record and return a reconciliation result."""

        self.calls.append(("reconcile", job_id, dry_run))
        return _result("reconcile", dry_run=dry_run)

    def reconcile_all(
        self,
        *,
        dry_run: bool = False,
        page_size: int = 100,
    ) -> ScheduledJobReconciliationBatchResult:
        """Record and return an all-job reconciliation summary."""

        self.calls.append(("reconcile-all", str(page_size), dry_run))
        return _batch_result(dry_run=dry_run, warning=self.batch_warning)

    def dispose(self) -> None:
        """Record disposal when the CLI owns this fake."""

        self.disposed = True


def test_plan_prints_exact_safe_unit_artifacts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Plan output should expose render details without performing mutations."""

    service = _FakeProvisioningService()

    exit_code = main(["plan", "--job-id", _JOB_ID], service=service)  # type: ignore[arg-type]

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["action"] == "plan"
    assert document["dry_run"] is True
    assert document["timer_unit"].endswith(f"{_JOB_ID}.timer")
    assert len(document["unit_files"]) == 2
    assert service.calls == [("plan", _JOB_ID, True)]


def test_install_service_is_a_separate_jobless_deployment_action(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should install the shared service without accepting a job id."""

    service = _FakeProvisioningService()

    exit_code = main(
        ["install-service", "--dry-run"],
        service=service,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["action"] == "install-service"
    assert document["unit_name"] == "furnacemind-job@.service"
    assert "ExecStart=" in document["content"]
    assert service.calls == [("install-service", "", True)]


def test_install_control_sync_renders_both_jobless_units(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should install the Jetson queue bridge without a job id."""

    service = _FakeProvisioningService()

    exit_code = main(
        ["install-control-sync", "--dry-run"],
        service=service,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["action"] == "install-control-sync"
    assert [item["unit_name"] for item in document["units"]] == [
        "furnacemind-control-sync.service",
        "furnacemind-control-sync.timer",
    ]
    assert all("content" in item for item in document["units"])
    assert service.calls == [("install-control-sync", "", True)]


def test_reconcile_all_uses_no_job_id_and_returns_partial_exit_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A batch warning should be machine-readable and return operational failure."""

    service = _FakeProvisioningService()
    service.batch_warning = True

    exit_code = main(
        ["reconcile-all", "--page-size", "25"],
        service=service,  # type: ignore[arg-type]
    )

    assert exit_code == 1
    document = json.loads(capsys.readouterr().out)
    assert document["action"] == "reconcile-all"
    assert document["clean"] is False
    assert document["warnings"] == 1
    assert document["issues"][0]["job_id"] == _JOB_ID
    assert service.calls == [("reconcile-all", "25", False)]


def test_dry_run_flag_is_forwarded_without_implicit_provisioning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every mutating verb should retain its explicit dry-run boundary."""

    service = _FakeProvisioningService()

    exit_code = main(
        ["archive", "--job-id", _JOB_ID, "--dry-run"],
        service=service,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert service.calls == [("archive", _JOB_ID, True)]


def test_cli_reports_executor_gate_as_a_rejected_request(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Production activation must fail clearly without deployment approval."""

    service = _FakeProvisioningService(reject_activation=True)

    exit_code = main(
        ["provision", "--job-id", _JOB_ID],
        service=service,  # type: ignore[arg-type]
    )

    assert exit_code == 2
    assert capsys.readouterr().out == ""
    assert "executor is not ready" in caplog.text


def test_stable_provisioner_wrapper_help_works_without_pythonpath(
    tmp_path: Path,
) -> None:
    """The deployment command should resolve project imports from any directory."""

    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "furnacemind_systemd_provisioner.py"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "furnacemind-systemd-provisioner" in completed.stdout
    assert "reconcile" in completed.stdout


def test_default_posix_paths_construct_without_touching_windows_filesystem() -> None:
    """Dry-run setup must defer Linux path checks until a real mutation."""

    service = build_provisioning_service_from_environment(
        {
            "DATABASE_URL": "postgresql+psycopg2://user:pass@localhost/furnacemind",
        }
    )

    service.dispose()


def test_build_service_exposes_code_capability_but_retains_environment_gate() -> None:
    """Ready code must not bypass the deployment-owned executor opt-in."""

    base_environment = {
        "DATABASE_URL": "postgresql+psycopg2://user:pass@localhost/furnacemind",
    }
    blocked = build_provisioning_service_from_environment(base_environment)
    enabled = build_provisioning_service_from_environment(
        {
            **base_environment,
            "SCHEDULED_TASKS_EXECUTOR_READY": "true",
        }
    )
    try:
        assert blocked._executor_capability_ready is True  # noqa: SLF001
        assert blocked._policy.production_executor_ready is False  # noqa: SLF001
        assert blocked._executor_ready is False  # noqa: SLF001
        assert enabled._executor_capability_ready is True  # noqa: SLF001
        assert enabled._policy.production_executor_ready is True  # noqa: SLF001
        assert enabled._executor_ready is True  # noqa: SLF001
    finally:
        blocked.dispose()
        enabled.dispose()


def test_install_service_dry_run_does_not_require_database_configuration() -> None:
    """The deployment-owned shared template must remain database-independent."""

    service = build_provisioning_service_from_environment({})
    try:
        result = service.install_service_template(dry_run=True)
    finally:
        service.dispose()

    assert result.dry_run is True
    assert result.artifact.unit_name == "furnacemind-job@.service"
