"""Tests for safe scheduled-task systemd commands and atomic unit storage.

The command tests use a recording subprocess adapter, so this suite never
contacts the host systemd manager.  Filesystem tests operate only inside each
pytest temporary directory.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

import utils.scheduled_tasks.systemd_controller as controller_module
from utils.scheduled_tasks.systemd_controller import (
    MAX_COMMAND_ERROR_CHARS,
    SystemdCommandError,
    SystemdController,
    SystemdSafetyError,
    UnitFileSafetyError,
    UnitFileStore,
    validate_unit_root,
)

MANAGED_MARKER = "# Managed by FurnaceMind scheduled tasks. Do not edit."


class RecordingCommandRunner:
    """Record subprocess arguments and return a configurable result."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        exception: BaseException | None = None,
    ) -> None:
        """Configure the result or exception produced for every invocation."""

        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.exception = exception
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(
        self,
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Record one call and return its configured completed process."""

        self.calls.append((list(args), dict(kwargs)))
        if self.exception is not None:
            raise self.exception
        return subprocess.CompletedProcess(
            args=args,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _controller(runner: RecordingCommandRunner) -> SystemdController:
    """Build a controller that uses inert absolute executable paths."""

    return SystemdController(
        systemctl_path="/usr/bin/systemctl",
        systemd_analyze_path="/usr/bin/systemd-analyze",
        command_runner=runner,
    )


def _managed_content(label: str) -> str:
    """Return a minimal marked timer file for storage tests."""

    return f"{MANAGED_MARKER}\n[Unit]\nDescription={label}\n"


def _file_status(*, mode: int, owner_uid: int) -> os.stat_result:
    """Build deterministic POSIX-like metadata for ownership policy tests."""

    return os.stat_result((mode, 0, 0, 1, owner_uid, 0, 0, 0, 0, 0))


def test_controller_uses_argv_and_a_minimal_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commands remain discrete argv values and do not inherit host secrets."""

    monkeypatch.setenv("SHOULD_NOT_LEAK", "private-value")
    runner = RecordingCommandRunner(stdout="normalized calendar")
    controller = _controller(runner)
    expression = "Mon *-*-* 12:00:00; touch /tmp/not-run"

    assert controller.validate_calendar(expression) == "normalized calendar"
    controller.enable("furnacemind-job-123.timer")

    calendar_args, calendar_options = runner.calls[0]
    assert calendar_args == [
        "/usr/bin/systemd-analyze",
        "calendar",
        expression,
    ]
    assert runner.calls[1][0] == [
        "/usr/bin/systemctl",
        "enable",
        "furnacemind-job-123.timer",
    ]
    assert calendar_options["shell"] is False
    assert calendar_options["check"] is False
    assert calendar_options["timeout"] == 15.0
    assert "SHOULD_NOT_LEAK" not in calendar_options["env"]
    assert set(calendar_options["env"]) == {
        "LANG",
        "LC_ALL",
        "PATH",
        "SYSTEMD_COLORS",
        "SYSTEMD_PAGER",
    }


def test_controller_rejects_option_and_path_injection() -> None:
    """Untrusted executable and unit-name shapes fail before subprocess use."""

    runner = RecordingCommandRunner()
    controller = _controller(runner)

    with pytest.raises(SystemdSafetyError):
        controller.enable("--now")
    with pytest.raises(SystemdSafetyError):
        controller.start("../other.timer")
    with pytest.raises(SystemdSafetyError):
        SystemdController(
            systemctl_path="systemctl",
            systemd_analyze_path="/usr/bin/systemd-analyze",
            command_runner=runner,
        )

    assert runner.calls == []


def test_verify_passes_absolute_files_as_separate_arguments(tmp_path: Path) -> None:
    """Unit verification preserves an absolute path containing spaces as one arg."""

    unit_dir = tmp_path / "rendered units"
    unit_dir.mkdir()
    service_path = unit_dir / "furnacemind-job@.service"
    timer_path = unit_dir / "furnacemind-job-123.timer"
    service_path.write_text("[Service]\nType=oneshot\n", encoding="utf-8")
    timer_path.write_text("[Timer]\nOnCalendar=daily\n", encoding="utf-8")
    runner = RecordingCommandRunner()
    controller = _controller(runner)

    controller.verify([service_path, timer_path])

    assert runner.calls[0][0] == [
        "/usr/bin/systemd-analyze",
        "verify",
        str(service_path),
        str(timer_path),
    ]


def test_failed_command_redacts_and_caps_diagnostics() -> None:
    """Persistable command failures omit common credential forms and long output."""

    runner = RecordingCommandRunner(
        returncode=1,
        stderr=(
            "password=hunter2 Authorization Bearer abcdefghijk "
            "postgresql://admin:database-pass@db.local/jobs "
            "api_key=sk-abcdefghijklmnopqrstuvwxyz " + ("x" * 5_000)
        ),
    )
    controller = _controller(runner)

    with pytest.raises(SystemdCommandError) as error_info:
        controller.start("furnacemind-job-123.timer")

    message = str(error_info.value)
    assert error_info.value.returncode == 1
    assert len(message) <= MAX_COMMAND_ERROR_CHARS
    assert "hunter2" not in message
    assert "abcdefghijk" not in message
    assert "database-pass" not in message
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in message
    assert "[REDACTED]" in message


def test_timeout_becomes_a_redacted_command_error() -> None:
    """A bounded subprocess timeout is translated without leaking its stderr."""

    timeout = subprocess.TimeoutExpired(
        cmd=["systemctl"],
        timeout=15,
        stderr="token=timeout-secret",
    )
    controller = _controller(RecordingCommandRunner(exception=timeout))

    with pytest.raises(SystemdCommandError) as error_info:
        controller.daemon_reload()

    assert error_info.value.returncode is None
    assert "timed out after 15 seconds" in str(error_info.value)
    assert "timeout-secret" not in str(error_info.value)


@pytest.mark.parametrize(
    ("method_name", "subcommand"),
    [
        ("daemon_reload", ["daemon-reload"]),
        ("enable", ["enable", "furnacemind-job-123.timer"]),
        ("start", ["start", "furnacemind-job-123.timer"]),
        ("stop", ["stop", "furnacemind-job-123.timer"]),
        ("disable", ["disable", "furnacemind-job-123.timer"]),
        ("clean_state", ["clean", "--what=state", "furnacemind-job-123.timer"]),
    ],
)
def test_controller_exposes_only_fixed_systemctl_operations(
    method_name: str,
    subcommand: list[str],
) -> None:
    """Each mutating controller method maps to one predictable systemctl argv."""

    runner = RecordingCommandRunner()
    controller = _controller(runner)
    method = getattr(controller, method_name)

    if method_name == "daemon_reload":
        method()
    else:
        method("furnacemind-job-123.timer")

    assert runner.calls[0][0] == ["/usr/bin/systemctl", *subcommand]


@pytest.mark.parametrize(("returncode", "expected"), [(0, True), (3, False)])
def test_controller_state_queries_return_boolean(
    returncode: int,
    expected: bool,
) -> None:
    """Expected systemctl query statuses become booleans rather than failures."""

    runner = RecordingCommandRunner(returncode=returncode)
    controller = _controller(runner)

    assert controller.is_enabled("furnacemind-job-123.timer") is expected
    assert controller.is_active("furnacemind-job-123.timer") is expected


def test_install_and_restore_preserve_prior_managed_content(tmp_path: Path) -> None:
    """An atomic installation returns enough prior state for exact rollback."""

    store = UnitFileStore(tmp_path)
    unit_name = "furnacemind-job-123.timer"
    original = _managed_content("original")
    replacement = _managed_content("replacement")
    unit_path = tmp_path / unit_name
    unit_path.write_text(original, encoding="utf-8")
    original_mode = stat.S_IMODE(unit_path.stat().st_mode)

    snapshot = store.install(unit_name, replacement)
    assert snapshot.existed
    assert unit_path.read_text(encoding="utf-8") == replacement
    if os.name == "posix":
        assert stat.S_IMODE(unit_path.stat().st_mode) == 0o644

    store.restore(snapshot)
    assert unit_path.read_text(encoding="utf-8") == original
    assert stat.S_IMODE(unit_path.stat().st_mode) == original_mode


def test_restore_of_absent_snapshot_removes_new_managed_unit(tmp_path: Path) -> None:
    """Rollback removes a unit that did not exist before installation."""

    store = UnitFileStore(tmp_path)
    unit_name = "furnacemind-job-123.timer"

    snapshot = store.install(unit_name, _managed_content("new"))
    assert snapshot.existed is False
    store.restore(snapshot)

    assert not (tmp_path / unit_name).exists()


def test_failed_atomic_replace_preserves_original_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement failure leaves the original file and no temporary artifact."""

    store = UnitFileStore(tmp_path)
    unit_name = "furnacemind-job-123.timer"
    unit_path = tmp_path / unit_name
    original = _managed_content("original")
    unit_path.write_text(original, encoding="utf-8")

    def _fail_replace(source: Path, destination: Path) -> None:
        """Assert same-directory replacement and emulate an OS failure."""

        assert Path(source).parent == tmp_path
        assert Path(destination).parent == tmp_path
        raise OSError("simulated replace failure")

    monkeypatch.setattr(controller_module.os, "replace", _fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.install(unit_name, _managed_content("replacement"))

    assert unit_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(f".{unit_name}.*.tmp")) == []


def test_post_replace_fsync_failure_restores_prior_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durability error after replacement must still restore the old unit."""

    store = UnitFileStore(tmp_path)
    unit_name = "furnacemind-job-123.timer"
    unit_path = tmp_path / unit_name
    original = _managed_content("original")
    unit_path.write_text(original, encoding="utf-8")
    fsync_calls = 0

    def _fail_first_directory_fsync(directory: Path) -> None:
        """Fail after the first replace and allow the rollback fsync to finish."""

        nonlocal fsync_calls
        assert directory == tmp_path
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        controller_module,
        "_fsync_directory",
        _fail_first_directory_fsync,
    )

    with pytest.raises(OSError, match="directory fsync failure"):
        store.install(unit_name, _managed_content("replacement"))

    assert unit_path.read_text(encoding="utf-8") == original
    assert fsync_calls == 2
    assert list(tmp_path.glob(f".{unit_name}.*.tmp")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX unit modes are required")
def test_post_replace_fsync_failure_restores_prior_mode_for_same_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback must restore mode even when replacement bytes were unchanged."""

    store = UnitFileStore(tmp_path)
    unit_name = "furnacemind-job-123.timer"
    unit_path = tmp_path / unit_name
    content = _managed_content("unchanged")
    unit_path.write_text(content, encoding="utf-8")
    unit_path.chmod(0o600)
    fsync_calls = 0

    def _fail_first_directory_fsync(directory: Path) -> None:
        """Fail the installation fsync but allow rollback durability checks."""

        nonlocal fsync_calls
        assert directory == tmp_path
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        controller_module,
        "_fsync_directory",
        _fail_first_directory_fsync,
    )

    with pytest.raises(OSError, match="directory fsync failure"):
        store.install(unit_name, content)

    assert unit_path.read_text(encoding="utf-8") == content
    assert stat.S_IMODE(unit_path.stat().st_mode) == 0o600
    assert fsync_calls == 2


def test_remove_is_idempotent_but_refuses_unmanaged_content(tmp_path: Path) -> None:
    """Removal ignores absence while preserving a colliding administrator file."""

    store = UnitFileStore(tmp_path)
    unit_name = "furnacemind-job-123.timer"
    unit_path = tmp_path / unit_name

    assert store.remove(unit_name).existed is False
    unit_path.write_text("[Timer]\nOnCalendar=daily\n", encoding="utf-8")

    with pytest.raises(UnitFileSafetyError, match="managed marker"):
        store.remove(unit_name)
    assert unit_path.exists()


def test_install_refuses_unmanaged_content_and_unsafe_basename(tmp_path: Path) -> None:
    """Only marked content and exact unit basenames may enter the unit root."""

    store = UnitFileStore(tmp_path)

    with pytest.raises(UnitFileSafetyError, match="managed marker"):
        store.install("furnacemind-job-123.timer", "[Timer]\nOnCalendar=daily\n")
    with pytest.raises(SystemdSafetyError, match="basename"):
        store.install("../furnacemind-job-123.timer", _managed_content("unsafe"))
    assert list(tmp_path.iterdir()) == []


def test_unit_root_rejects_relative_non_directory_and_world_writable_paths(
    tmp_path: Path,
) -> None:
    """Unsafe storage roots fail before any unit content can be touched."""

    with pytest.raises(UnitFileSafetyError, match="absolute"):
        validate_unit_root("relative/systemd")

    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("data", encoding="utf-8")
    with pytest.raises(UnitFileSafetyError, match="directory"):
        validate_unit_root(regular_file)

    if os.name == "posix":
        os.chmod(tmp_path, 0o777)
    try:
        with pytest.raises(UnitFileSafetyError, match="world-writable"):
            validate_unit_root(tmp_path, enforce_posix_permissions=True)
    finally:
        if os.name == "posix":
            os.chmod(tmp_path, 0o700)


def test_root_trust_policy_rejects_non_root_ownership() -> None:
    """Production trust requires ownership by the root account."""

    status = _file_status(mode=stat.S_IFDIR | 0o755, owner_uid=1_000)

    with pytest.raises(UnitFileSafetyError, match="root-owned"):
        controller_module._validate_root_trust_status(
            status,
            subject="Systemd unit root",
        )


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_root_trust_policy_rejects_group_or_world_write(mode: int) -> None:
    """Production trust rejects every group/world-writable mode combination."""

    status = _file_status(mode=stat.S_IFDIR | mode, owner_uid=0)

    with pytest.raises(UnitFileSafetyError, match="group or by others"):
        controller_module._validate_root_trust_status(
            status,
            subject="Systemd unit root",
        )


def test_strict_store_fails_closed_without_posix_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict production trust cannot silently degrade on an unsupported host."""

    monkeypatch.setattr(controller_module, "_supports_root_trust", lambda: False)

    with pytest.raises(UnitFileSafetyError, match="POSIX"):
        UnitFileStore(tmp_path, require_root_trust=True)

    lazy_store = UnitFileStore(
        tmp_path,
        validate_on_init=False,
        require_root_trust=True,
    )
    with pytest.raises(UnitFileSafetyError, match="POSIX"):
        lazy_store.snapshot("furnacemind-job-123.timer")


def test_strict_store_checks_existing_unit_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every existing managed target crosses the strict ownership check."""

    checked_subjects: list[str] = []

    def _record_trust_check(
        _file_status: os.stat_result,
        *,
        subject: str,
    ) -> None:
        """Record checks and reject the existing unit as non-root-owned."""

        checked_subjects.append(subject)
        if subject == "Existing managed unit file":
            raise UnitFileSafetyError("Existing managed unit file must be root-owned.")

    monkeypatch.setattr(controller_module, "_supports_root_trust", lambda: True)
    monkeypatch.setattr(
        controller_module,
        "_validate_root_trust_status",
        _record_trust_check,
    )
    unit_name = "furnacemind-job-123.timer"
    (tmp_path / unit_name).write_text(_managed_content("existing"), encoding="utf-8")
    store = UnitFileStore(tmp_path, require_root_trust=True)

    with pytest.raises(UnitFileSafetyError, match="root-owned"):
        store.snapshot(unit_name)

    assert checked_subjects == [
        "Systemd unit root",
        "Systemd unit root",
        "Existing managed unit file",
    ]


def test_strict_store_checks_new_temporary_unit_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strict install verifies its new inode before replacing the target."""

    checked_subjects: list[str] = []

    def _record_trust_check(
        _file_status: os.stat_result,
        *,
        subject: str,
    ) -> None:
        """Record each strict metadata check without relying on host ownership."""

        checked_subjects.append(subject)

    monkeypatch.setattr(controller_module, "_supports_root_trust", lambda: True)
    monkeypatch.setattr(
        controller_module,
        "_validate_root_trust_status",
        _record_trust_check,
    )
    store = UnitFileStore(tmp_path, require_root_trust=True)

    store.install("furnacemind-job-123.timer", _managed_content("new"))

    assert checked_subjects == [
        "Systemd unit root",
        "Systemd unit root",
        "Temporary managed unit file",
    ]


def test_store_can_defer_root_validation_for_a_dry_run(tmp_path: Path) -> None:
    """Planning may construct a store before its Linux destination exists."""

    missing_root = tmp_path / "not-installed" / "systemd"
    store = UnitFileStore(missing_root, validate_on_init=False)

    with pytest.raises(UnitFileSafetyError, match="does not exist"):
        store.snapshot("furnacemind-job-123.timer")


def test_unit_root_and_target_reject_symlinks(tmp_path: Path) -> None:
    """Neither the managed directory nor an existing target may be a symlink."""

    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is not permitted on this platform.")

    with pytest.raises(UnitFileSafetyError, match="symlink"):
        UnitFileStore(linked_root)

    store = UnitFileStore(real_root)
    external = tmp_path / "external.timer"
    external.write_text(_managed_content("external"), encoding="utf-8")
    target = real_root / "furnacemind-job-123.timer"
    try:
        target.symlink_to(external)
    except OSError:
        pytest.skip("Creating file symlinks is not permitted on this platform.")

    with pytest.raises(UnitFileSafetyError, match="symlink"):
        store.remove(target.name)
    assert external.exists()
