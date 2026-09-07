"""Safe systemd command and unit-file side effects for scheduled tasks.

This module is the narrow privileged boundary used by scheduled-task
provisioning.  It executes only fixed systemd operations with argument vectors
and a controlled environment, and it installs only explicitly managed unit
files through atomic same-directory replacements.  The command runner and file
root are injected so the behavior can be tested without invoking systemd.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, Protocol

from utils.scheduled_tasks.systemd_units import MANAGED_UNIT_MARKER

MAX_COMMAND_TIMEOUT_SECONDS: Final = 60.0
MAX_COMMAND_ERROR_CHARS: Final = 1_200
MAX_UNIT_FILE_BYTES: Final = 256 * 1024

_UNIT_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,239}\.(?:service|timer)\Z"
)
_AUTHORIZATION_TOKEN_PATTERN = re.compile(
    r"(?i)\b(?P<scheme>bearer|basic)\s+[A-Za-z0-9._~+/=-]+"
)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?ix)"
    r"(?P<label>\b(?:[a-z][a-z0-9]*[\s_-])*"
    r"(?:api[\s_-]?key|password|passwd|token|secret|credentials?))"
    r"(?P<separator>\s*(?::|=|\bis\b)\s*)"
    r"(?P<value>\{[^}]*\}|\[[^]]*\]|\"(?:\\.|[^\"])*\"|"
    r"'(?:\\.|[^'])*'|[^\s,;]+)"
)
_URI_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?P<scheme>[a-z][a-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@"
)
_KNOWN_CREDENTIAL_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"eyJ[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]+){1,2})"
)
_CONTROLLED_ENVIRONMENT: Final[dict[str, str]] = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "SYSTEMD_COLORS": "0",
    "SYSTEMD_PAGER": "",
}


class CommandRunner(Protocol):
    """Callable compatible with the subset of ``subprocess.run`` we use."""

    def __call__(
        self,
        args: Sequence[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Execute one argument vector and return its completed process."""


class SystemdSafetyError(ValueError):
    """Raised when untrusted input would cross the systemd safety boundary."""


class UnitFileSafetyError(SystemdSafetyError):
    """Raised when a unit path or file is unsafe to modify."""


class SystemdCommandError(RuntimeError):
    """Raised when a fixed systemd command cannot be completed safely."""

    def __init__(
        self,
        operation: str,
        *,
        returncode: int | None,
        detail: str,
    ) -> None:
        """Create a bounded operational error without exposing command secrets."""

        self.operation = operation
        self.returncode = returncode
        status = (
            f"exit status {returncode}" if returncode is not None else "execution error"
        )
        message = f"{operation} failed ({status}): {detail}"
        super().__init__(_cap_text(message, MAX_COMMAND_ERROR_CHARS))


@dataclass(frozen=True, slots=True)
class UnitFileSnapshot:
    """Exact prior state of one unit file for compensating rollback."""

    unit_name: str
    content: bytes | None
    mode: int | None

    @property
    def existed(self) -> bool:
        """Return whether the unit existed when the snapshot was taken."""

        return self.content is not None


def _cap_text(value: str, limit: int) -> str:
    """Return text no longer than ``limit``, marking truncation when possible."""

    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _safe_command_detail(value: object) -> str:
    """Redact credentials and bound one line of command diagnostic output."""

    if isinstance(value, bytes):
        raw = value.decode("utf-8", errors="replace")
    else:
        raw = str(value or "")
    message = " ".join(raw[:16_000].split()).strip()
    message = _URI_CREDENTIAL_PATTERN.sub(r"\g<scheme>[REDACTED]@", message)
    message = _AUTHORIZATION_TOKEN_PATTERN.sub(
        lambda match: f"{match.group('scheme')} [REDACTED]",
        message,
    )
    message = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group('label')}{match.group('separator')}[REDACTED]",
        message,
    )
    message = _KNOWN_CREDENTIAL_PATTERN.sub("[REDACTED]", message)
    return _cap_text(message or "no diagnostic output", 900)


def _validate_executable_path(value: str | os.PathLike[str], field_name: str) -> str:
    """Return a trusted absolute executable path or raise a safety error."""

    raw = os.fspath(value)
    if not raw or any(ord(character) < 32 for character in raw):
        raise SystemdSafetyError(f"{field_name} must be a non-empty path.")
    if not (PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute()):
        raise SystemdSafetyError(f"{field_name} must be an absolute path.")
    return raw


def _validate_unit_name(unit_name: str) -> str:
    """Return a safe service or timer basename accepted as one argv item."""

    if not isinstance(unit_name, str) or not _UNIT_NAME_PATTERN.fullmatch(unit_name):
        raise SystemdSafetyError(
            "Unit name must be a plain .service or .timer basename."
        )
    if len(unit_name.encode("utf-8")) > 255:
        raise SystemdSafetyError("Unit name exceeds the filesystem name limit.")
    return unit_name


def _validate_calendar_expression(expression: str) -> str:
    """Return a bounded single-line calendar expression for systemd-analyze."""

    if not isinstance(expression, str):
        raise SystemdSafetyError("Calendar expression must be text.")
    cleaned = expression.strip()
    if (
        not cleaned
        or len(cleaned) > 1_024
        or cleaned.startswith("-")
        or any(ord(character) < 32 for character in cleaned)
    ):
        raise SystemdSafetyError(
            "Calendar expression must be non-empty, bounded, and single-line."
        )
    return cleaned


def _validate_verify_path(value: str | os.PathLike[str]) -> Path:
    """Return an existing absolute, regular, non-symlink unit path."""

    path = Path(value)
    if not path.is_absolute():
        raise SystemdSafetyError("Unit verification paths must be absolute.")
    if path.is_symlink() or not path.is_file():
        raise SystemdSafetyError(
            "Unit verification paths must be regular, non-symlink files."
        )
    _validate_unit_name(path.name)
    return path


def _command_diagnostic(completed: subprocess.CompletedProcess[str]) -> object:
    """Choose stderr, then stdout, as the diagnostic for a failed command."""

    return completed.stderr or completed.stdout or "no diagnostic output"


class SystemdController:
    """Execute a fixed allow-list of systemd operations without a shell."""

    def __init__(
        self,
        *,
        systemctl_path: str | os.PathLike[str],
        systemd_analyze_path: str | os.PathLike[str],
        timeout_seconds: float = 15.0,
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        """Create a controller with trusted executables and an injected runner."""

        if not 0 < timeout_seconds <= MAX_COMMAND_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between 0 and "
                f"{MAX_COMMAND_TIMEOUT_SECONDS}."
            )
        self._systemctl_path = _validate_executable_path(
            systemctl_path, "systemctl_path"
        )
        self._systemd_analyze_path = _validate_executable_path(
            systemd_analyze_path, "systemd_analyze_path"
        )
        self._timeout_seconds = float(timeout_seconds)
        self._command_runner = command_runner
        self._environment = dict(_CONTROLLED_ENVIRONMENT)

    def _run(
        self,
        executable: str,
        arguments: Sequence[str],
        *,
        operation: str,
        accepted_returncodes: Collection[int] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        """Run one fixed command and translate failures into bounded errors."""

        command = [executable, *arguments]
        try:
            completed = self._command_runner(
                command,
                capture_output=True,
                check=False,
                close_fds=True,
                encoding="utf-8",
                env=dict(self._environment),
                errors="replace",
                shell=False,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stderr or exc.stdout
            detail = f"timed out after {self._timeout_seconds:g} seconds"
            if output:
                detail = f"{detail}; {_safe_command_detail(output)}"
            raise SystemdCommandError(
                operation,
                returncode=None,
                detail=detail,
            ) from exc
        except OSError as exc:
            raise SystemdCommandError(
                operation,
                returncode=None,
                detail=_safe_command_detail(exc),
            ) from exc

        if completed.returncode not in accepted_returncodes:
            raise SystemdCommandError(
                operation,
                returncode=completed.returncode,
                detail=_safe_command_detail(_command_diagnostic(completed)),
            )
        return completed

    def validate_calendar(self, expression: str) -> str:
        """Validate one ``OnCalendar`` expression and return analyzer output."""

        calendar = _validate_calendar_expression(expression)
        completed = self._run(
            self._systemd_analyze_path,
            ["calendar", calendar],
            operation="systemd calendar validation",
        )
        return (completed.stdout or "").strip()

    def verify(self, paths: Sequence[str | os.PathLike[str]]) -> None:
        """Ask systemd-analyze to verify one bounded set of rendered unit files."""

        if not paths or len(paths) > 32:
            raise SystemdSafetyError("Between 1 and 32 unit files must be verified.")
        verified_paths = [_validate_verify_path(path) for path in paths]
        self._run(
            self._systemd_analyze_path,
            ["verify", *(str(path) for path in verified_paths)],
            operation="systemd unit verification",
        )

    def daemon_reload(self) -> None:
        """Reload systemd manager configuration after unit-file changes."""

        self._run(
            self._systemctl_path,
            ["daemon-reload"],
            operation="systemd daemon reload",
        )

    def enable(self, unit_name: str) -> None:
        """Enable one validated unit without starting it."""

        unit = _validate_unit_name(unit_name)
        self._run(
            self._systemctl_path,
            ["enable", unit],
            operation="systemd unit enable",
        )

    def start(self, unit_name: str) -> None:
        """Start one validated unit."""

        unit = _validate_unit_name(unit_name)
        self._run(
            self._systemctl_path,
            ["start", unit],
            operation="systemd unit start",
        )

    def stop(self, unit_name: str) -> None:
        """Stop one validated unit."""

        unit = _validate_unit_name(unit_name)
        self._run(
            self._systemctl_path,
            ["stop", unit],
            operation="systemd unit stop",
        )

    def disable(self, unit_name: str) -> None:
        """Disable one validated unit without implicitly stopping it."""

        unit = _validate_unit_name(unit_name)
        self._run(
            self._systemctl_path,
            ["disable", unit],
            operation="systemd unit disable",
        )

    def clean_state(self, unit_name: str) -> None:
        """Remove a stopped timer's persistent timestamp state before deletion."""

        unit = _validate_unit_name(unit_name)
        self._run(
            self._systemctl_path,
            ["clean", "--what=state", unit],
            operation="systemd timer-state cleanup",
        )

    def is_enabled(self, unit_name: str) -> bool:
        """Return whether systemd reports a validated unit as enabled."""

        unit = _validate_unit_name(unit_name)
        completed = self._run(
            self._systemctl_path,
            ["is-enabled", "--quiet", unit],
            operation="systemd enabled-state query",
            accepted_returncodes=(0, 1, 3, 4),
        )
        return completed.returncode == 0

    def is_active(self, unit_name: str) -> bool:
        """Return whether systemd reports a validated unit as active."""

        unit = _validate_unit_name(unit_name)
        completed = self._run(
            self._systemctl_path,
            ["is-active", "--quiet", unit],
            operation="systemd active-state query",
            accepted_returncodes=(0, 1, 3, 4),
        )
        return completed.returncode == 0


def validate_unit_root(
    unit_root: str | os.PathLike[str],
    *,
    enforce_posix_permissions: bool | None = None,
    require_root_trust: bool = False,
) -> Path:
    """Return a safe unit directory, optionally enforcing production ownership."""

    root = Path(unit_root)
    if not root.is_absolute():
        raise UnitFileSafetyError("Systemd unit root must be an absolute path.")
    try:
        root_status = root.lstat()
    except FileNotFoundError as exc:
        raise UnitFileSafetyError("Systemd unit root does not exist.") from exc
    if stat.S_ISLNK(root_status.st_mode):
        raise UnitFileSafetyError("Systemd unit root must not be a symlink.")
    if not stat.S_ISDIR(root_status.st_mode):
        raise UnitFileSafetyError("Systemd unit root must be a directory.")
    if require_root_trust:
        if not _supports_root_trust():
            raise UnitFileSafetyError(
                "Root-owned systemd unit storage requires a POSIX host."
            )
        _validate_root_trust_status(root_status, subject="Systemd unit root")
    check_permissions = (
        os.name == "posix"
        if enforce_posix_permissions is None
        else enforce_posix_permissions
    )
    if check_permissions and root_status.st_mode & stat.S_IWOTH:
        raise UnitFileSafetyError("Systemd unit root must not be world-writable.")
    return root


def _supports_root_trust() -> bool:
    """Return whether host file metadata carries meaningful POSIX ownership."""

    return os.name == "posix"


def _validate_root_trust_status(
    file_status: os.stat_result,
    *,
    subject: str,
) -> None:
    """Require root ownership and reject group- or world-writable metadata."""

    if file_status.st_uid != 0:
        raise UnitFileSafetyError(f"{subject} must be root-owned.")
    if file_status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise UnitFileSafetyError(
            f"{subject} must not be writable by its group or by others."
        )


def _has_managed_marker(content: bytes, marker: bytes) -> bool:
    """Return whether bytes begin with the exact managed-unit marker line."""

    return (
        content == marker
        or content.startswith(marker + b"\n")
        or content.startswith(marker + b"\r\n")
    )


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry update on POSIX filesystems."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class UnitFileStore:
    """Atomically manage marked unit files inside one trusted directory."""

    def __init__(
        self,
        unit_root: str | os.PathLike[str],
        *,
        managed_marker: str = MANAGED_UNIT_MARKER,
        validate_on_init: bool = True,
        require_root_trust: bool = False,
    ) -> None:
        """Create a store with optional lazy validation and production trust checks."""

        if (
            not managed_marker
            or "\n" in managed_marker
            or "\r" in managed_marker
            or "\x00" in managed_marker
        ):
            raise ValueError("managed_marker must be one non-empty line.")
        self._unit_root = Path(unit_root)
        self._require_root_trust = require_root_trust
        if validate_on_init:
            validate_unit_root(
                self._unit_root,
                require_root_trust=self._require_root_trust,
            )
        self._managed_marker = managed_marker.encode("utf-8")

    def _validate_root(self) -> None:
        """Recheck the unit root immediately before every filesystem mutation."""

        validate_unit_root(
            self._unit_root,
            require_root_trust=self._require_root_trust,
        )

    def _target(self, unit_name: str) -> Path:
        """Resolve a validated basename directly beneath the configured root."""

        return self._unit_root / _validate_unit_name(unit_name)

    def _validate_managed_content(self, content: bytes) -> None:
        """Reject content that lacks the exact ownership marker."""

        if not _has_managed_marker(content, self._managed_marker):
            raise UnitFileSafetyError(
                "Refusing to modify a unit without the FurnaceMind managed marker."
            )

    def snapshot(self, unit_name: str) -> UnitFileSnapshot:
        """Capture one absent or marked regular unit before any external action."""

        self._validate_root()
        target = self._target(unit_name)
        if target.is_symlink():
            raise UnitFileSafetyError("Managed unit target must not be a symlink.")
        try:
            target_status = target.stat()
        except FileNotFoundError:
            return UnitFileSnapshot(unit_name=unit_name, content=None, mode=None)
        if not stat.S_ISREG(target_status.st_mode):
            raise UnitFileSafetyError("Managed unit target must be a regular file.")
        if self._require_root_trust:
            _validate_root_trust_status(
                target_status,
                subject="Existing managed unit file",
            )
        if target_status.st_size > MAX_UNIT_FILE_BYTES:
            raise UnitFileSafetyError("Managed unit file exceeds the allowed size.")
        content = target.read_bytes()
        if len(content) > MAX_UNIT_FILE_BYTES:
            raise UnitFileSafetyError("Managed unit file exceeds the allowed size.")
        self._validate_managed_content(content)
        return UnitFileSnapshot(
            unit_name=unit_name,
            content=content,
            mode=stat.S_IMODE(target_status.st_mode),
        )

    def _atomic_write(self, target: Path, content: bytes, *, mode: int) -> None:
        """Replace one target from a flushed same-directory temporary file."""

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=self._unit_root,
        )
        temporary_path = Path(temporary_name)
        descriptor_open = True
        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                descriptor_open = False
                if hasattr(os, "fchmod"):
                    os.fchmod(temporary_file.fileno(), mode)
                else:
                    os.chmod(temporary_path, mode)
                if self._require_root_trust:
                    _validate_root_trust_status(
                        os.fstat(temporary_file.fileno()),
                        subject="Temporary managed unit file",
                    )
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, target)
            _fsync_directory(self._unit_root)
        except Exception:
            if descriptor_open:
                os.close(file_descriptor)
            if temporary_path.exists() or temporary_path.is_symlink():
                temporary_path.unlink()
            raise

    def install(self, unit_name: str, content: str) -> UnitFileSnapshot:
        """Atomically install marked UTF-8 content and return its prior state."""

        if not isinstance(content, str):
            raise TypeError("Unit content must be text.")
        encoded = content.encode("utf-8")
        if not encoded or len(encoded) > MAX_UNIT_FILE_BYTES or b"\x00" in encoded:
            raise UnitFileSafetyError(
                "Unit content is empty, too large, or contains NUL."
            )
        self._validate_managed_content(encoded)
        previous = self.snapshot(unit_name)
        if previous.content is not None:
            self._validate_managed_content(previous.content)
        target = self._target(unit_name)
        try:
            self._atomic_write(target, encoded, mode=0o644)
        except Exception:
            try:
                self._restore_failed_install(
                    target=target,
                    attempted_content=encoded,
                    previous=previous,
                )
            except Exception as rollback_exc:
                raise UnitFileSafetyError(
                    "Unit installation failed and its prior state could not be "
                    "durably restored."
                ) from rollback_exc
            raise
        return previous

    def _restore_failed_install(
        self,
        *,
        target: Path,
        attempted_content: bytes,
        previous: UnitFileSnapshot,
    ) -> None:
        """Best-effort restore state when failure may follow ``os.replace``."""

        if previous.content is not None:
            if target.is_symlink() or not target.is_file():
                raise UnitFileSafetyError(
                    "Unit target disappeared or changed type during failed install."
                )
            current = target.read_bytes()
            current_mode = stat.S_IMODE(target.stat().st_mode)
            previous_mode = previous.mode if previous.mode is not None else 0o644
            if current == previous.content:
                if current_mode == previous_mode:
                    return
                self._atomic_write(target, previous.content, mode=previous_mode)
                return
            if current != attempted_content:
                raise UnitFileSafetyError(
                    "Unit target changed concurrently after failed install."
                )
            self._atomic_write(target, previous.content, mode=previous_mode)
            return
        if not target.exists() and not target.is_symlink():
            return
        if target.is_symlink() or not target.is_file():
            raise UnitFileSafetyError(
                "Refusing to remove an unexpected target after failed install."
            )
        current = target.read_bytes()
        if current != attempted_content:
            raise UnitFileSafetyError(
                "Unit target changed concurrently after failed install."
            )
        target.unlink()
        _fsync_directory(self._unit_root)

    def restore(self, snapshot: UnitFileSnapshot) -> None:
        """Restore a prior marked snapshot, or remove a newly created unit."""

        current = self.snapshot(snapshot.unit_name)
        if current.content is not None:
            self._validate_managed_content(current.content)
        if snapshot.content is None:
            if current.content is not None:
                target = self._target(snapshot.unit_name)
                target.unlink()
                _fsync_directory(self._unit_root)
            return
        self._validate_managed_content(snapshot.content)
        mode = snapshot.mode if snapshot.mode is not None else 0o644
        self._atomic_write(
            self._target(snapshot.unit_name),
            snapshot.content,
            mode=mode,
        )

    def remove(self, unit_name: str) -> UnitFileSnapshot:
        """Remove one existing marked unit and return its recoverable snapshot."""

        previous = self.snapshot(unit_name)
        if previous.content is None:
            return previous
        self._validate_managed_content(previous.content)
        self._target(unit_name).unlink()
        _fsync_directory(self._unit_root)
        return previous
