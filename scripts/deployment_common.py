"""Shared helpers for Phase 13 deployment scripts."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SECRET_PATTERNS = (
    re.compile(
        r"(?m)^\s*(?:export\s+)?[A-Z0-9_]*(?:API[_-]?KEY|APIKEY|SECRET_KEY|PASSWORD|"
        r"ACCESS_TOKEN|REFRESH_TOKEN|AUTH_TOKEN)[A-Z0-9_]*"
        r"\s*=\s*(?!<|$|#)(?!change-me\b)(?!changeme\b)(?!dev-only-secret-change-me\b)[^\s#]+"
    ),
    re.compile(r"(?i)(postgresql|postgres|mysql)://[^:\s]+:[^@\s]+@"),
)
PLACEHOLDER_VALUES = {
    "",
    "change-me",
    "changeme",
    "dev-only-secret-change-me",
    "<set-a-strong-random-secret>",
    "<optional-set-outside-git>",
}
STANDARD_RUNTIME_SUBDIRS = (
    "cache",
    "jobs",
    "uploads",
    "uploads/feedback",
    "feedback",
    "datasets",
    "datasets/results",
    "datasets/static",
    "logs",
    "temp",
    "compute",
    "copilot",
    "furnacemind",
    "audit",
    "backups",
    "qdrant",
)


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def repo_root() -> Path:
    return _REPO_ROOT


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return default


def env_path(name: str, default: str | Path) -> Path:
    raw = os.getenv(name, str(default)).strip() or str(default)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def runtime_dir() -> Path:
    return env_path("EVONITH_RUNTIME_DIR", "runtime")


def backup_dir(runtime: Path | None = None) -> Path:
    runtime = runtime or runtime_dir()
    return env_path("EVONITH_BACKUP_DIR", runtime / "backups")


def is_production_like(profile: str) -> bool:
    return profile.lower() in {"production", "prod", "staging", "edge"}


def is_placeholder_secret(value: str | None) -> bool:
    normalized = str(value or "").strip()
    return normalized.lower() in PLACEHOLDER_VALUES or normalized.startswith("<")


def redact(value: str | None) -> str:
    if not value:
        return ""
    return "[REDACTED]"


def safe_status(results: list[CheckResult]) -> str:
    if any(result.status == "fail" for result in results):
        return "fail"
    if any(result.status == "warn" for result in results):
        return "warn"
    return "pass"


def print_results(results: list[CheckResult], *, json_output: bool = False) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "status": safe_status(results),
                    "checks": [
                        {
                            "name": result.name,
                            "status": result.status,
                            "message": result.message,
                            "details": result.details,
                        }
                        for result in results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    for result in results:
        print(f"{result.status.upper()} {result.name}: {result.message}")


def exit_code(results: list[CheckResult], *, strict_warnings: bool = False) -> int:
    if any(result.status == "fail" for result in results):
        return 1
    if strict_warnings and any(result.status == "warn" for result in results):
        return 1
    return 0


def unsafe_runtime_path(path: Path, *, allow_repo_root: bool = False) -> str | None:
    resolved = path.resolve()
    root = repo_root().resolve()
    if resolved == resolved.anchor:
        return "runtime directory cannot be filesystem root"
    blocked = {
        Path("/bin"),
        Path("/etc"),
        Path("/usr"),
        Path("/var"),
        Path("/opt"),
    }
    if os.name == "nt":
        blocked.update(Path(p) for p in (r"C:\Windows", r"C:\Program Files"))
    for candidate in blocked:
        try:
            if resolved == candidate.resolve():
                return f"runtime directory cannot be {candidate}"
        except OSError:
            continue
    if resolved == root and not allow_repo_root:
        return "runtime directory cannot be repository root"
    return None


def disk_free_mb(path: Path) -> int:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    return int(usage.free / 1024 / 1024)


def min_free_mb() -> int:
    raw = os.getenv("EVONITH_RUNTIME_MIN_FREE_MB", "1024")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1024


def warn_free_mb() -> int:
    raw = os.getenv("EVONITH_RUNTIME_WARN_FREE_MB", "4096")
    try:
        return max(0, int(raw))
    except ValueError:
        return 4096


def validate_writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".evonith-write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def scan_files_for_secrets(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else list(path.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() in {".pyc", ".png", ".jpg", ".webp"}:
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    findings.append(str(candidate.relative_to(repo_root())))
                    break
    return sorted(set(findings))


def run_python_script(script: str, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo_root() / "scripts" / script), *args],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def required_docs() -> list[Path]:
    root = repo_root()
    return [
        root / "docs" / "migration" / "phase-13-production-deployment-cutover.md",
        root / "docs" / "migration" / "phase-13-test-execution-report.md",
        root / "docs" / "testing" / "phase-13-testing-guide.md",
        root / "docs" / "deployment" / "production-deployment-guide.md",
        root / "docs" / "deployment" / "edge-device-deployment-guide.md",
        root / "docs" / "deployment" / "local-staging-deployment-guide.md",
        root / "docs" / "deployment" / "cutover-guide.md",
        root / "docs" / "deployment" / "rollback-guide.md",
        root / "docs" / "deployment" / "backup-restore-guide.md",
        root / "docs" / "deployment" / "release-checklist.md",
        root / "docs" / "deployment" / "environment-variables-production.md",
    ]


def is_within(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
