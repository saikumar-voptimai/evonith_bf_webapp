"""Runtime directory and disk status checks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from furnace_data.runtime_paths import get_runtime_dir, runtime_path

_EXPECTED_DIRS = (
    "cache",
    "jobs",
    "uploads",
    "feedback",
    "datasets",
    "logs",
    "temp",
    "compute",
    "copilot",
    "furnacemind",
    "audit",
)


class RuntimeStatusService:
    """Inspect runtime storage without exposing internal paths."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def status(self, *, create_missing: bool = True, include_sizes: bool = True) -> dict[str, Any]:
        runtime_dir = get_runtime_dir()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        checks: dict[str, str] = {
            "runtime_dir": "ok" if runtime_dir.exists() else "missing",
            "writable": "ok" if os.access(runtime_dir, os.W_OK) else "not_writable",
        }
        directories: dict[str, dict[str, Any]] = {}
        for name in _EXPECTED_DIRS:
            path = runtime_path(name)
            if create_missing:
                path.mkdir(parents=True, exist_ok=True)
            exists = path.exists() and path.is_dir()
            checks[name] = "ok" if exists else "missing"
            data: dict[str, Any] = {"exists": exists, "label": name}
            if include_sizes and exists:
                data.update(self._directory_stats(path, runtime_dir=runtime_dir))
            directories[name] = data

        disk: dict[str, Any] = {}
        warnings: list[dict[str, Any]] = []
        status = "ok"
        try:
            usage = shutil.disk_usage(runtime_dir)
            used_pct = (usage.used / usage.total) * 100 if usage.total else 0.0
            free_mb = usage.free / (1024 * 1024)
            disk = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_percent": round(used_pct, 2),
            }
            if free_mb < self.settings.runtime_min_free_mb:
                status = "degraded"
                warnings.append(
                    {
                        "code": "RUNTIME_FREE_SPACE_LOW",
                        "message": "Runtime free space is below the configured minimum.",
                    }
                )
            elif free_mb < self.settings.runtime_warn_free_mb:
                status = "warning"
                warnings.append(
                    {
                        "code": "RUNTIME_FREE_SPACE_WARNING",
                        "message": "Runtime free space is below the configured warning threshold.",
                    }
                )
        except OSError:
            status = "degraded"
            disk = {}
            warnings.append({"code": "RUNTIME_DISK_STATUS_FAILED", "message": "Could not inspect runtime disk usage."})

        if any(value != "ok" for value in checks.values()):
            status = "degraded"

        return {
            "status": status,
            "runtime": {"configured": True, "label": "runtime"},
            "checks": checks,
            "directories": directories,
            "disk": disk,
            "warnings": warnings,
        }

    @staticmethod
    def _directory_stats(path: Path, *, runtime_dir: Path, max_entries: int = 5000) -> dict[str, Any]:
        file_count = 0
        total_bytes = 0
        truncated = False
        try:
            for child in path.rglob("*"):
                if file_count >= max_entries:
                    truncated = True
                    break
                try:
                    resolved = child.resolve()
                    resolved.relative_to(runtime_dir.resolve())
                    if child.is_symlink() or not child.is_file():
                        continue
                    file_count += 1
                    total_bytes += child.stat().st_size
                except OSError:
                    continue
        except OSError:
            return {"file_count": 0, "size_bytes": 0, "scan_failed": True}
        return {"file_count": file_count, "size_bytes": total_bytes, "truncated": truncated}


def expected_runtime_dirs() -> tuple[str, ...]:
    return _EXPECTED_DIRS

