"""Safe runtime cleanup service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from furnace_data.runtime_paths import get_runtime_dir, runtime_path


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    category: str
    size_bytes: int
    age_hours: float


class RuntimeCleanupService:
    """Delete only safe files under EVONITH_RUNTIME_DIR."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def dry_run(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.run({**(options or {}), "dry_run": True})

    def run(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.settings.cleanup_enabled:
            raise ApiError("CLEANUP_DISABLED", "Runtime cleanup is disabled.", 409)
        options = options or {}
        dry_run = bool(options.get("dry_run", self.settings.cleanup_dry_run_default))
        include_logs = bool(options.get("include_logs", self.settings.cleanup_include_logs))
        include_uploads = bool(options.get("include_uploads", self.settings.cleanup_include_uploads))
        max_delete = min(
            self.settings.cleanup_max_delete_per_run,
            max(1, int(options.get("max_delete", self.settings.cleanup_max_delete_per_run))),
        )
        candidates = self._candidates(include_logs=include_logs, include_uploads=include_uploads)
        selected = candidates[:max_delete]
        deleted = 0
        bytes_deleted = 0
        errors: list[dict[str, Any]] = []
        for candidate in selected:
            if dry_run:
                continue
            try:
                self._assert_safe(candidate.path)
                candidate.path.unlink()
                deleted += 1
                bytes_deleted += candidate.size_bytes
            except OSError as exc:
                errors.append({"path": self._label(candidate.path), "error": str(exc)})
        return {
            "dry_run": dry_run,
            "would_delete": len(selected),
            "deleted": deleted,
            "bytes_selected": sum(item.size_bytes for item in selected),
            "bytes_deleted": bytes_deleted,
            "max_delete": max_delete,
            "truncated": len(candidates) > len(selected),
            "candidates": [
                {
                    "path": self._label(item.path),
                    "category": item.category,
                    "size_bytes": item.size_bytes,
                    "age_hours": round(item.age_hours, 2),
                }
                for item in selected[:100]
            ],
            "errors": errors,
        }

    def _candidates(self, *, include_logs: bool, include_uploads: bool) -> list[CleanupCandidate]:
        roots: list[tuple[str, Path, int]] = [
            ("temp", runtime_path("temp"), self.settings.cleanup_temp_ttl_hours),
            ("jobs", runtime_path("jobs"), self.settings.cleanup_job_ttl_hours),
            ("compute_artifacts", runtime_path("compute", "artifacts"), self.settings.cleanup_artifact_ttl_hours),
            ("dataset_artifacts", runtime_path("datasets", "results", "artifacts"), self.settings.cleanup_artifact_ttl_hours),
        ]
        if include_logs:
            roots.append(("logs", runtime_path("logs"), self.settings.cleanup_job_ttl_hours))
        if include_uploads:
            roots.append(("uploads", runtime_path("uploads"), self.settings.cleanup_artifact_ttl_hours))
        now = datetime.now(timezone.utc)
        output: list[CleanupCandidate] = []
        for category, root, ttl_hours in roots:
            cutoff = now - timedelta(hours=max(1, int(ttl_hours)))
            if not root.exists() or not root.is_dir():
                continue
            for path in root.rglob("*"):
                try:
                    self._assert_safe(path)
                    if path.is_symlink() or not path.is_file() or path.name == ".gitkeep":
                        continue
                    stat = path.stat()
                    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    if modified > cutoff:
                        continue
                    output.append(
                        CleanupCandidate(
                            path=path,
                            category=category,
                            size_bytes=stat.st_size,
                            age_hours=(now - modified).total_seconds() / 3600,
                        )
                    )
                except OSError:
                    continue
        output.sort(key=lambda item: item.age_hours, reverse=True)
        return output

    @staticmethod
    def _assert_safe(path: Path) -> None:
        runtime_root = get_runtime_dir().resolve()
        resolved = path.resolve()
        resolved.relative_to(runtime_root)

    @staticmethod
    def _label(path: Path) -> str:
        try:
            return path.resolve().relative_to(get_runtime_dir().resolve()).as_posix()
        except Exception:
            return "[RUNTIME_DIR]"

