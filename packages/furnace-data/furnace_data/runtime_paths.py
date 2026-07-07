"""Shared runtime data paths for the Evonith BF application.

Runtime files are mutable local artifacts: databases, uploads, generated CSVs,
logs, caches, and temporary files.  Source assets and configuration remain in
the repository; new runtime writes should use the helpers in this module.
"""

from __future__ import annotations

import os
from pathlib import Path

_RUNTIME_ENV_VAR = "EVONITH_RUNTIME_DIR"
_DEFAULT_RUNTIME_DIR = "runtime"

_RUNTIME_SUBDIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cache", ("cache",)),
    ("jobs", ("jobs",)),
    ("uploads", ("uploads",)),
    ("feedback_uploads", ("uploads", "feedback")),
    ("feedback", ("feedback",)),
    ("datasets", ("datasets",)),
    ("dataset_results", ("datasets", "results")),
    ("dataset_static", ("datasets", "static")),
    ("logs", ("logs",)),
    ("qdrant", ("qdrant",)),
    ("temp", ("temp",)),
)


def get_repo_root() -> Path:
    """Return the repository root containing the canonical application sources."""
    current = Path(__file__).resolve()
    canonical_markers = (
        Path("apps") / "backend_api" / "app",
        Path("apps") / "frontend_streamlit",
        Path("packages") / "furnace-data" / "furnace_data",
    )
    for candidate in current.parents:
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").exists():
            return candidate
        if all((candidate / marker).exists() for marker in canonical_markers):
            return candidate
    return current.parents[2]


def get_runtime_dir() -> Path:
    """Return the configured runtime directory without creating it."""
    configured = os.getenv(_RUNTIME_ENV_VAR, "").strip()
    raw_path = Path(configured).expanduser() if configured else Path(_DEFAULT_RUNTIME_DIR)
    if not raw_path.is_absolute():
        raw_path = get_repo_root() / raw_path
    return raw_path.resolve()


def runtime_path(*parts: str, create_parent: bool = False) -> Path:
    """Return a path under the runtime directory.

    Args:
        *parts: Path components below the runtime root.
        create_parent: When true, create the returned path's parent directory.
    """
    path = get_runtime_dir().joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_runtime_dirs() -> dict[str, Path]:
    """Create and return the standard runtime directory layout."""
    paths: dict[str, Path] = {"runtime": get_runtime_dir()}
    for name, parts in _RUNTIME_SUBDIRS:
        path = runtime_path(*parts)
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths


def get_feedback_dir() -> Path:
    """Return the feedback database directory."""
    return runtime_path("feedback")


def get_feedback_db_path() -> Path:
    """Return the feedback ticket SQLite database path."""
    return runtime_path("feedback", "tickets.db")


def get_feedback_upload_dir() -> Path:
    """Return the feedback screenshot upload directory."""
    return runtime_path("uploads", "feedback")


def get_dataset_results_dir() -> Path:
    """Return the generated dataset result directory."""
    return runtime_path("datasets", "results")


def get_dataset_static_dir() -> Path:
    """Return the generated static dataset cache directory."""
    return runtime_path("datasets", "static")


def get_jobs_dir() -> Path:
    """Return the runtime jobs directory."""
    return runtime_path("jobs")


def get_cache_dir() -> Path:
    """Return the runtime cache directory."""
    return runtime_path("cache")


def get_logs_dir() -> Path:
    """Return the runtime logs directory."""
    return runtime_path("logs")


def get_temp_dir() -> Path:
    """Return the runtime temporary-file directory."""
    return runtime_path("temp")


def get_qdrant_runtime_dir() -> Path:
    """Return the runtime directory reserved for local Qdrant data."""
    return runtime_path("qdrant")
