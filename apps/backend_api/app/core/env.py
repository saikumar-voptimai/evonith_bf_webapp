"""Backend environment bootstrap helpers."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


_BACKEND_ENV_FILES = (".env", "apps/backend_api/.env")


def load_backend_env(repo_root: Path | None = None) -> None:
    """Load backend .env files without overriding real environment variables."""
    root = repo_root or Path(__file__).resolve().parents[4]
    for relative in _BACKEND_ENV_FILES:
        load_dotenv(root / relative, override=False)