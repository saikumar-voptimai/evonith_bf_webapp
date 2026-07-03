"""SQLAlchemy engine/session bootstrap helpers for ticket persistence."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from furnace_data.runtime_paths import get_feedback_db_path, get_repo_root

PROJECT_ROOT = get_repo_root()
LEGACY_TICKETS_DB_PATHS = (
    PROJECT_ROOT / "src" / "storage" / "feedback" / "tickets.db",
    PROJECT_ROOT / "storage" / "feedback" / "tickets.db",
)


def default_tickets_db_url() -> str:
    """Return the default runtime-backed SQLite ticket database URL."""
    return f"sqlite:///{get_feedback_db_path().as_posix()}"


def resolve_tickets_db_url(db_url: str | None = None) -> str:
    """Resolve the ticket database URL from explicit input or environment.

    Args:
        db_url: Optional explicit database URL.

    Returns:
        A SQLAlchemy-compatible database URL.
    """
    if db_url:
        return db_url
    return os.getenv("TICKETS_DB_URL", default_tickets_db_url())


def _sqlite_url_to_path(db_url: str) -> Path | None:
    """Return the filesystem path for a file-based SQLite URL."""
    if not db_url.startswith("sqlite:///"):
        return None

    sqlite_path = db_url.replace("sqlite:///", "", 1)
    if sqlite_path == ":memory:":
        return None

    path = Path(sqlite_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    """Create parent directories for file-based SQLite URLs."""
    db_path = _sqlite_url_to_path(db_url)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_copy_legacy_sqlite_db(*, db_url: str, explicit_db_url: str | None) -> None:
    """Copy old tickets DB to new default location once, if present."""
    if explicit_db_url is not None:
        return
    if os.getenv("TICKETS_DB_URL"):
        return
    if db_url != default_tickets_db_url():
        return

    target_path = _sqlite_url_to_path(db_url)
    if target_path is None:
        return

    if target_path.exists():
        return

    legacy_path = next((path for path in LEGACY_TICKETS_DB_PATHS if path.exists()), None)
    if legacy_path is None:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_path, target_path)


def build_tickets_engine(db_url: str | None = None) -> Engine:
    """Build and return the SQLAlchemy engine for ticket persistence."""
    resolved_url = resolve_tickets_db_url(db_url=db_url)
    _maybe_copy_legacy_sqlite_db(db_url=resolved_url, explicit_db_url=db_url)
    _ensure_sqlite_parent_dir(resolved_url)

    connect_args = (
        {"check_same_thread": False} if resolved_url.startswith("sqlite") else {}
    )
    return create_engine(
        resolved_url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def build_tickets_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a reusable SQLAlchemy session factory."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
