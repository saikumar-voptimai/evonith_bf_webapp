"""SQLAlchemy engine/session bootstrap helpers for daemon job persistence."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DAEMON_JOBS_DB_PATH = (
    PROJECT_ROOT / "src" / "assets" / "data" / "daemon_jobs.sqlite3"
)
DEFAULT_DAEMON_JOBS_DB_URL = f"sqlite:///{DEFAULT_DAEMON_JOBS_DB_PATH.as_posix()}"


def resolve_daemon_jobs_db_url(db_url: str | None = None) -> str:
    """Resolve the daemon jobs database URL from explicit input or environment."""
    if db_url:
        return db_url
    return os.getenv("DAEMON_JOBS_DB_URL", DEFAULT_DAEMON_JOBS_DB_URL)


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


def build_daemon_jobs_engine(db_url: str | None = None) -> Engine:
    """Build and return the SQLAlchemy engine for daemon job persistence."""
    resolved_url = resolve_daemon_jobs_db_url(db_url=db_url)
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


def build_daemon_jobs_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a reusable SQLAlchemy session factory."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
