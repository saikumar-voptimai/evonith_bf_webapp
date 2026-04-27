"""SQLAlchemy engine/session bootstrap helpers for ticket persistence."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

#TODO: Default db should be in config file
DEFAULT_TICKETS_DB_URL = "sqlite:///./storage/feedback/tickets.db"


def resolve_tickets_db_url(db_url: str | None = None) -> str:
    """Resolve the ticket database URL from explicit input or environment.

    Args:
        db_url: Optional explicit database URL.

    Returns:
        A SQLAlchemy-compatible database URL.
    """
    if db_url:
        return db_url
    return os.getenv("TICKETS_DB_URL", DEFAULT_TICKETS_DB_URL)


def _ensure_sqlite_parent_dir(db_url: str) -> None:
    """Create parent directories for file-based SQLite URLs."""
    if not db_url.startswith("sqlite:///"):
        return

    sqlite_path = db_url.replace("sqlite:///", "", 1)
    if sqlite_path == ":memory:":
        return

    db_path = Path(sqlite_path)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)


def build_tickets_engine(db_url: str | None = None) -> Engine:
    """Build and return the SQLAlchemy engine for ticket persistence."""
    resolved_url = resolve_tickets_db_url(db_url=db_url)
    _ensure_sqlite_parent_dir(resolved_url)

    connect_args = {"check_same_thread": False} if resolved_url.startswith("sqlite") else {}
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
