"""SQLAlchemy 2.0 engine and session bootstrap helpers."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

def resolve_database_url(db_url: str | None = None) -> str:
    """Resolve relational database URL from explicit arg or environment.

    Args:
        db_url: Optional explicit URL override.

    Returns:
        Resolved SQLAlchemy database URL.

    Raises:
        ValueError: If URL is missing both in args and environment.
    """
    resolved = db_url or os.getenv("DATABASE_URL")
    if not resolved:
        raise ValueError("Missing DATABASE_URL environment variable.")
    return resolved


def build_relational_engine(db_url: str | None = None) -> Engine:
    """Build a pooled SQLAlchemy engine for relational operations."""
    resolved_url = resolve_database_url(db_url=db_url)
    if resolved_url.startswith("sqlite"):
        raise ValueError("Shared relational persistence requires PostgreSQL.")
    return create_engine(resolved_url, future=True, pool_pre_ping=True)


def build_relational_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a SQLAlchemy session factory with stable defaults."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
