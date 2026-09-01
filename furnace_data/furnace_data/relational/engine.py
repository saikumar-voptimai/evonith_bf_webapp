"""SQLAlchemy 2.0 engine and session bootstrap helpers."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Checked in order. DATABASE_URL is the plant PostgreSQL; the others name a
# read replica used when the plant server is unreachable - it sits behind a
# firewall that only admits whitelisted addresses, so analysis run from a
# developer machine or CI often cannot reach it at all.
_URL_ENV_VARS = ("DATABASE_URL", "NEON_DATABASE_URL", "NEON_STR")


def resolve_database_url(db_url: str | None = None) -> str:
    """Resolve relational database URL from explicit arg or environment.

    Falls back through a replica if the primary is not configured. Note this
    only covers a MISSING primary, not an unreachable one - a URL that is set
    but refuses connections still raises from the driver, which is the right
    behaviour: silently reading a replica when the primary is down would hide
    an outage.

    Args:
        db_url: Optional explicit URL override.

    Returns:
        Resolved SQLAlchemy database URL.

    Raises:
        ValueError: If no URL is available from args or environment.
    """
    if db_url:
        return db_url
    for name in _URL_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    raise ValueError(
        "No database URL. Set one of: " + ", ".join(_URL_ENV_VARS)
    )


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
