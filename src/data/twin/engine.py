"""Connection to the ``twin`` schema (SV-0.9, #278).

URL: ``BF_TWIN_DB_URL``, else ``DATABASE_URL`` (Neon). The twin store writes, so it never falls back to a read replica.
``BF_TWIN_DB_URL=off`` disables the store (the test suite sets it). A SQLite URL serves the local outage buffer
(SV-6.5) and the portability test: the schema prefix is removed there.
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from data.twin.models import SCHEMA

URL_ENV = ("BF_TWIN_DB_URL", "DATABASE_URL")
OFF = frozenset({"off", "none", "0", "false"})


def resolve_twin_url(url: str | None = None) -> str | None:
    """The explicit URL, else the first of ``BF_TWIN_DB_URL`` and ``DATABASE_URL`` that is set, else None
    (also when ``BF_TWIN_DB_URL`` is ``off``)."""
    if url:
        return url
    if os.environ.get("BF_TWIN_DB_URL", "").strip().lower() in OFF:
        return None
    for name in URL_ENV:
        value = os.environ.get(name)
        if value:
            return value
    return None


def build_engine(url: str, *, application_name: str = "twin") -> Engine:
    """Pooled engine for PostgreSQL (pre-ping, 10 s connect timeout, recycled after 30 min: Neon suspends idle
    computes); for SQLite, an engine without the ``twin`` schema prefix."""
    if url.startswith("sqlite"):
        return create_engine(url).execution_options(schema_translate_map={SCHEMA: None})
    if url.startswith("postgres://"):  # scheme of some hosted URLs; SQLAlchemy 2 accepts only postgresql://
        url = "postgresql://" + url.removeprefix("postgres://")
    options: dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 5,
        "pool_recycle": 1800,
    }
    if url.startswith("postgresql"):
        options["connect_args"] = {"connect_timeout": 10, "application_name": application_name}
    return create_engine(url, **options)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions whose objects stay readable after commit (repositories return detached rows)."""
    return sessionmaker(engine, expire_on_commit=False, autoflush=False)
