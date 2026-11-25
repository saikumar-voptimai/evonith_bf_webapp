"""The plant PostgreSQL of the admin tables (``identity``, ``plant_master``, ``ops_config``) for the twin-api.

The Streamlit tools, the sign-in (``services.auth.NeonIdentityStore``) and every offline-data fetch read these tables on
``DATABASE_URL``. The twin store may sit on another database (``BF_TWIN_DB_URL``, e.g. a Neon branch whose copies of
the plant tables are not kept up to date), so the admin views and writes of ``services.plant_admin`` use this module's
database, not the twin store's:

* ``BF_PLANT_DB_URL``: a URL, or ``twin`` for the twin store's own database (the test suite sets ``twin``:
  ``data.db`` loads ``.env``, whose ``DATABASE_URL`` is the real plant server);
* else ``DATABASE_URL`` (never a read replica: an admin write must reach the plant server);
* else the twin store's database.

When the plant database is the twin store's, a write and its audit row commit in one transaction
(``PlantDatabase.shared``); otherwise ``services.audit.admin_write(target=...)`` commits the audit row first, then the
change. No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from data.twin.repositories import TwinStore

PLANT_URL_ENV = "BF_PLANT_DB_URL"
#: ``BF_PLANT_DB_URL`` value naming the twin store's own database.
SAME_AS_TWIN = "twin"

_lock = threading.Lock()
_databases: dict[str, PlantDatabase] = {}


@dataclass(frozen=True)
class PlantDatabase:
    """Sessions on the plant database; ``shared`` when they are the twin store's own (one transaction per write)."""

    sessions: sessionmaker[Session]
    shared: bool
    #: ``host/database`` for messages (no user, no password).
    label: str


def plant_url() -> str | None:
    """``BF_PLANT_DB_URL``, else ``DATABASE_URL``; None for the twin store's database (unset, or ``twin``)."""
    explicit = os.environ.get(PLANT_URL_ENV, "").strip()
    if explicit:
        return None if explicit.lower() == SAME_AS_TWIN else explicit
    return os.environ.get("DATABASE_URL", "").strip() or None


def _label(url: URL) -> str:
    return f"{url.host or 'local'}/{url.database or ''}"


def _store_url(store: TwinStore) -> URL:
    engine = store.engine if store.engine is not None else store.sessions.kw["bind"]
    return engine.url


def same_database(a: URL, b: URL) -> bool:
    """Same server, port, database and role (the password and query options aside)."""
    return (a.host, a.port, a.database, a.username) == (b.host, b.port, b.database, b.username)


def for_store(store: TwinStore) -> PlantDatabase:
    """The plant database for the admin tables next to ``store`` (see the module docstring)."""
    twin = _store_url(store)
    raw = plant_url()
    url = make_url(raw) if raw else None
    if url is None or same_database(url, twin):
        return PlantDatabase(store.sessions, True, _label(twin))
    key = url.render_as_string(hide_password=False)
    with _lock:
        database = _databases.get(key)
        if database is None:
            from furnace_data.relational import build_relational_engine, build_relational_session_factory

            engine = build_relational_engine(db_url=key)
            database = PlantDatabase(build_relational_session_factory(engine), False, _label(url))
            _databases[key] = database
    return database


def dispose() -> None:
    """Close the engines this module opened (tests; a process that changes ``DATABASE_URL``)."""
    with _lock:
        for database in _databases.values():
            database.sessions.kw["bind"].dispose()
        _databases.clear()


__all__ = ["PLANT_URL_ENV", "SAME_AS_TWIN", "PlantDatabase", "dispose", "for_store", "plant_url", "same_database"]
