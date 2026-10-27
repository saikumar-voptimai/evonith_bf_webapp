"""Alembic environment of the ``twin`` schema (SV-0.9, #278).

    alembic -n twin upgrade head
    alembic -n twin downgrade -1
    alembic -n twin -x url=postgresql://... upgrade head

Database URL: ``-x url=``, else the section's ``sqlalchemy.url``, else ``BF_TWIN_DB_URL``, else ``DATABASE_URL``.
The schema is created here, before Alembic writes its version table ``twin.alembic_version``; the migrations create
and drop the tables inside it. Only objects of the ``twin`` schema are compared by ``--autogenerate``.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool, text

from data.twin.engine import resolve_twin_url
from data.twin.models import SCHEMA, TwinBase

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = TwinBase.metadata
VERSION_TABLE = "alembic_version"


def database_url() -> str:
    url = context.get_x_argument(as_dictionary=True).get("url") or config.get_main_option("sqlalchemy.url")
    url = resolve_twin_url(url or None)
    if not url:
        raise SystemExit("No database URL: pass -x url=..., or set BF_TWIN_DB_URL or DATABASE_URL.")
    if url.startswith("postgres://"):
        url = "postgresql://" + url.removeprefix("postgres://")
    return url


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    """Compare only the twin schema, and not Alembic's own version table."""
    if type_ == "schema":
        return name == SCHEMA
    if type_ == "table":
        return name != VERSION_TABLE
    return True


def _configure(**options: object) -> None:
    context.configure(
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_name=include_name,
        compare_type=True,
        **options,
    )


def run_migrations_offline() -> None:
    """Write the SQL script (``--sql``) instead of running it."""
    _configure(url=database_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.commit()
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
