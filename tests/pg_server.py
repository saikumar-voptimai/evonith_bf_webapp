"""A throw-away PostgreSQL server for the tests of the ``twin`` schema (SV-0.9, #278).

``BF_TEST_DATABASE_URL`` is used when set: a server whose role may create databases (the CI service container, or a
Neon branch made for the test). Otherwise ``initdb`` and ``pg_ctl`` (from PATH, ``pg_config --bindir`` or
``/usr/lib/postgresql/<version>/bin``) start a cluster in a temporary directory, as the ``postgres`` system user when
the tests run as root (PostgreSQL refuses root). Tests that need it are skipped without either, and fail instead with
``BF_TEST_REQUIRE_SERVICES=1`` (CI; marker ``requires_service`` in ``tests/conftest.py``).

Every test module works in a database of its own (``fresh_database``), dropped afterwards, so the server's existing
databases and schemas are never touched.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.pool import NullPool

SERVER_URL = os.environ.get("BF_TEST_DATABASE_URL") or None


def _bindir() -> Path | None:
    found = shutil.which("initdb")
    if found:
        return Path(found).parent
    pg_config = shutil.which("pg_config")
    if pg_config:
        out = subprocess.run([pg_config, "--bindir"], capture_output=True, text=True, check=False).stdout.strip()
        if out and (Path(out) / "initdb").exists():
            return Path(out)

    def major(path: str) -> int:
        m = re.search(r"/(\d+)/bin$", path)
        return int(m.group(1)) if m else 0

    candidates = sorted(glob.glob("/usr/lib/postgresql/*/bin"), key=major, reverse=True)
    for candidate in candidates:
        if (Path(candidate) / "initdb").exists():
            return Path(candidate)
    return None


BINDIR = None if SERVER_URL else _bindir()
needs_postgres = pytest.mark.requires_service(
    SERVER_URL is not None or BINDIR is not None,
    "no PostgreSQL: set BF_TEST_DATABASE_URL or install the PostgreSQL server (initdb, pg_ctl)",
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _as_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


@contextmanager
def postgres_server() -> Iterator[str]:
    """Yield the URL of a server's ``postgres`` database; a local cluster is stopped and removed afterwards."""
    if SERVER_URL is not None:
        yield SERVER_URL
        return
    assert BINDIR is not None
    root = Path(tempfile.mkdtemp(prefix="bf-pg-"))
    run_as: list[str] = []
    if _as_root():
        shutil.chown(root, "postgres")
        run_as = ["runuser", "-u", "postgres", "--"]
    data = root / "data"
    port = _free_port()
    options = f"-p {port} -c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off -c full_page_writes=off"
    if os.name == "posix":
        options += f" -k {root}"
    subprocess.run(
        [*run_as, str(BINDIR / "initdb"), "-D", str(data), "-A", "trust", "-U", "postgres", "-E", "UTF8", "--no-sync"],
        check=True,
        capture_output=True,
    )
    pg_ctl = [*run_as, str(BINDIR / "pg_ctl"), "-D", str(data)]
    # On Windows the server inherits pg_ctl's output handles, so captured pipes would never close; its log goes to
    # server.log through -l in any case.
    output = (
        {"capture_output": True} if os.name == "posix" else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    )
    subprocess.run(
        [*pg_ctl, "-l", str(root / "server.log"), "-w", "-t", "60", "-o", options, "start"], check=True, **output
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("PostgreSQL did not start") from None
                time.sleep(0.1)
        yield f"postgresql+psycopg2://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run([*pg_ctl, "-m", "immediate", "-w", "stop"], capture_output=True, check=False)
        shutil.rmtree(root, ignore_errors=True)


@contextmanager
def fresh_database(server_url: str) -> Iterator[str]:
    """Create the database ``twin_test_<hex>`` on the server, yield its URL and drop it afterwards."""
    url = make_url(server_url)
    if url.drivername in ("postgresql", "postgres"):
        url = url.set(drivername="postgresql+psycopg2")
    name = f"twin_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # e.g. a role without CREATEDB
        admin.dispose()
        if os.environ.get("BF_TEST_REQUIRE_SERVICES") == "1":
            raise
        pytest.skip(f"cannot create a test database on {url.host}: {exc}")
    try:
        yield url.set(database=name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()
