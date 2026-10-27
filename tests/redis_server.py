"""A throw-away Redis server for tests of the event stream and the job queue.

Starts ``redis-server`` (no persistence) on a free port in a temporary directory; tests that need it are skipped when
the binary is not installed (install: ``apt install redis-server``; Windows: Memurai or WSL), and fail instead with
``BF_TEST_REQUIRE_SERVICES=1`` (CI; marker ``requires_service`` in ``tests/conftest.py``).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

BINARY = shutil.which("redis-server")
needs_redis = pytest.mark.requires_service(BINARY is not None, "redis-server is not installed")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def redis_server(tmp_path: Path) -> Iterator[str]:
    """Yield the URL of a fresh Redis server; stop it afterwards."""
    assert BINARY is not None
    port = _free_port()
    proc = subprocess.Popen(
        [
            BINARY,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("redis-server did not start")
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
