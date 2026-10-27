"""Append-only JSON-lines files shared by several processes (SV-1.5, #285): incremental reads and locked appends.

The forecast ledgers (``utils.bmo.si_forecast.ledger``, ``utils.bmo.charged_coke.ledger``) are written by the worker
every 5 min and hourly, and read by page 9, the API and the worker itself. Two properties of a single-process ledger do
not hold any more and are provided here:

* **Appends.** "Read, check that the key is new, append" must exclude other processes, not only other threads.
  ``exclusive()`` holds a process-wide thread lock and an OS lock on ``<file>.lock`` (``fcntl.flock`` on POSIX,
  ``msvcrt.locking`` on Windows) for the whole sequence.
* **Reads.** Every read used to parse the whole file. ``records()`` parses it once per process and afterwards only the
  bytes appended since (byte offset per path). A file that shrank or was replaced (another inode) is read again from
  the start. A line is consumed only once its newline has been written, so a reader never sees half a record.

The file format is unchanged: one JSON object per line, UTF-8, ``\\n`` line ends.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCK_TIMEOUT_S = 30.0


@dataclass
class _Cache:
    ident: tuple[int, int] | None = None  # (st_dev, st_ino) of the file that was read
    offset: int = 0  # bytes consumed (up to the last complete line)
    records: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0  # nesting of exclusive() in the thread that holds ``lock``


_CACHES: dict[str, _Cache] = {}
_CACHES_LOCK = threading.Lock()


def _cache(path: Path) -> _Cache:
    key = os.path.abspath(path)
    with _CACHES_LOCK:
        cache = _CACHES.get(key)
        if cache is None:
            cache = _CACHES[key] = _Cache()
        return cache


@contextmanager
def _os_lock(lock_path: Path, timeout_s: float) -> Iterator[None]:
    """Exclusive OS lock on ``lock_path`` (created if missing), polled until ``timeout_s``."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{lock_path} is held by another process for more than {timeout_s:g} s"
                    ) from None
                time.sleep(0.02)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class JsonlLog:
    """One append-only JSON-lines file. Instances are cheap; the parsed records are shared per path and process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cache = _cache(self.path)

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def records(self) -> list[dict[str, Any]]:
        """Every complete record, oldest first. The dicts are shared by all readers of the path: do not modify them."""
        c = self._cache
        with c.lock:
            try:
                st = os.stat(self.path)
            except FileNotFoundError:
                c.ident, c.offset, c.records = None, 0, []
                return []
            ident = (st.st_dev, st.st_ino)
            if ident != c.ident or st.st_size < c.offset:
                c.ident, c.offset, c.records = ident, 0, []
            if st.st_size > c.offset:
                with open(self.path, "rb") as handle:
                    handle.seek(c.offset)
                    chunk = handle.read()
                end = chunk.rfind(b"\n") + 1  # a partly written last line waits for its newline
                if end:
                    for line in chunk[:end].decode("utf-8").splitlines():
                        if line.strip():
                            c.records.append(json.loads(line))
                    c.offset += end
            return list(c.records)

    def append(self, record: dict[str, Any]) -> None:
        """Append one record (already JSON-serialisable). Call inside ``exclusive()`` when the append depends on what
        the file holds."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.exclusive():
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    @contextmanager
    def exclusive(self, timeout_s: float = LOCK_TIMEOUT_S) -> Iterator[None]:
        """Hold this file against every other writer, in this process and in others (re-entrant in one thread)."""
        c = self._cache
        with c.lock:
            if c.depth:  # already held by this thread: the OS lock is held too
                c.depth += 1
                try:
                    yield
                finally:
                    c.depth -= 1
                return
            with _os_lock(self.lock_path, timeout_s):
                c.depth = 1
                try:
                    yield
                finally:
                    c.depth = 0


__all__ = ["LOCK_TIMEOUT_S", "JsonlLog"]
