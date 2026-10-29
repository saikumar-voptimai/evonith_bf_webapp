"""The release a process runs (SV-6.4, #318).

``deploy/jetson/release.sh`` writes ``RELEASE.json`` (tag, commit, installation time) into each release directory and
unpacks the SPA build next to it (``frontend/build/_app/version.json`` holds the build's version name). A process reads
the files of its own directory: systemd resolves ``/opt/bf-webapp/current`` once, at the start, so the record names the
release that runs, also after a later switch. In a checkout the record is absent (development).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_FILE = "RELEASE.json"
BUILD_VERSION_FILE = Path("frontend", "build", "_app", "version.json")
# what release.sh accepts as a tag and writes as a commit / build version (no quoting needed in its JSON)
_NAME = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,79}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class ReleaseInfo:
    """``tag`` / ``commit`` / ``installed_at`` from ``RELEASE.json``; ``frontend_build`` from the SPA build."""

    tag: str | None = None
    commit: str | None = None
    installed_at: datetime | None = None
    frontend_build: str | None = None


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("release: %s unreadable: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _text(value: object, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def read_release(root: Path) -> ReleaseInfo:
    """The release record and SPA build version of ``root``; fields that are absent or malformed are None."""
    record = _read_json(root / RELEASE_FILE)
    build = _read_json(root / BUILD_VERSION_FILE)
    return ReleaseInfo(
        tag=_text(record.get("tag"), _NAME),
        commit=_text(record.get("commit"), _COMMIT),
        installed_at=_time(record.get("installed_at")),
        frontend_build=_text(build.get("version"), _NAME),
    )


@lru_cache(maxsize=1)
def current_release() -> ReleaseInfo:
    """The release of this process, read once (the directory does not change under a running process)."""
    return read_release(REPO_ROOT)
