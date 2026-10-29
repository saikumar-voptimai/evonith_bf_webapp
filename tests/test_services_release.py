"""The release a process runs (SV-6.4, #318): ``services.release`` reads the record ``deploy/jetson/release.sh`` writes,
and ``GET /api/version`` serves it (the SPA footer)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.routers import system
from api.settings import ApiSettings
from services.release import BUILD_VERSION_FILE, RELEASE_FILE, ReleaseInfo, read_release

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write(root: Path, record: object, build: object | None = None) -> None:
    (root / RELEASE_FILE).write_text(json.dumps(record), encoding="utf-8")
    if build is not None:
        (root / BUILD_VERSION_FILE).parent.mkdir(parents=True)
        (root / BUILD_VERSION_FILE).write_text(json.dumps(build), encoding="utf-8")


def test_record_and_build_version_are_read(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {"tag": "v1.2.0", "commit": COMMIT, "installed_at": "2026-10-29T17:05:00Z", "twin_revision_after": "twin_0009"},
        {"version": "v1.2.0"},
    )
    info = read_release(tmp_path)
    assert info == ReleaseInfo(
        tag="v1.2.0",
        commit=COMMIT,
        installed_at=datetime(2026, 10, 29, 17, 5, tzinfo=UTC),
        frontend_build="v1.2.0",
    )


def test_a_checkout_has_no_release(tmp_path: Path) -> None:
    assert read_release(tmp_path) == ReleaseInfo()


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"tag": "v1.2.0", "commit": "unknown"}, ReleaseInfo(tag="v1.2.0")),  # source archive without commit id
        ({"tag": "v1 2", "commit": "XYZ", "installed_at": "yesterday"}, ReleaseInfo()),
        (
            {"tag": 12, "installed_at": "2026-10-29T22:35:00"},
            ReleaseInfo(installed_at=datetime(2026, 10, 29, 22, 35, tzinfo=UTC)),
        ),
        (["not", "an", "object"], ReleaseInfo()),
    ],
)
def test_malformed_fields_are_none(tmp_path: Path, record: object, expected: ReleaseInfo) -> None:
    _write(tmp_path, record)
    assert read_release(tmp_path) == expected


def test_unreadable_record_is_no_release(tmp_path: Path) -> None:
    (tmp_path / RELEASE_FILE).write_text("{not json", encoding="utf-8")
    assert read_release(tmp_path) == ReleaseInfo()


def test_version_route_serves_the_release(monkeypatch: pytest.MonkeyPatch) -> None:
    info = ReleaseInfo(
        tag="v1.2.0", commit=COMMIT, installed_at=datetime(2026, 10, 29, 17, 5, tzinfo=UTC), frontend_build="v1.2.0"
    )
    monkeypatch.setattr(system, "current_release", lambda: info)
    body = TestClient(create_app(ApiSettings(static_dir=None))).get("/api/version").json()
    assert body["release"] == "v1.2.0" and body["git_sha"] == COMMIT and body["frontend_build"] == "v1.2.0"
    assert body["installed_at"].startswith("2026-10-29T17:05:00")


def test_version_route_without_a_release_uses_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system, "current_release", ReleaseInfo)
    body = TestClient(create_app(ApiSettings(static_dir=None))).get("/api/version").json()
    assert body["release"] is None and body["frontend_build"] is None and body["installed_at"] is None
    assert body["git_sha"] == system.get_settings().git_sha
