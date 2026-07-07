"""Compatibility cleanup guards for removed legacy frontend surfaces."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_backend_sidecar_is_removed() -> None:
    assert not (REPO_ROOT / ("furnace-data" + "-service")).exists()


def test_legacy_frontend_folder_is_removed() -> None:
    assert not (REPO_ROOT / "src").exists()