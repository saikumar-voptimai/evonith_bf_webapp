"""Tests for runtime-backed artifacts."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.artifact_service import create_csv_artifact, get_artifact_path


def test_create_csv_artifact_uses_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    artifact = create_csv_artifact(pd.DataFrame({"a": [1, 2]}), "test_export")

    path = get_artifact_path(artifact.artifact_id)

    assert path.exists()
    assert str(path).startswith(str(tmp_path / "runtime"))
    assert artifact.row_count == 2


def test_invalid_artifact_id_rejects_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    with pytest.raises(ValueError):
        get_artifact_path("../escape")
