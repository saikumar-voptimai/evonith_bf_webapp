"""Tests for runtime-backed artifacts."""

from __future__ import annotations

import json
import multiprocessing
import os

import pandas as pd
import pytest

from apps.backend_api.app.services.artifact_service import (
    ArtifactExpiredError,
    ArtifactIdempotencyConflictError,
    create_csv_artifact,
    get_artifact_metadata,
    get_artifact_path,
)


def _create_idempotent_artifact_in_subprocess(
    runtime_dir: str,
    ready: object,
    start: object,
    results: object,
) -> None:
    """Create one export in a fresh process for registry contention coverage."""
    os.environ["EVONITH_RUNTIME_DIR"] = runtime_dir
    ready.wait(timeout=30)
    start.wait(timeout=30)
    try:
        artifact = create_csv_artifact(
            pd.DataFrame({"a": [1, 2]}),
            "multiprocess_export",
            owner_user_id="operator-1",
            query_fingerprint="same-query",
            idempotency_key="same-key",
        )
        results.put(("ok", artifact.artifact_id))
    except Exception as exc:  # pragma: no cover - asserted by parent process
        results.put(("error", repr(exc)))


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


def test_expired_artifact_is_deleted_on_access(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    artifact = create_csv_artifact(pd.DataFrame({"a": [1]}), "expired_export")
    path = get_artifact_path(artifact.artifact_id)
    metadata_path = path.parent / f"{artifact.artifact_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["expires_at"] = "2000-01-01T00:00:00+00:00"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactExpiredError):
        get_artifact_metadata(artifact.artifact_id)
    # An expiry tombstone keeps repeated download attempts semantically 410,
    # rather than degrading to a not-found response after the first access.
    with pytest.raises(ArtifactExpiredError):
        get_artifact_metadata(artifact.artifact_id)

    assert not path.exists()
    assert not metadata_path.exists()
    assert (path.parent / f"{artifact.artifact_id}.expired.json").exists()


def test_create_csv_artifact_enforces_owner_scoped_idempotency(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    kwargs = {
        "owner_user_id": "operator-1",
        "query_fingerprint": "query-a",
        "idempotency_key": "same-key",
    }
    first = create_csv_artifact(pd.DataFrame({"a": [1]}), "first", **kwargs)
    replay = create_csv_artifact(pd.DataFrame({"a": [2]}), "second", **kwargs)

    assert replay.artifact_id == first.artifact_id
    with pytest.raises(ArtifactIdempotencyConflictError):
        create_csv_artifact(
            pd.DataFrame({"a": [3]}),
            "conflict",
            owner_user_id="operator-1",
            query_fingerprint="query-b",
            idempotency_key="same-key",
        )


def test_idempotency_registry_is_atomic_across_processes(tmp_path):
    """Two Uvicorn-style workers must replay one owner/key export artifact."""
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    start = context.Event()
    results = context.Queue()
    runtime_dir = str(tmp_path / "runtime")
    processes = [
        context.Process(
            target=_create_idempotent_artifact_in_subprocess,
            args=(runtime_dir, ready, start, results),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        # Both workers rendezvous before they try to create the same export.
        start.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=10) for _ in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()

    assert [kind for kind, _ in outcomes] == ["ok", "ok"], outcomes
    assert len({artifact_id for _, artifact_id in outcomes}) == 1
    artifact_dir = tmp_path / "runtime" / "datasets" / "results" / "artifacts"
    assert len(list(artifact_dir.glob("*.csv"))) == 1
    assert len(list(artifact_dir.glob("*.json"))) == 1


def test_expired_idempotency_mapping_allows_a_new_export(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    first = create_csv_artifact(
        pd.DataFrame({"a": [1]}),
        "expired_export",
        owner_user_id="operator-1",
        query_fingerprint="query-a",
        idempotency_key="same-key",
    )
    metadata_path = (
        tmp_path
        / "runtime"
        / "datasets"
        / "results"
        / "artifacts"
        / f"{first.artifact_id}.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["expires_at"] = "2000-01-01T00:00:00+00:00"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    replacement = create_csv_artifact(
        pd.DataFrame({"a": [2]}),
        "replacement_export",
        owner_user_id="operator-1",
        query_fingerprint="query-b",
        idempotency_key="same-key",
    )

    assert replacement.artifact_id != first.artifact_id
