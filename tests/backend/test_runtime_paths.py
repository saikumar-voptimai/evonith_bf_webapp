from __future__ import annotations

import inspect

from furnace_data import runtime_paths


def test_default_runtime_dir_resolves_to_repo_runtime(monkeypatch) -> None:
    monkeypatch.delenv("EVONITH_RUNTIME_DIR", raising=False)

    assert runtime_paths.get_runtime_dir() == runtime_paths.get_repo_root() / "runtime"


def test_absolute_runtime_dir_is_respected(monkeypatch, tmp_path) -> None:
    runtime_dir = tmp_path / "evonith-runtime"
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(runtime_dir))

    assert runtime_paths.get_runtime_dir() == runtime_dir.resolve()


def test_relative_runtime_dir_resolves_from_repo_root(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", "local-runtime")

    assert (
        runtime_paths.get_runtime_dir()
        == (runtime_paths.get_repo_root() / "local-runtime").resolve()
    )


def test_ensure_runtime_dirs_creates_expected_layout(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    paths = runtime_paths.ensure_runtime_dirs()

    for key in (
        "cache",
        "jobs",
        "feedback_uploads",
        "feedback",
        "dataset_results",
        "dataset_static",
        "logs",
        "qdrant",
        "temp",
    ):
        assert paths[key].is_dir()


def test_runtime_path_create_parent_creates_parent(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    target = runtime_paths.runtime_path(
        "cache", "nested", "value.json", create_parent=True
    )

    assert target.parent.is_dir()
    assert not target.exists()


def test_feedback_db_path_is_under_runtime_feedback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert (
        runtime_paths.get_feedback_db_path()
        == tmp_path / "runtime" / "feedback" / "tickets.db"
    )


def test_dataset_results_path_is_under_runtime_datasets(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert (
        runtime_paths.get_dataset_results_dir()
        == tmp_path / "runtime" / "datasets" / "results"
    )


def test_runtime_paths_does_not_import_streamlit() -> None:
    assert "streamlit" not in inspect.getsource(runtime_paths)
