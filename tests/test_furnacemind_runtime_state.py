"""Tests for FurnaceMind's UI-compatible, run-scoped tool state adapter."""

from __future__ import annotations

from collections import UserDict
from pathlib import Path

import pandas as pd
import pytest

from agents import furnace_tools
from agents.furnacemind import runtime_state
from agents.furnacemind.runtime_state import bind_runtime_state, get_runtime_state


def test_unbound_state_uses_streamlit_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive calls should keep using the UI session-state mapping."""

    ui_state: dict[str, object] = {"surface": "streamlit"}
    monkeypatch.setattr(
        runtime_state,
        "_streamlit_runtime_state",
        lambda: ui_state,
    )

    assert get_runtime_state() is ui_state


def test_furnace_tools_preserve_unbound_ui_state_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unbound tools should read and write the interactive session mapping."""

    ui_state: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_state,
        "_streamlit_runtime_state",
        lambda: ui_state,
    )

    store = furnace_tools._ensure_dataset_store()
    store["ui_dataset"] = {"df": object()}

    assert ui_state["fm_datasets"] is store
    datasets = ui_state["fm_datasets"]
    assert isinstance(datasets, dict)
    assert "ui_dataset" in datasets


def test_bound_state_is_isolated_and_restores_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A headless binding should neither read nor overwrite the UI state."""

    ui_state: dict[str, object] = {"surface": "streamlit"}
    headless_state: dict[str, object] = {"surface": "scheduled-run"}
    monkeypatch.setattr(
        runtime_state,
        "_streamlit_runtime_state",
        lambda: ui_state,
    )

    with bind_runtime_state(headless_state) as bound:
        assert bound is headless_state
        assert get_runtime_state() is headless_state
        get_runtime_state()["result"] = "complete"

    assert headless_state["result"] == "complete"
    assert "result" not in ui_state
    assert get_runtime_state() is ui_state


def test_nested_binding_restores_outer_state_after_exception() -> None:
    """Nested agent calls should restore their caller's run state reliably."""

    outer: dict[str, object] = {"run": "outer"}
    inner: dict[str, object] = {"run": "inner"}

    with bind_runtime_state(outer):
        with pytest.raises(RuntimeError, match="stop inner run"):
            with bind_runtime_state(inner):
                assert get_runtime_state() is inner
                raise RuntimeError("stop inner run")
        assert get_runtime_state() is outer


def test_binding_accepts_any_mutable_mapping() -> None:
    """Callers may provide mapping implementations other than dictionaries."""

    state: UserDict[str, object] = UserDict()

    with bind_runtime_state(state):
        get_runtime_state()["job_id"] = "job-1"

    assert state["job_id"] == "job-1"


def test_binding_rejects_non_mutable_state() -> None:
    """Immutable or non-mapping state should fail before changing the context."""

    with pytest.raises(TypeError, match="mutable mapping"):
        with bind_runtime_state(("not", "mutable")):  # type: ignore[arg-type]
            pytest.fail("invalid state must not enter the context")


def test_furnace_tools_store_data_in_the_bound_run_state() -> None:
    """Dataset and artifact handoff should work without Streamlit state."""

    frame = pd.DataFrame({"body_etaco": [41.5]})
    metadata = {"source": "scheduled-test"}

    with bind_runtime_state({}) as state:
        state["fm_current_artifact_turn_id"] = "run-123"
        dataset_id = furnace_tools._new_dataset_id("online")
        furnace_tools._save_dataset(
            dataset_id=dataset_id,
            df=frame,
            meta=metadata,
        )

        assert state["fm_dataset_counter"] == 1
        assert state["fm_df"] is frame
        assert state["fm_df_meta"] is metadata
        assert state["fm_df_turn_id"] == "run-123"
        assert state["fm_datasets"] == {dataset_id: {"df": frame, "meta": metadata}}


def test_sequential_headless_runs_do_not_share_tool_state() -> None:
    """A new default binding should start with no prior datasets or counters."""

    with bind_runtime_state() as first_state:
        first_id = furnace_tools._new_dataset_id("online")
        furnace_tools._ensure_dataset_store()[first_id] = {"df": object()}

    with bind_runtime_state() as second_state:
        second_id = furnace_tools._new_dataset_id("online")
        assert second_state["fm_dataset_counter"] == 1
        assert furnace_tools._ensure_dataset_store() == {}

    assert first_state is not second_state
    assert first_id in first_state["fm_datasets"]
    assert second_id not in second_state["fm_datasets"]


def test_headless_run_can_disable_source_tree_error_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scheduled tool failures must not write parameters into the source tree."""

    error_path = tmp_path / "tool_errors.md"
    monkeypatch.setattr(furnace_tools, "_TOOL_ERRORS_PATH", error_path)

    with bind_runtime_state({"fm_disable_tool_error_file": True}):
        furnace_tools._append_tool_error(
            tool_name="fetch_online_data",
            params={"token": "must-not-be-written"},
            error="unavailable",
        )

    assert not error_path.exists()
