from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from agents.furnacemind.tools.artifact_store import (
    InMemoryArtifactStore,
    set_artifact_store,
)
from agents.furnacemind.tools import data_tool_adapters
from agents.furnacemind.tools.memory_tool_adapters import (
    configure_memory_stores,
    search_knowledge_docs,
    search_shift_history,
)
from agents.furnacemind.tools.plotting_tool_adapters import execute_python_plot
from agents.furnacemind.tools.plotting_sandbox import safe_exec
from agents.furnacemind.tools.registry import (
    configure_artifact_store,
    execute_openai_tool_call,
    get_openai_tool_schemas,
)
from furnace_data.services import data_fetch_service, ml_dataset_service, ml_service


ROOT = Path(__file__).resolve().parents[1]


def _imports_for(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_backend_services_do_not_import_streamlit() -> None:
    offenders = []
    for path in (ROOT / "furnace_data" / "furnace_data" / "services").glob("*.py"):
        imports = _imports_for(path)
        if "streamlit" in imports or any(name.startswith("streamlit.") for name in imports):
            offenders.append(path)

    assert offenders == []


def test_agent_tools_do_not_import_streamlit_or_ui_artifacts() -> None:
    offenders = []
    banned_exact = {"streamlit", "data.ml.static_csv"}
    banned_prefixes = ("streamlit.", "ui.furnacemind", "data.ml.static_csv.")

    for path in (ROOT / "src" / "agents" / "furnacemind" / "tools").glob("*.py"):
        imports = _imports_for(path)
        if imports & banned_exact or any(
            name.startswith(prefix) for name in imports for prefix in banned_prefixes
        ):
            offenders.append(path)

    assert offenders == []


def test_in_memory_artifact_store_round_trips_dataset_and_figure() -> None:
    store = InMemoryArtifactStore()
    df = pd.DataFrame(
        {"fuel_rate": [500.0]},
        index=pd.DatetimeIndex(["2026-05-05 06:00:00"], name="time"),
    )

    dataset_id = store.new_dataset_id("ml")
    store.save_dataset(dataset_id=dataset_id, df=df, meta={"dataset_id": dataset_id})
    store.set_ml_cache(df)
    store.save_figure({"figure": True}, "fig = {}")
    store.append_plot_error("boom")

    assert store.get_dataset(dataset_id)["df"].equals(df)
    assert store.get_active_df().equals(df)
    assert store.get_ml_cache().equals(df)
    assert store.figure == {"figure": True}
    assert store.last_plot_error == "boom"


def test_fetch_ml_data_uses_pure_loader_and_injected_store(monkeypatch) -> None:
    store = InMemoryArtifactStore()
    set_artifact_store(store)

    df = pd.DataFrame(
        {"ACT. FUEL RATEKG/THM.": [500.0, 505.0]},
        index=pd.DatetimeIndex(
            ["2026-05-05 06:00:00", "2026-05-05 07:00:00"],
            name="time",
        ),
    )
    monkeypatch.setattr(ml_dataset_service, "load_static_dataset", lambda: df)

    result = data_tool_adapters.fetch_ml_data(
        start_time="2026-05-05 06:00:00",
        end_time="2026-05-05 07:00:00",
    )

    assert "ML STATIC DATA" in result
    assert store.get_ml_cache().equals(df)
    assert store.get_active_df() is not None
    assert store.get_active_df().equals(df)


def test_registry_dispatch_uses_injected_artifact_store(monkeypatch) -> None:
    store = InMemoryArtifactStore()
    configure_artifact_store(store)

    df = pd.DataFrame(
        {"ACT. FUEL RATEKG/THM.": [500.0]},
        index=pd.DatetimeIndex(["2026-05-05 06:00:00"], name="time"),
    )
    monkeypatch.setattr(ml_dataset_service, "load_static_dataset", lambda: df)

    result = execute_openai_tool_call(
        name="fetch_ml_data",
        arguments={
            "start_time": "2026-05-05 06:00:00",
            "end_time": "2026-05-05 07:00:00",
        },
    )

    assert "ML STATIC DATA" in result
    assert store.get_active_df() is not None


def test_plotting_adapter_stores_figure_through_artifact_interface() -> None:
    store = InMemoryArtifactStore()
    set_artifact_store(store)
    df = pd.DataFrame(
        {"fuel_rate": [500.0, 505.0]},
        index=pd.DatetimeIndex(["2026-05-05 06:00:00", "2026-05-05 07:00:00"]),
    )
    store.save_dataset(dataset_id="active", df=df, meta={"dataset_id": "active"})

    result = execute_python_plot("fig = px.line(df, y='fuel_rate')")

    assert result == "Successfully generated Plotly figure."
    assert store.figure is not None
    assert store.last_plot_code == "fig = px.line(df, y='fuel_rate')"


def test_plotting_sandbox_blocks_dunder_escape() -> None:
    try:
        safe_exec("x = ().__class__", {})
    except ValueError as exc:
        assert "Disallowed token" in str(exc)
    else:
        raise AssertionError("dunder traversal should be blocked")


def test_memory_tools_use_configured_stores() -> None:
    class ShiftStore:
        def search_similar_windows(self, *, query_text: str, top_k: int):
            assert query_text == "bad permeability"
            assert top_k == 5
            return [{"payload": {"window_id": "2026-05-05-A", "summary_text": "Stable."}}]

    class KnowledgeStore:
        def search(self, query: str, top_k: int):
            assert query == "slag basicity"
            assert top_k == 5
            return [{"payload": {"source": "sop.pdf", "content": "Keep it steady."}}]

    configure_memory_stores(shift_store=ShiftStore(), knowledge_store=KnowledgeStore())

    assert "2026-05-05-A" in search_shift_history("bad permeability")
    assert "sop.pdf" in search_knowledge_docs("slag basicity")


def test_memory_tools_report_unconfigured_stores() -> None:
    configure_memory_stores(shift_store=None, knowledge_store=None)

    assert search_shift_history("anything") == "Shift store not initialized."
    assert search_knowledge_docs("anything") == "Knowledge store not initialized."


def test_tool_schemas_forbid_extra_arguments() -> None:
    tools = get_openai_tool_schemas()
    assert tools
    for tool in tools:
        params = tool["function"]["parameters"]
        assert params["additionalProperties"] is False


def test_static_dataset_service_filters_and_renames_database_columns(monkeypatch) -> None:
    def config(_: str) -> dict:
        return {
            "DATA": "src/assets/data/furnace_dataset.csv",
            "ml_dataset": {"local_tz": "Asia/Kolkata"},
            "rename_dict": {
                "pellet_sio2_pct": "PELLET_PCT_SIO2",
                "weighted_coke_angle": "WEIGHTED_COKE_ANGLE",
            },
            "cleaning": {
                "column_groups": {
                    "rm_params": ["pellet_sio2_pct"],
                    "bd_params": ["weighted_coke_angle", "coke__p01_angles"],
                },
                "extra_keep_columns": {"alias_keys": ["missing_alias"]},
            },
        }

    calls: list[dict] = []

    def fake_fetch(table_name, time_range, **kwargs):
        calls.append({"table_name": table_name, "time_range": time_range, **kwargs})
        return pd.DataFrame(
            {"pellet_sio2_pct": [4.2], "weighted_coke_angle": [37.0]},
            index=pd.DatetimeIndex(["2026-05-05T00:00:00Z"], name="time"),
        )

    monkeypatch.setattr(ml_dataset_service, "load_config", config)
    monkeypatch.setattr(
        ml_dataset_service,
        "available_static_dataset_columns",
        lambda: {"pellet_sio2_pct", "weighted_coke_angle", "coke__p01_angles"},
    )
    monkeypatch.setattr(ml_dataset_service, "fetch_offline_data", fake_fetch)

    assert ml_dataset_service.static_dataset_fetch_columns() == [
        "pellet_sio2_pct",
        "weighted_coke_angle",
    ]

    df = ml_dataset_service.fetch_static_dataset_from_database()

    assert calls == [
        {
            "table_name": "historical_static_ml_dataset",
            "time_range": "full",
            "query_type": "raw",
            "columns": ["pellet_sio2_pct", "weighted_coke_angle"],
        }
    ]
    assert list(df.columns) == ["PELLET_PCT_SIO2", "WEIGHTED_COKE_ANGLE"]
    assert df.index[0] == pd.Timestamp("2026-05-05 05:30:00")


def test_static_dataset_service_loads_local_csv_without_database(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "furnace_dataset.csv"
    pd.DataFrame(
        {"pellet_sio2_pct": [4.2]},
        index=pd.DatetimeIndex(["2026-05-05 05:30:00"], name="time"),
    ).to_csv(csv_path)

    monkeypatch.setattr(
        ml_dataset_service,
        "load_config",
        lambda _: {
            "DATA": "src/assets/data/furnace_dataset.csv",
            "ml_dataset": {"local_tz": "Asia/Kolkata"},
            "rename_dict": {"pellet_sio2_pct": "PELLET_PCT_SIO2"},
        },
    )
    monkeypatch.setattr(
        ml_dataset_service,
        "fetch_static_dataset_from_database",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("database hit")),
    )

    df = ml_dataset_service.load_static_dataset(csv_path)

    assert list(df.columns) == ["PELLET_PCT_SIO2"]
    assert df.index[0] == pd.Timestamp("2026-05-05 05:30:00")


def test_data_fetch_service_time_window_and_concat_helpers() -> None:
    assert data_fetch_service.normalize_time_range("8h") == "last 8 hours"
    assert data_fetch_service.normalize_time_range("last 2 days") == "last 2 days"
    assert data_fetch_service.resolve_online_window(
        lookback=pd.Timedelta(days=2).to_pytimedelta(),
        window=None,
    ) == "1 hour"
    assert data_fetch_service.resolve_online_window(
        lookback=pd.Timedelta(hours=8).to_pytimedelta(),
        window=None,
    ) == "15 minutes"

    static_df = pd.DataFrame(
        {"value": [1.0]},
        index=pd.DatetimeIndex(["2026-05-05 06:00:00"]),
    )
    online_df = pd.DataFrame(
        {"value": [2.0]},
        index=pd.DatetimeIndex(["2026-05-05T00:30:00Z"]),
    )

    combined = data_fetch_service.concat_dfs([static_df, online_df])

    assert str(combined.index.tz) == "UTC"
    assert combined.iloc[0]["value"] == 2.0


def test_ml_service_shift_slice_and_group_summary() -> None:
    start, end = ml_service.shift_window("2026-05-05", "C")
    assert start == pd.Timestamp("2026-05-05 22:00:00")
    assert end == pd.Timestamp("2026-05-06 06:00:00")

    df = pd.DataFrame(
        {
            "ACT. FUEL RATEKG/THM.": [500.0, 510.0],
            "PELLET_PCT_SIO2": [4.2, 4.4],
        },
        index=pd.DatetimeIndex(["2026-05-05 06:00:00", "2026-05-05 07:00:00"]),
    )

    sliced = ml_service.slice_ml_df(
        df,
        pd.Timestamp("2026-05-05 06:00:00"),
        pd.Timestamp("2026-05-05 07:00:00"),
        columns=["fuel"],
    )

    assert list(sliced.columns) == ["ACT. FUEL RATEKG/THM."]
    assert "KPIs" in ml_service.ml_column_summary(df)
