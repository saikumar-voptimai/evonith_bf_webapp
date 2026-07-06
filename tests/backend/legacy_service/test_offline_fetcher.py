"""Tests for the shared PostgreSQL offline fetcher.

Database I/O is mocked; these tests validate query selection, whitelisting, and
logical report composition without connecting to PostgreSQL.
"""

import pandas as pd
import pytest

import furnace_data.offline as offline


class FakeEngine:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


def test_offline_table_discovery_includes_migrated_reports():
    listing = offline.list_offline_tables()

    assert "RM_COMPOSITION" in listing["report_map"]
    assert listing["report_map"]["RAW_MATERIAL_STRENGTH"] == [
        "offline_feed.raw_material_strength_analysis"
    ]
    assert "BURDEN_DISTRIBUTION" in listing["report_map"]
    assert "HOPPER_MANAGEMENT" in listing["report_map"]
    assert "offline_feed.ore_chemistry" in listing["report_map"]["RM_COMPOSITION"]
    assert "ops_config.burden_history" in listing["tables"]
    assert "ops_config.hopper_raw_material_history" in listing["tables"]
    assert listing["aliases"]["rm_hm"] == "offline_feed.raw_material_strength_analysis"
    assert "ore_6_mt" in listing["tables"]["offline_feed.charge_data"]["columns"]
    strength_columns = listing["tables"]["offline_feed.raw_material_strength_analysis"]["columns"]
    assert "property_1" in strength_columns
    assert "property_1_name" in strength_columns
    assert "material_code" in strength_columns
    assert "material_name" in strength_columns
    assert "ai" not in strength_columns
    assert "rdi" not in strength_columns
    assert "ri" not in strength_columns
    assert "ti" not in strength_columns
    assert "plant_master.material_property_mapping" in listing["tables"]


def test_fetch_table_uses_database_url_and_sets_time_index(monkeypatch):
    calls = {}
    fake_engine = FakeEngine()

    def fake_build_engine(database_url=None):
        calls["database_url"] = database_url
        return fake_engine

    def fake_read_sql_query(query, engine, params):
        calls["query"] = str(query)
        calls["engine"] = engine
        calls["params"] = params
        return pd.DataFrame(
            {
                "date_time": ["2026-01-01T00:00:00Z"],
                "ore_1_mt": [10.0],
            }
        )

    monkeypatch.setattr(offline, "build_relational_engine", fake_build_engine)
    monkeypatch.setattr(offline.pd, "read_sql_query", fake_read_sql_query)

    df = offline.fetch_offline_data(
        table_name="charge_data",
        time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        columns=["date_time", "ore_1_mt"],
        database_url="postgresql://unit-test",
    )

    assert calls["database_url"] == "postgresql://unit-test"
    assert calls["engine"] is fake_engine
    assert '"offline_feed"."charge_data"' in calls["query"]
    assert fake_engine.disposed is True
    assert df.index.name == "time"
    assert str(df.index.tz) == "UTC"


def test_strength_fetch_maps_properties_without_selecting_legacy_columns(monkeypatch):
    calls = {}
    fake_engine = FakeEngine()

    def fake_build_engine(database_url=None):
        calls["database_url"] = database_url
        return fake_engine

    def fake_read_sql_query(query, engine, params):
        calls["query"] = str(query)
        calls["params"] = params
        return pd.DataFrame(
            {
                "date_time": ["2026-01-01T00:00:00Z"],
                "material_code": ["sinter_3"],
                "property_1": [5.2],
                "property_2": [79.0],
                "property_3": [32.0],
                "property_4": [68.0],
                "property_1_name": ["AI"],
                "material_name": ["sinter_sp02_online"],
                "material_category": ["SINTER"],
            }
        )

    monkeypatch.setattr(offline, "build_relational_engine", fake_build_engine)
    monkeypatch.setattr(offline.pd, "read_sql_query", fake_read_sql_query)

    df = offline.fetch_offline_data(
        table_name="rm_hm",
        time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        database_url="postgresql://unit-test",
    )

    query = calls["query"]
    assert '"offline_feed"."raw_material_strength_analysis" AS r' in query
    assert '"plant_master"."material_property_mapping" AS m' in query
    assert '"plant_master"."materials" AS mat' in query
    assert '"plant_master"."material_categories" AS cat' in query
    assert '"property_1_name"' in query
    assert 'r."ai"' not in query
    assert 'r."rdi"' not in query
    assert 'r."ri"' not in query
    assert 'r."ti"' not in query
    assert "ai" not in df.columns
    assert "rdi" not in df.columns
    assert "ri" not in df.columns
    assert "ti" not in df.columns
    assert df.iloc[0]["material_category"] == "SINTER"


def test_combined_rm_report_concats_source_tables(monkeypatch):
    def fake_fetch_table(table_name, time_range, query_type="ts", window=None, database_url=None, columns=None):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")], name="time")
        return pd.DataFrame({"value": [table_name]}, index=idx)

    monkeypatch.setattr(offline, "fetch_offline_data", fake_fetch_table)

    df = offline.fetch_offline_report(
        report_type="RM_COMPOSITION",
        time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
    )

    assert set(df["source_table"]) == set(offline.OFFLINE_REPORT_MAP["RM_COMPOSITION"])
    assert len(df) == len(offline.OFFLINE_REPORT_MAP["RM_COMPOSITION"])


def test_report_with_mixed_time_and_master_tables_does_not_sort_mixed_index(monkeypatch):
    def fake_fetch_table(table_name, time_range, query_type="ts", window=None, database_url=None, columns=None):
        if table_name == "plant_master.materials":
            return pd.DataFrame({"material_code": ["ore_1"]})
        idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")], name="time")
        return pd.DataFrame({"value": [table_name]}, index=idx)

    monkeypatch.setattr(offline, "fetch_offline_data", fake_fetch_table)

    df = offline.fetch_offline_report(
        report_type="RM_COMPOSITION",
        time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
    )

    assert "plant_master.materials" in set(df["source_table"])
    assert len(df) == len(offline.OFFLINE_REPORT_MAP["RM_COMPOSITION"])


def test_burden_history_rejects_aggregation():
    with pytest.raises(ValueError, match="does not support SQL aggregation"):
        offline.fetch_offline_data(
            table_name="burden_distribution_history",
            time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
            query_type="average",
        )


def test_legacy_table_alias_resolves_to_schema_qualified_name():
    assert (
        offline.resolve_offline_table_name("hot_metal_chemistry")
        == "offline_feed.hot_metal_slag_analysis"
    )
    assert offline.resolve_offline_table_name("rm_hm") == (
        "offline_feed.raw_material_strength_analysis"
    )
