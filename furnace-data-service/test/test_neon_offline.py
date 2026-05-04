"""Tests for the shared Neon offline fetcher.

Database I/O is mocked; these tests validate query selection, whitelisting, and
logical report composition without connecting to Neon.
"""

import pandas as pd
import pytest

from furnace_data.neon_db import offline as neon


class FakeEngine:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


def test_neon_table_discovery_includes_migrated_reports():
    listing = neon.list_neon_offline_tables()

    assert "RM_COMPOSITION" in listing["report_map"]
    assert "BURDEN_DISTRIBUTION" in listing["report_map"]
    assert "HOPPER_MANAGEMENT" in listing["report_map"]
    assert "ore_chemistry" in listing["report_map"]["RM_COMPOSITION"]
    assert "burden_distribution_history" in listing["tables"]
    assert "hopper_material_history" in listing["tables"]


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

    monkeypatch.setattr(neon, "build_relational_engine", fake_build_engine)
    monkeypatch.setattr(neon.pd, "read_sql_query", fake_read_sql_query)

    df = neon.fetch_offline_data(
        table_name="charge_data",
        time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        columns=["date_time", "ore_1_mt"],
        database_url="postgresql://unit-test",
    )

    assert calls["database_url"] == "postgresql://unit-test"
    assert calls["engine"] is fake_engine
    assert '"charge_data"' in calls["query"]
    assert fake_engine.disposed is True
    assert df.index.name == "time"
    assert str(df.index.tz) == "UTC"


def test_combined_rm_report_concats_source_tables(monkeypatch):
    def fake_fetch_table(table_name, time_range, query_type="ts", window=None, database_url=None, columns=None):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")], name="time")
        return pd.DataFrame({"value": [table_name]}, index=idx)

    monkeypatch.setattr(neon, "fetch_offline_data", fake_fetch_table)

    df = neon.fetch_offline_report(
        report_type="RM_COMPOSITION",
        time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
    )

    assert set(df["source_table"]) == set(neon.NEON_OFFLINE_REPORT_MAP["RM_COMPOSITION"])
    assert len(df) == len(neon.NEON_OFFLINE_REPORT_MAP["RM_COMPOSITION"])


def test_burden_history_rejects_aggregation():
    with pytest.raises(ValueError, match="does not support SQL aggregation"):
        neon.fetch_offline_data(
            table_name="burden_distribution_history",
            time_range=("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
            query_type="average",
        )
