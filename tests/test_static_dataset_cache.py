from __future__ import annotations

import pandas as pd

from data.ml import static_csv


def _config(_: str) -> dict:
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
                "bd_params": ["weighted_coke_angle"],
            },
            "zero_fill": {"alias_keys": ["pellet_sio2_pct"]},
            "sinter": {},
            "unit_cost": {},
        },
    }


def test_load_static_dataset_prefers_database_when_local_copy_exists(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "furnace_dataset.csv"
    pd.DataFrame(
        {"pellet_sio2_pct": [4.2]},
        index=pd.DatetimeIndex(["2026-05-05 05:30:00"], name="time"),
    ).to_csv(csv_path)

    db_df = pd.DataFrame(
        {"PELLET_PCT_SIO2": [5.1]},
        index=pd.DatetimeIndex(["2026-05-06 05:30:00"], name="time"),
    )

    monkeypatch.setattr(
        static_csv.ml_dataset_service,
        "load_static_dataset",
        lambda *args, **kwargs: db_df,
    )
    static_csv.load_static_dataset.clear()

    df = static_csv.load_static_dataset(csv_path)

    assert list(df.columns) == ["PELLET_PCT_SIO2"]
    assert df.index[0] == pd.Timestamp("2026-05-06 05:30:00")


def test_load_static_dataset_fetches_canonical_table_when_no_copy(
    monkeypatch,
    tmp_path,
) -> None:
    queries: list[str] = []

    class FakeEngine:
        def dispose(self):
            pass

    def fake_read_sql(query, engine):
        queries.append(str(query))
        return pd.DataFrame(
            {
                "time": ["2026-05-05T00:00:00Z"],
                "pellet_sio2_pct": [4.2],
                "weighted_coke_angle": [37.0],
            }
        )

    monkeypatch.setattr(static_csv.ml_dataset_service, "load_config", _config)
    monkeypatch.setattr(static_csv.ml_dataset_service, "build_relational_engine", lambda: FakeEngine())
    monkeypatch.setattr(static_csv.ml_dataset_service.pd, "read_sql_query", fake_read_sql)
    monkeypatch.setattr(
        static_csv.ml_dataset_service,
        "available_static_dataset_columns",
        lambda: {
            "time",
            "pellet_sio2_pct",
            "weighted_coke_angle",
            "coke__p01_angles",
            "coke__p01_rings",
        },
    )
    static_csv.load_static_dataset.clear()

    csv_path = tmp_path / "missing.csv"
    df = static_csv.load_static_dataset(csv_path)

    assert "ml_dataset" in queries[0]
    assert "active_hourly" in queries[0]
    assert '"pellet_sio2_pct"' in queries[0]
    assert "coke__p01_angles" not in queries[0]
    assert list(df.columns) == ["PELLET_PCT_SIO2", "WEIGHTED_COKE_ANGLE"]
    assert df.index[0] == pd.Timestamp("2026-05-05 05:30:00")
    assert not csv_path.exists()
