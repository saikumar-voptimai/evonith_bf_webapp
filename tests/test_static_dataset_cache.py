from __future__ import annotations

import json

import pandas as pd

import data.ml.static_dataset_manager as manager_module
from data.ml import static_csv
from data.ml.static_dataset_manager import StaticDatasetManager


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


def test_load_static_dataset_uses_local_copy_when_present(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "furnace_dataset.csv"
    pd.DataFrame(
        {"pellet_sio2_pct": [4.2]},
        index=pd.DatetimeIndex(["2026-05-05 05:30:00"], name="time"),
    ).to_csv(csv_path)

    def fail_fetch(*args, **kwargs):
        raise AssertionError("database fetch should not be used")

    monkeypatch.setattr(static_csv, "load_config", _config)
    monkeypatch.setattr(static_csv, "fetch_offline_data", fail_fetch)
    static_csv.load_static_dataset.clear()

    df = static_csv.load_static_dataset(csv_path)

    assert list(df.columns) == ["PELLET_PCT_SIO2"]
    assert df.index[0] == pd.Timestamp("2026-05-05 05:30:00")


def test_load_static_dataset_fetches_canonical_table_when_no_copy(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []
    selected_columns: list[str] = []

    def fake_fetch(table_name, *args, **kwargs):
        calls.append(table_name)
        selected_columns.extend(kwargs["columns"])
        return pd.DataFrame(
            {"pellet_sio2_pct": [4.2], "weighted_coke_angle": [37.0]},
            index=pd.DatetimeIndex(["2026-05-05T00:00:00Z"], name="time"),
        )

    class IdentityCleaner:
        def __init__(self, config):
            self.config = config

        def clean(self, df):
            return df

    monkeypatch.setattr(static_csv, "load_config", _config)
    monkeypatch.setattr(static_csv, "fetch_offline_data", fake_fetch)
    monkeypatch.setattr(
        static_csv,
        "_available_static_dataset_columns",
        lambda: {
            "pellet_sio2_pct",
            "weighted_coke_angle",
            "coke__p01_angles",
            "coke__p01_rings",
        },
    )
    monkeypatch.setattr(manager_module, "build_default_config", lambda: object())
    monkeypatch.setattr(manager_module, "DataCleaner", IdentityCleaner)
    static_csv.load_static_dataset.clear()

    csv_path = tmp_path / "missing.csv"
    df = static_csv.load_static_dataset(csv_path)

    assert calls == ["historical_static_ml_dataset"]
    assert selected_columns == ["pellet_sio2_pct", "weighted_coke_angle"]
    assert "coke__p01_angles" not in selected_columns
    assert list(df.columns) == ["PELLET_PCT_SIO2", "WEIGHTED_COKE_ANGLE"]
    assert df.index[0] == pd.Timestamp("2026-05-05 05:30:00")
    assert csv_path.exists()


def test_static_dataset_manager_saves_rotating_full_copy(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "furnace_dataset.csv"
    manager = StaticDatasetManager(csv_path)
    monkeypatch.setattr(static_csv.load_static_dataset, "clear", lambda: None)

    for day in range(1, 5):
        df = pd.DataFrame(
            {"PELLET_PCT_SIO2": [float(day)]},
            index=pd.DatetimeIndex([f"2026-05-0{day} 05:30:00"], name="time"),
        )
        manager.save(df)

    versioned = sorted(tmp_path.glob("furnace_dataset_*.csv"))
    meta = json.loads((tmp_path / "cache_meta.json").read_text(encoding="utf-8"))

    assert csv_path.exists()
    assert len(versioned) == manager._MAX_VERSIONED_FILES
    assert meta["rows"] == 1
    assert meta["csv_file"] == versioned[-1].name


def test_static_dataset_manager_cleans_before_save(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "furnace_dataset.csv"
    manager = StaticDatasetManager(csv_path)

    raw_df = pd.DataFrame(
        {"raw": [1.0]},
        index=pd.DatetimeIndex(["2026-05-05 05:30:00"], name="time"),
    )

    class FakeCleaner:
        def __init__(self, config):
            self.config = config

        def clean(self, df):
            out = df.copy()
            out["cleaned_feature"] = out.pop("raw") + 1
            return out

    monkeypatch.setattr(manager_module, "fetch_static_dataset_from_database", lambda: raw_df)
    monkeypatch.setattr(manager_module, "build_default_config", lambda: object())
    monkeypatch.setattr(manager_module, "DataCleaner", FakeCleaner)
    monkeypatch.setattr(static_csv.load_static_dataset, "clear", lambda: None)

    cleaned = manager.update_static()
    saved_path = manager.save(cleaned)
    saved = pd.read_csv(saved_path)

    assert list(cleaned.columns) == ["cleaned_feature"]
    assert "cleaned_feature" in saved.columns
    assert "raw" not in saved.columns
