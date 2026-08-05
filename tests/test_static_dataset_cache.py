from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

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
            "coke_cri": "COKE_CRI",
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
        if table_name == "raw_material_strength_analysis":
            return pd.DataFrame(
                {
                    "material_code": ["coke_1"],
                    "property_1": [82.0],
                    "property_2": [6.0],
                    "property_3": [24.0],
                    "property_4": [65.0],
                    "property_1_name": ["M-40"],
                    "property_2_name": ["M-10"],
                    "property_3_name": ["CRI"],
                    "property_4_name": ["CSR"],
                },
                index=pd.DatetimeIndex(["2026-05-04T18:30:00Z"], name="time"),
            )
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
    monkeypatch.setattr(manager_module, "load_config", _config)
    monkeypatch.setattr(
        StaticDatasetManager,
        "_current_local_hour",
        staticmethod(lambda: pd.Timestamp("2026-05-05 23:00:00")),
    )
    static_csv.load_static_dataset.clear()

    csv_path = tmp_path / "missing.csv"
    df = static_csv.load_static_dataset(csv_path)

    assert calls == ["historical_static_ml_dataset", "raw_material_strength_analysis"]
    assert selected_columns == ["pellet_sio2_pct", "weighted_coke_angle"]
    assert "coke__p01_angles" not in selected_columns
    assert list(df.columns) == ["PELLET_PCT_SIO2", "WEIGHTED_COKE_ANGLE", "COKE_CRI"]
    assert float(df.iloc[0]["COKE_CRI"]) == 24.0
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


def test_static_dataset_manager_clips_future_hours_before_save(
    monkeypatch,
    tmp_path,
) -> None:
    csv_path = tmp_path / "furnace_dataset.csv"
    manager = StaticDatasetManager(csv_path)
    monkeypatch.setattr(static_csv.load_static_dataset, "clear", lambda: None)
    monkeypatch.setattr(
        StaticDatasetManager,
        "_current_local_hour",
        staticmethod(lambda: pd.Timestamp("2026-05-05 04:00:00")),
    )

    df = pd.DataFrame(
        {"TOPBAR": [0.30, 0.31]},
        index=pd.DatetimeIndex(
            ["2026-05-05 04:00:00", "2026-05-05 05:00:00"],
            name="time",
        ),
    )

    saved_path = manager.save(df)
    saved = pd.read_csv(saved_path, index_col=0, parse_dates=True)
    meta = json.loads((tmp_path / "cache_meta.json").read_text(encoding="utf-8"))

    assert saved.index.max() == pd.Timestamp("2026-05-05 04:00:00")
    assert meta["rows"] == 1


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


def test_static_dataset_manager_ignores_db_rows_after_cutoff(
    monkeypatch,
    tmp_path,
) -> None:
    manager = StaticDatasetManager(tmp_path / "furnace_dataset.csv")
    cutoff = date(2026, 5, 3)
    calls: list[tuple[date, date, str]] = []

    base = pd.DataFrame(
        {"TOPBAR": [0.30, 9.99]},
        index=pd.DatetimeIndex(
            [pd.Timestamp(cutoff), pd.Timestamp(cutoff + timedelta(days=1))],
            name="time",
        ),
    )
    delta = pd.DataFrame(
        {"TOPBAR": [0.31]},
        index=pd.DatetimeIndex([pd.Timestamp(cutoff + timedelta(days=1))], name="time"),
    )

    monkeypatch.setattr(manager_module, "fetch_static_dataset_from_database", lambda: base)
    monkeypatch.setattr(StaticDatasetManager, "_clean_dataset", lambda self, df: df)
    monkeypatch.setattr(
        manager_module,
        "load_config",
        lambda _: {
            "ml_dataset": {
                "local_tz": "Asia/Kolkata",
                "cutoff_date": cutoff.isoformat(),
            }
        },
    )
    monkeypatch.setattr(
        StaticDatasetManager,
        "_current_local_hour",
        staticmethod(lambda: pd.Timestamp(cutoff + timedelta(days=2))),
    )

    def fake_delta(start: date, end: date, rm_mode: str) -> pd.DataFrame:
        calls.append((start, end, rm_mode))
        return delta

    monkeypatch.setattr(manager, "_fetch_and_clean_delta", fake_delta)

    combined = manager.update_static()

    assert calls == [(cutoff + timedelta(days=1), cutoff + timedelta(days=2), "charge")]
    assert combined.loc[pd.Timestamp(cutoff), "TOPBAR"] == 0.30
    assert combined.loc[pd.Timestamp(cutoff + timedelta(days=1)), "TOPBAR"] == 0.31


def test_static_dataset_delta_resample_sums_material_quantities_hourly() -> None:
    raw = pd.DataFrame(
        {
            "ORE_1_CALC_MT": [4.0, 6.0, 5.0],
            "FLUX_1_CALC_MT": [1.0, 2.0, 3.0],
            "ORE_SIO2%": [3.0, 5.0, 7.0],
        },
        index=pd.DatetimeIndex(
            [
                "2026-05-05 00:05:00",
                "2026-05-05 00:45:00",
                "2026-05-05 01:10:00",
            ],
            name="time",
        ),
    )

    hourly = StaticDatasetManager._resample_local_delta_hourly(raw)

    assert hourly.loc[pd.Timestamp("2026-05-05 00:00:00"), "ORE_1_CALC_MT"] == 10.0
    assert hourly.loc[pd.Timestamp("2026-05-05 00:00:00"), "FLUX_1_CALC_MT"] == 3.0
    assert hourly.loc[pd.Timestamp("2026-05-05 00:00:00"), "ORE_SIO2%"] == 4.0
    assert hourly.loc[pd.Timestamp("2026-05-05 01:00:00"), "ORE_1_CALC_MT"] == 5.0


def test_static_dataset_coke_cri_is_effective_until_next_lab_result() -> None:
    df = pd.DataFrame(
        {"TOPBAR": [0.30, 0.31, 0.32, 0.33]},
        index=pd.date_range("2026-05-05 00:00:00", periods=4, freq="1h", name="time"),
    )
    samples = pd.Series(
        [22.0, 24.0],
        index=pd.DatetimeIndex(
            ["2026-05-05 00:30:00", "2026-05-05 02:30:00"],
            name="time",
        ),
    )

    enriched = static_csv.merge_coke_cri_samples(df, samples)

    assert enriched["COKE_CRI"].tolist() == [22.0, 22.0, 22.0, 24.0]
    assert enriched["COKE_CRI"].notna().all()


def test_static_dataset_delta_fills_missing_pci_quantity_rowwise(
    monkeypatch,
    tmp_path,
) -> None:
    manager = StaticDatasetManager(tmp_path / "furnace_dataset.csv")
    raw = pd.DataFrame(
        {
            "PCI_CALC_MT": [1.0, None],
            "PCI_KG/THM": [100.0, 200.0],
            "PRODUCTIONTONNESPERHR": [10.0, 10.0],
        },
        index=pd.DatetimeIndex(
            ["2026-05-05 00:00:00", "2026-05-05 01:00:00"],
            name="time",
        ),
    )

    class FakeFetcher:
        def build_local_delta(self, *args, **kwargs):
            return raw

    class IdentityCleaner:
        def __init__(self, config):
            self.config = config

        def clean(self, df):
            return df

    monkeypatch.setattr(
        "furnace_data.dataset.fetcher.DatasetFetcher",
        lambda: FakeFetcher(),
    )
    monkeypatch.setattr(manager_module, "build_default_config", lambda: object())
    monkeypatch.setattr(manager_module, "DataCleaner", IdentityCleaner)

    cleaned = manager._fetch_and_clean_delta(
        date(2026, 5, 5),
        date(2026, 5, 5),
        "charge",
    )

    assert cleaned.loc[pd.Timestamp("2026-05-05 00:00:00"), "PCI_CALC_MT"] == 1.0
    assert cleaned.loc[pd.Timestamp("2026-05-05 01:00:00"), "PCI_CALC_MT"] == 2.0


def test_static_dataset_manager_does_not_backfill_new_quantity_columns(
    monkeypatch,
    tmp_path,
) -> None:
    csv_path = tmp_path / "furnace_dataset.csv"
    manager = StaticDatasetManager(csv_path)
    base_day = date.today() - timedelta(days=2)
    delta_day = base_day + timedelta(days=1)

    base = pd.DataFrame(
        {"ORE_CALC_MT": [40.0], "ORE_SIO2%": [3.0]},
        index=pd.DatetimeIndex([pd.Timestamp(base_day)], name="time"),
    )
    delta = pd.DataFrame(
        {"ORE_CALC_MT": [45.0], "ORE_1_CALC_MT": [12.0], "ORE_SIO2%": [4.0]},
        index=pd.DatetimeIndex([pd.Timestamp(delta_day)], name="time"),
    )

    monkeypatch.setattr(manager_module, "fetch_static_dataset_from_database", lambda: base)
    monkeypatch.setattr(StaticDatasetManager, "_clean_dataset", lambda self, df: df)
    monkeypatch.setattr(manager, "_fetch_and_clean_delta", lambda *args, **kwargs: delta)

    combined = manager.update_static()

    assert pd.isna(combined.loc[pd.Timestamp(base_day), "ORE_1_CALC_MT"])
    assert combined.loc[pd.Timestamp(delta_day), "ORE_1_CALC_MT"] == 12.0


def test_static_dataset_manager_repairs_material_quantity_totals() -> None:
    df = pd.DataFrame(
        {
            "ORE_CALC_MT": [53.772],
            **{f"ORE_{i}_CALC_MT": [0.0] for i in range(1, 13)},
            "FLUX_CALC_MT": [8.0],
            "FLUX_1_CALC_MT": [1.0],
            "FLUX_2_CALC_MT": [2.0],
            "FLUX_3_CALC_MT": [3.0],
        }
    )

    repaired = StaticDatasetManager._repair_material_quantity_totals(df)

    assert repaired.loc[0, "ORE_CALC_MT"] == 0.0
    assert repaired.loc[0, "FLUX_CALC_MT"] == 6.0


def test_static_dataset_manager_raises_when_required_delta_is_empty(
    monkeypatch,
    tmp_path,
) -> None:
    manager = StaticDatasetManager(tmp_path / "furnace_dataset.csv")
    base_day = date(2026, 5, 3)

    base = pd.DataFrame(
        {"TOPBAR": [0.30]},
        index=pd.DatetimeIndex([pd.Timestamp(base_day)], name="time"),
    )

    monkeypatch.setattr(manager_module, "fetch_static_dataset_from_database", lambda: base)
    monkeypatch.setattr(StaticDatasetManager, "_clean_dataset", lambda self, df: df)
    monkeypatch.setattr(
        StaticDatasetManager,
        "_current_local_hour",
        staticmethod(lambda: pd.Timestamp("2026-05-05 00:00:00")),
    )
    monkeypatch.setattr(
        manager,
        "_fetch_and_clean_delta",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="Delta fetch returned no rows"):
        manager.update_static()


def test_static_dataset_manager_does_not_median_fill_base_only_columns(
    monkeypatch,
    tmp_path,
) -> None:
    manager = StaticDatasetManager(tmp_path / "furnace_dataset.csv")
    base_day = date(2026, 5, 3)
    delta_day = date(2026, 5, 4)

    base = pd.DataFrame(
        {"TOPBAR": [0.30], "ORE_SIO2%": [3.0]},
        index=pd.DatetimeIndex([pd.Timestamp(base_day)], name="time"),
    )
    delta = pd.DataFrame(
        {"ORE_SIO2%": [4.0]},
        index=pd.DatetimeIndex([pd.Timestamp(delta_day)], name="time"),
    )

    monkeypatch.setattr(manager_module, "fetch_static_dataset_from_database", lambda: base)
    monkeypatch.setattr(StaticDatasetManager, "_clean_dataset", lambda self, df: df)
    monkeypatch.setattr(
        StaticDatasetManager,
        "_current_local_hour",
        staticmethod(lambda: pd.Timestamp("2026-05-05 00:00:00")),
    )
    monkeypatch.setattr(manager, "_fetch_and_clean_delta", lambda *args, **kwargs: delta)

    combined = manager.update_static()

    assert combined.loc[pd.Timestamp(base_day), "TOPBAR"] == 0.30
    assert pd.isna(combined.loc[pd.Timestamp(delta_day), "TOPBAR"])
