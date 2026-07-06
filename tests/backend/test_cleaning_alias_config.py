from __future__ import annotations

import pytest
import pandas as pd

import furnace_data.dataset.cleaning as cleaning


def test_default_cleaning_config_resolves_key_columns_from_alias_config() -> None:
    cfg = cleaning.build_default_config()

    assert "SINTER_SP_01_CALC_MT" in cfg.zero_fill_columns
    assert "SINTER_CALC_MT" in cfg.zero_fill_columns
    assert "STEAMKGS/HR." in cfg.zero_fill_columns
    assert "FLUX_CALC_MT" in cfg.zero_fill_columns
    assert "FLUX_1_CALC_MT" in cfg.zero_fill_columns
    assert "FLUX_3_CALC_MT" in cfg.zero_fill_columns
    assert "ORE_1_CALC_MT" in cfg.zero_fill_columns
    assert "ORE_12_CALC_MT" in cfg.zero_fill_columns
    assert "FLUX_SIO2%" in cfg.zero_fill_columns
    assert "PELLET_PCT_AL2O3" in cfg.zero_fill_columns
    assert "PELLET_PCT_SIO2" in cfg.zero_fill_columns
    assert "FLUX_1_CALC_MT" in cfg.columns.rm_params
    assert "FLUX_3_CALC_MT" in cfg.columns.rm_params
    assert "ORE_1_CALC_MT" in cfg.columns.rm_params
    assert "ORE_12_CALC_MT" in cfg.columns.rm_params
    assert cfg.col_max_nan_fraction == pytest.approx(0.70)
    assert cfg.sinter_combined_col == "SINTER_CALC_MT"


def test_default_cleaning_config_uses_mapping_names_not_parameter_aliases() -> None:
    full_config = cleaning.load_config("setting_ds_dv.yml")

    assert "coke_vm_pct" in full_config["cleaning"]["column_groups"]["rm_params"]
    assert "pellet_sio2_pct" in full_config["cleaning"]["column_groups"]["rm_params"]
    assert "pellet_pct_sio2" not in full_config["cleaning"]["column_groups"]["rm_params"]
    assert "COKE_VM%" not in full_config["cleaning"]["column_groups"]["rm_params"]
    assert "hot_blast_volume_nm3hr" in {
        row["alias_key"] for row in full_config["cleaning"]["cruising_filters"]
    }
    assert full_config["ml_dataset"]["online_params"]["body_dp_top"] == "top_bar"


def test_cleaner_drops_only_non_protected_columns_above_nan_threshold() -> None:
    cfg = cleaning.CleaningConfig(
        columns=cleaning.ColumnGroups(
            rm_params=(),
            hm_slag_params=(),
            bd_params=(),
            temp_params=(),
            op_params=(),
            prcs_params=(),
            proxy_params=(),
            extra_keep_columns=("DROP_71", "KEEP_69", "SPARSE_ZERO"),
        ),
        row_min_non_na_fraction=0.0,
        col_max_nan_fraction=0.70,
        zero_fill_columns=("SPARSE_ZERO",),
        add_unit_cost_feature=False,
    )
    df = pd.DataFrame(
        {
            "DROP_71": [1.0, 2.0] + [None] * 8,
            "KEEP_69": [1.0, 2.0, 3.0, 4.0] + [None] * 6,
            "SPARSE_ZERO": [None] * 10,
        },
        index=pd.date_range("2026-05-01", periods=10, freq="h"),
    )

    cleaned = cleaning.DataCleaner(cfg).clean(df)

    assert "DROP_71" not in cleaned.columns
    assert "KEEP_69" in cleaned.columns
    assert "SPARSE_ZERO" in cleaned.columns
    assert cleaned["SPARSE_ZERO"].eq(0).all()


def test_cleaner_preserves_high_nan_imputation_skip_columns() -> None:
    cfg = cleaning.CleaningConfig(
        columns=cleaning.ColumnGroups(
            rm_params=(),
            hm_slag_params=(),
            bd_params=(),
            temp_params=(),
            op_params=(),
            prcs_params=(),
            proxy_params=(),
            extra_keep_columns=("SPARSE_IMPUTE",),
        ),
        row_min_non_na_fraction=0.0,
        col_max_nan_fraction=0.70,
        add_unit_cost_feature=False,
        imputation_plan=cleaning.ImputationPlan(skip_columns=("SPARSE_IMPUTE",)),
    )
    df = pd.DataFrame(
        {"SPARSE_IMPUTE": [1.0] + [None] * 9},
        index=pd.date_range("2026-05-01", periods=10, freq="h"),
    )

    cleaned = cleaning.DataCleaner(cfg).clean(df)

    assert "SPARSE_IMPUTE" in cleaned.columns
    assert cleaned["SPARSE_IMPUTE"].notna().all()


def test_cleaner_keeps_absent_configured_columns_absent(caplog) -> None:
    cfg = cleaning.CleaningConfig(
        columns=cleaning.ColumnGroups(
            rm_params=(),
            hm_slag_params=(),
            bd_params=(),
            temp_params=(),
            op_params=(),
            prcs_params=(),
            proxy_params=(),
            extra_keep_columns=("PRESENT", "MISSING"),
        ),
        row_min_non_na_fraction=0.0,
        add_unit_cost_feature=False,
    )
    df = pd.DataFrame(
        {"PRESENT": [1.0]},
        index=pd.DatetimeIndex(["2026-05-01"], name="time"),
    )

    cleaned = cleaning.DataCleaner(cfg).clean(df)

    assert "PRESENT" in cleaned.columns
    assert "MISSING" not in cleaned.columns
    assert "configured columns missing" in caplog.text


def test_tonnage_caps_keep_missing_values_for_model_training_cleanup() -> None:
    cfg = cleaning.CleaningConfig(
        columns=cleaning.ColumnGroups(
            rm_params=(),
            hm_slag_params=(),
            bd_params=(),
            temp_params=(),
            op_params=(),
            prcs_params=(),
            proxy_params=(),
            extra_keep_columns=("COKE_CALC_MT",),
        ),
        row_min_non_na_fraction=0.0,
        col_max_nan_fraction=1.0,
        add_unit_cost_feature=False,
        tonnage_caps={"COKE_CALC_MT": 55.0},
    )
    df = pd.DataFrame(
        {"COKE_CALC_MT": [None, 54.0, 55.0]},
        index=pd.date_range("2026-05-01", periods=3, freq="h"),
    )

    cleaned = cleaning.DataCleaner(cfg).clean(df)

    assert len(cleaned) == 2
    assert cleaned.index.tolist() == [df.index[0], df.index[1]]


def test_default_cleaning_config_fails_for_missing_mapping_name(monkeypatch) -> None:
    monkeypatch.setattr(
        cleaning,
        "load_config",
        lambda _: {
            "rename_dict": {},
            "cleaning": {
                "column_groups": {"rm_params": ["missing_cleaning_mapping"]},
                "sinter": {
                    "sp01_calc_alias_key": "sinter01_mt",
                    "sp02_alias_key": "sinter_mt",
                    "combined_alias_key": "sinter_mt",
                },
                "unit_cost": {
                    "output_alias_key": "unit_cost_lakhs_thm",
                    "coke_rate_alias_key": "coke_rate",
                    "pci_rate_alias_key": "actual_kg_thm",
                },
            },
        },
    )

    with pytest.raises(KeyError, match="missing_cleaning_mapping"):
        cleaning.build_default_config()
