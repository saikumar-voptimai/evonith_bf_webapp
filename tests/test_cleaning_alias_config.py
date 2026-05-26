from __future__ import annotations

import pytest

import furnace_data.dataset.cleaning as cleaning


def test_default_cleaning_config_resolves_key_columns_from_alias_config() -> None:
    cfg = cleaning.build_default_config()

    assert "SINTER_SP_01_CALC_MT" in cfg.zero_fill_columns
    assert "SINTER_CALC_MT" in cfg.zero_fill_columns
    assert "STEAMKGS/HR." in cfg.zero_fill_columns
    assert "FLUX_CALC_MT" in cfg.zero_fill_columns
    assert "FLUX_SIO2%" in cfg.zero_fill_columns
    assert "PELLET_PCT_AL2O3" in cfg.zero_fill_columns
    assert "PELLET_PCT_SIO2" in cfg.zero_fill_columns
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
