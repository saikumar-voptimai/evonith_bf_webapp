"""Tests for the BMO editor->input converters (``ui.bmo.editor_inputs``).

These pure helpers turn Streamlit data-editor dataframes back into the typed
input records consumed by the LP/DE optimizers and the slag balance. They were
extracted from the Blend Optimizer page so the conversion logic can be verified
without a Streamlit runtime.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ui.bmo.editor_inputs import (
    dust_inputs_from_editor,
    float_from_row,
    flux_inputs_from_editor,
    fuel_ash_inputs_from_editor,
    slag_balance_settings_from_editor,
)


class TestFloatFromRow:
    def test_parses_numeric(self):
        row = pd.Series({"a": "3.5"})
        assert float_from_row(row, "a") == 3.5

    def test_blank_and_nan_use_default(self):
        row = pd.Series({"a": float("nan"), "b": None})
        assert float_from_row(row, "a", default=1.0) == 1.0
        assert float_from_row(row, "b", default=2.0) == 2.0

    def test_missing_key_uses_default(self):
        assert float_from_row(pd.Series({"a": 1.0}), "missing", default=7.0) == 7.0

    def test_non_numeric_uses_default(self):
        assert float_from_row(pd.Series({"a": "abc"}), "a", default=0.0) == 0.0


class TestFuelAshInputsFromEditor:
    def test_empty_df_returns_empty_list(self):
        assert fuel_ash_inputs_from_editor(pd.DataFrame()) == []

    def test_builds_typed_inputs_and_skips_blank_ids(self):
        df = pd.DataFrame(
            [
                {
                    "fuel_id": "coke",
                    "fuel_name": "Coke",
                    "enabled": True,
                    "rate_kg_per_thm": 340.0,
                    "moisture_pct": 0.4,
                    "vm_pct": 0.9,
                    "ash_pct": 11.5,
                    "sio2_pct": 55.0,
                },
                {"fuel_id": "", "fuel_name": "blank"},  # skipped
            ]
        )
        out = fuel_ash_inputs_from_editor(df)
        assert len(out) == 1
        coke = out[0]
        assert coke.fuel_id == "coke"
        assert coke.display_name == "Coke"
        assert coke.rate_kg_per_thm == 340.0
        assert coke.moisture_pct == 0.4
        assert coke.vm_pct == 0.9
        assert coke.ash_pct == 11.5
        assert coke.sio2_pct == 55.0
        # Unspecified chemistry columns default to 0.0 via float_from_row.
        assert coke.cao_pct == 0.0


class TestFluxInputsFromEditor:
    def test_maps_flux_name_to_display_name(self):
        df = pd.DataFrame(
            [
                {
                    "flux_id": "limestone",
                    "flux_name": "Limestone",
                    "enabled": True,
                    "wet_qty_mt": 12.0,
                    "cao_pct": 47.7,
                    "loi_pct": 40.8,
                }
            ]
        )
        out = flux_inputs_from_editor(df)
        assert len(out) == 1
        assert out[0].flux_id == "limestone"
        assert out[0].display_name == "Limestone"
        assert out[0].wet_qty_mt == 12.0
        assert out[0].cao_pct == 47.7
        assert out[0].loi_pct == 40.8

    def test_disabled_flag_preserved(self):
        df = pd.DataFrame([{"flux_id": "quartz", "enabled": False, "sio2_pct": 96.5}])
        out = flux_inputs_from_editor(df)
        assert out[0].enabled is False
        assert out[0].display_name == "quartz"  # falls back to id when no name


class TestDustInputsFromEditor:
    def test_builds_dust_inputs(self):
        df = pd.DataFrame(
            [{"dust_id": "bf_gas_dust", "dust_name": "BF Gas Dust", "wet_qty_mt": 5.0}]
        )
        out = dust_inputs_from_editor(df)
        assert len(out) == 1
        assert out[0].dust_id == "bf_gas_dust"
        assert out[0].display_name == "BF Gas Dust"
        assert out[0].wet_qty_mt == 5.0


class TestSlagBalanceSettingsFromEditor:
    def test_pi_chemistry_from_hm_chem_values_and_snapshot(self):
        settings = {"enabled": True, "mn_recovery_pct": 55.0}
        hm_chem = {
            "carbon_pct": 4.2,
            "silicon_pct": 0.7,
            "sulphur_pct": 0.03,
            "other_pct": 0.1,
        }
        snapshot = {"chem_pct_mn": 0.15, "chem_pct_ti": 0.02}
        out = slag_balance_settings_from_editor(settings, hm_chem, snapshot)
        assert out.enabled is True
        assert out.carbon_pct == 4.2
        assert out.silicon_pct == 0.7
        assert out.mn_pct == 0.15
        assert out.ti_pct == 0.02
        assert out.mn_recovery_pct == 55.0

    def test_defaults_when_snapshot_missing(self):
        out = slag_balance_settings_from_editor({}, {}, None)
        assert out.mn_pct == 0.0
        assert out.ti_pct == 0.0
        # Editable factor falls back to its default.
        assert math.isclose(out.si_to_sio2_factor, 2.14)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
