from __future__ import annotations

from apps.frontend_streamlit.ui.bmo import build_ore_editor_df
from furnace_data.bmo.data import context_provider as context_module
from furnace_data.bmo.data import EvonithBmoContextProvider
from furnace_data.bmo.data.ore_editor_preferences import (
    apply_ore_editor_preferences,
    load_ore_editor_preferences,
)
from furnace_data.config import get_config_path


def test_default_bmo_provider_loads_packaged_config_and_builds_ore_rows(monkeypatch):
    """The Streamlit BMO page uses the provider defaults; they must find packaged config."""

    def fail_fetch(**_kwargs):
        raise RuntimeError("offline source intentionally unavailable")

    monkeypatch.setattr(context_module, "_fetch_offline_data", fail_fetch)

    provider = EvonithBmoContextProvider()

    assert provider.setting_path.name == "setting_bmo.yml"
    assert provider.mapping_path.name == "bmo_ore_mapping.yml"
    assert "packages" in provider.setting_path.parts
    assert "furnace_data" in provider.setting_path.parts
    assert len(provider._ores_cfg) >= 10

    ores, diagnostics = provider.build_ore_inputs()
    preferences = load_ore_editor_preferences(get_config_path("bmo_operator_inputs.yml"))
    editor_df = build_ore_editor_df(ores, default_selected_ids=[])
    editor_df = apply_ore_editor_preferences(editor_df, preferences)

    assert len(ores) == len(provider._ores_cfg)
    assert not editor_df.empty
    assert {
        "ore_id",
        "ore_name",
        "stock_mt",
        "price_rs_per_mt",
        "min_share_pct",
        "max_share_pct",
        "fe_t_pct",
    }.issubset(editor_df.columns)
    assert "sinter_sp_02" in set(editor_df["ore_id"])
    assert editor_df["price_rs_per_mt"].gt(0).any()
    assert editor_df["max_share_pct"].gt(0).any()
    assert any("Stock query failed" in warning for warning in diagnostics["warnings"])
