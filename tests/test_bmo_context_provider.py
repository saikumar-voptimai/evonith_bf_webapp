from __future__ import annotations

import pandas as pd

from data.bmo import context_provider as context_module
from data.bmo.context_provider import EvonithBmoContextProvider


def _write_bmo_files(tmp_path):
    settings_path = tmp_path / "setting_bmo.yml"
    mapping_path = tmp_path / "bmo_ore_mapping.yml"

    settings_path.write_text(
        """
bmo:
  data_sources:
    stock_measurement: rm_stock
    stock_bucket: bf2_evonith_offline_utc
    stock_time_range: last 1 week
    stock_fallback_reference_qty_mt: 1000.0
    chemistry_measurement: rm_updated_data
    chemistry_bucket: bf2_evonith_offline_utc
    chemistry_time_range_days: 30
  optimization_runtime:
    dataset:
      static_dataset_path: missing.csv
""",
        encoding="utf-8",
    )
    mapping_path.write_text(
        """
ores:
  - id: sinter
    display_name: SINTER
    material_key: sinter_sp_02
    stock_field: sinter_stock
    price_rs_per_mt: 1000.0
    min_share_pct: 58.0
    max_share_pct: 70.0
    fallback_chemistry:
      fe_t_pct: 55.0
      moisture_pct: 8.0
  - id: ore1
    display_name: ORE 1
    material_key: ore_1
    stock_field: ore1_stock
    price_rs_per_mt: 1000.0
    min_share_pct: 0.0
    max_share_pct: 25.0
    fallback_stock_mt: 400.0
    fallback_chemistry:
      fe_t_pct: 61.0
      moisture_pct: 5.0
chemistry_field_map:
  fe_t_pct: fe_t
  moisture_pct: tm
""",
        encoding="utf-8",
    )
    return settings_path, mapping_path


def test_stock_snapshot_uses_planning_fallback_when_live_stock_fails(
    tmp_path, monkeypatch
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fail_fetch(**_kwargs):
        raise RuntimeError("stock source unavailable")

    monkeypatch.setattr(context_module, "_fetch_offline_data_safe", fail_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    stock_map, warnings = provider.get_stock_snapshot()

    assert stock_map["sinter"] == 700.0
    assert stock_map["ore1"] == 400.0
    assert any("planning stock fallback" in warning for warning in warnings)


def test_chemistry_snapshot_uses_explicit_date_range(tmp_path, monkeypatch) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)
    captured = {}

    def fake_fetch(**kwargs):
        captured["time_range"] = kwargs["time_range"]
        return pd.DataFrame()

    monkeypatch.setattr(context_module, "_fetch_offline_data_safe", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    chemistry_map, _warnings = provider.get_chemistry_snapshot(
        mode="latest", window_days=30
    )

    time_range = captured["time_range"]
    assert isinstance(time_range, tuple)
    assert len(time_range) == 2
    assert chemistry_map["sinter"].moisture_pct == 8.0
