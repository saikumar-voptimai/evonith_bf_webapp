from __future__ import annotations

import os

import pandas as pd
import pytest

from data.bmo import context_provider as context_module
from data.bmo.context_provider import EvonithBmoContextProvider
from utils.bmo.types import OreChemistry, OreInput


def _write_bmo_files(tmp_path):
    settings_path = tmp_path / "setting_bmo.yml"
    mapping_path = tmp_path / "bmo_ore_mapping.yml"

    settings_path.write_text(
        """
bmo:
  data_sources:
    stock_table: offline_feed.raw_material_stock
    stock_time_range: last 1 week
    stock_fallback_reference_qty_mt: 1000.0
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
    material_key: sinter_3
    price_rs_per_mt: 1000.0
    min_share_pct: 58.0
    max_share_pct: 70.0
    fallback_chemistry:
      fe_t_pct: 55.0
      moisture_pct: 8.0
  - id: ore1
    display_name: ORE 1
    material_key: ore_1
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


def test_stock_snapshot_uses_sinter_fallback_and_zero_ore_when_stock_fails(
    tmp_path, monkeypatch
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fail_fetch(**_kwargs):
        raise RuntimeError("stock source unavailable")

    monkeypatch.setattr(context_module, "_fetch_offline_data", fail_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    stock_map, warnings = provider.get_stock_snapshot()

    assert stock_map["sinter"] == 700.0
    assert stock_map["ore1"] == 0.0
    assert any("Sinter stock unavailable" in warning for warning in warnings)
    diagnostics = provider.get_data_diagnostics()["stock"]
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["zero_count"] == 1
    assert {row["source"] for row in diagnostics["material_rows"]} == {
        "sinter_fallback",
        "offline_db_missing_as_zero",
    }


def test_stock_snapshot_reads_latest_raw_material_stock_by_material_code(
    tmp_path, monkeypatch
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fake_fetch(**kwargs):
        assert kwargs["table_name"] == "offline_feed.raw_material_stock"
        assert kwargs["query_type"] == "raw"
        assert kwargs["columns"] == ["date_time", "material_code", "stock_mt"]
        return pd.DataFrame(
            {
                "date_time": pd.to_datetime(
                    ["2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z"]
                ),
                "material_code": ["sinter_3", "ore_1"],
                "stock_mt": [1234.0, 456.0],
            }
        ).set_index("date_time")

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    stock_map, warnings = provider.get_stock_snapshot()

    assert stock_map["sinter"] == 1234.0
    assert stock_map["ore1"] == 456.0
    assert warnings == []
    diagnostics = provider.get_data_diagnostics()["stock"]
    assert diagnostics["returned_rows"] == 2
    assert diagnostics["material_rows"][0]["source"] == "offline_db"
    assert diagnostics["material_rows"][0]["timestamp"]


def test_stock_snapshot_treats_null_stock_as_zero(tmp_path, monkeypatch) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fake_fetch(**_kwargs):
        return pd.DataFrame(
            {
                "date_time": pd.to_datetime(["2026-05-02T00:00:00Z"]),
                "material_code": ["ore_1"],
                "stock_mt": [None],
            }
        ).set_index("date_time")

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    stock_map, _warnings = provider.get_stock_snapshot()

    assert stock_map["ore1"] == 0.0
    diagnostics = provider.get_data_diagnostics()["stock"]
    ore_row = [row for row in diagnostics["material_rows"] if row["ore_id"] == "ore1"][
        0
    ]
    assert ore_row["source"] == "offline_db_null_as_zero"


def test_recent_active_pellet_ids_follow_charge_usage(tmp_path, monkeypatch) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)
    mapping_path.write_text(
        mapping_path.read_text(encoding="utf-8").replace(
            "chemistry_field_map:",
            """
  - id: pellet1
    display_name: LLOYDS PELLET
    material_key: pellet_1
    price_rs_per_mt: 9650.0
    min_share_pct: 0.0
    max_share_pct: 10.0
    fallback_chemistry:
      fe_t_pct: 64.0
  - id: pellet2
    display_name: SARDA PELLET
    material_key: pellet_2
    price_rs_per_mt: 9650.0
    min_share_pct: 0.0
    max_share_pct: 10.0
    fallback_chemistry:
      fe_t_pct: 64.0
chemistry_field_map:""",
        ),
        encoding="utf-8",
    )

    def fake_fetch(**kwargs):
        assert kwargs["table_name"] == "offline_feed.charge_data"
        assert "pellet_1_mt" in kwargs["columns"]
        assert "pellet_2_mt" in kwargs["columns"]
        return pd.DataFrame(
            {
                "date_time": pd.to_datetime(
                    ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"]
                ),
                "pellet_1_mt": [0.0, 12.0],
                "pellet_2_mt": [0.0, 0.0],
            }
        ).set_index("date_time")

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    active_ids, warnings = provider.get_recent_active_pellet_ids(lookback_days=30)

    assert active_ids == ["pellet1"]
    assert warnings == []
    diagnostics = provider.get_data_diagnostics()["pellet_usage"]
    assert diagnostics["rows"][0]["selected_by_default"] is True
    assert diagnostics["rows"][1]["selected_by_default"] is False


def test_charge_mix_snapshot_groups_burden_materials(tmp_path, monkeypatch) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fake_fetch(**kwargs):
        assert kwargs["table_name"] == "offline_feed.charge_data"
        assert kwargs["query_type"] == "raw"
        return pd.DataFrame(
            {
                "date_time": pd.to_datetime(
                    ["2026-06-05T08:00:00Z", "2026-06-05T09:00:00Z"]
                ),
                "sinter_3_mt": [50.0, 60.0],
                "ore_1_mt": [20.0, 10.0],
                "pellet_1_mt": [10.0, 20.0],
                "coke_1_mt": [15.0, 15.0],
                "nut_coke_1_mt": [3.0, 3.0],
                "flux_1_mt": [1.0, 1.0],
            }
        ).set_index("date_time")

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    snapshot = provider.get_charge_mix_snapshot()

    rows = pd.DataFrame(snapshot["rows"])
    latest = rows[rows["window"] == "latest_row"].set_index("group")
    assert latest.loc["sinter", "quantity_mt"] == 60.0
    assert latest.loc["ore", "quantity_mt"] == 10.0
    assert latest.loc["pellet", "quantity_mt"] == 20.0
    assert latest.loc["coke", "quantity_mt"] == 15.0


def test_recent_manual_blend_snapshot_uses_last_completed_shift(
    tmp_path, monkeypatch
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fake_fetch(**kwargs):
        assert kwargs["table_name"] == "offline_feed.charge_data"
        return pd.DataFrame(
            {
                "date_time": pd.to_datetime(
                    [
                        "2026-06-05T01:00:00Z",
                        "2026-06-05T07:00:00Z",
                        "2026-06-05T10:00:00Z",
                    ]
                ),
                "sinter_3_mt": [50.0, 70.0, 999.0],
                "ore_1_mt": [25.0, 55.0, 999.0],
            }
        ).set_index("date_time")

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )
    selected_ores = [
        OreInput(
            ore_id="sinter",
            display_name="SINTER",
            stock_mt=1000.0,
            price_rs_per_mt=1.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=55.0),
            metadata={"material_key": "sinter_3"},
        ),
        OreInput(
            ore_id="ore1",
            display_name="ORE 1",
            stock_mt=1000.0,
            price_rs_per_mt=1.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=62.0),
            metadata={"material_key": "ore_1"},
        ),
    ]

    snapshot = provider.get_recent_manual_blend_snapshot(selected_ores)

    rows = pd.DataFrame(snapshot["rows"]).set_index("ore_id")
    assert rows.loc["sinter", "quantity_mt"] == 120.0
    assert rows.loc["ore1", "quantity_mt"] == 80.0
    assert rows.loc["sinter", "share_pct"] == 60.0
    assert rows.loc["ore1", "share_pct"] == 40.0


def test_bmo_pellet_database_smoke_check_when_database_url_available() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    psycopg2 = pytest.importorskip("psycopg2")

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    count(*) filter (
                        where coalesce(pellet_1_mt, 0) + coalesce(pellet_2_mt, 0) > 0
                    ) as pellet_charge_rows,
                    coalesce(sum(coalesce(pellet_1_mt, 0)), 0) as pellet_1_mt,
                    coalesce(sum(coalesce(pellet_2_mt, 0)), 0) as pellet_2_mt
                from offline_feed.charge_data
                where date_time >= now() - interval '30 days'
                """
            )
            charge_rows, pellet_1_mt, pellet_2_mt = cur.fetchone()
            cur.execute(
                """
                select material_code, max(date_time)
                from offline_feed.ore_chemistry
                where material_code in ('pellet_1', 'pellet_2')
                group by material_code
                """
            )
            chemistry_rows = cur.fetchall()
            cur.execute(
                """
                select material_code, max(date_time)
                from offline_feed.raw_material_stock
                where material_code in ('pellet_1', 'pellet_2')
                group by material_code
                """
            )
            stock_rows = cur.fetchall()
    finally:
        conn.close()

    assert charge_rows is not None
    assert pellet_1_mt is not None
    assert pellet_2_mt is not None
    assert isinstance(chemistry_rows, list)
    assert isinstance(stock_rows, list)


def test_chemistry_snapshot_uses_offline_tables_and_material_codes(
    tmp_path, monkeypatch
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        table_name = kwargs["table_name"]
        if table_name == "offline_feed.charge_data":
            return pd.DataFrame(
                {
                    "date_time": pd.to_datetime(["2026-05-02T00:00:00Z"]),
                    "sinter_3_mt": [60.0],
                    "ore_1_mt": [20.0],
                }
            ).set_index("date_time")
        if table_name == "offline_feed.sinter_chemistry":
            return pd.DataFrame(
                {
                    "date_time": pd.to_datetime(["2026-05-01T00:00:00Z"]),
                    "material_code": ["sinter_3"],
                    "fe_t": [56.0],
                    "tm": [7.0],
                }
            ).set_index("date_time")
        if table_name == "offline_feed.ore_chemistry":
            return pd.DataFrame(
                {
                    "date_time": pd.to_datetime(
                        [
                            "2026-05-01T00:00:00Z",
                            "2026-05-01T12:00:00Z",
                            "2026-05-03T00:00:00Z",
                        ]
                    ),
                    "material_code": ["ore_1", "ore_1", "ore_1"],
                    "fe_t": [62.0, 66.0, 67.0],
                    "tm": [4.0, None, 6.0],
                }
            ).set_index("date_time")
        raise AssertionError(table_name)

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    chemistry_map, _warnings = provider.get_chemistry_snapshot(
        mode="latest", window_days=30
    )

    assert {call["table_name"] for call in calls} == {
        "offline_feed.charge_data",
        "offline_feed.sinter_chemistry",
        "offline_feed.ore_chemistry",
    }
    assert all(isinstance(call["time_range"], tuple) for call in calls)
    assert chemistry_map["sinter"].fe_t_pct == 56.0
    assert chemistry_map["sinter"].moisture_pct == 7.0
    assert chemistry_map["ore1"].fe_t_pct == 62.0
    assert chemistry_map["ore1"].moisture_pct == 4.0
    diagnostics = provider.get_data_diagnostics()["chemistry"]
    assert diagnostics["fallback_count"] == 0
    assert {row["source"] for row in diagnostics["material_rows"]} == {
        "offline_db_latest_used"
    }
    assert {row["material_code"] for row in diagnostics["material_rows"]} == {
        "sinter_3",
        "ore_1",
    }
    assert all(row["latest_used_time"] for row in diagnostics["material_rows"])
    assert all(row["rows_used"] == 1 for row in diagnostics["material_rows"])
    ore_row = [row for row in diagnostics["material_rows"] if row["ore_id"] == "ore1"][
        0
    ]
    assert ore_row["sample_timestamp"] == "2026-05-01T00:00:00+00:00"


def test_average_chemistry_snapshot_ignores_zero_values(tmp_path, monkeypatch) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fake_fetch(**kwargs):
        table_name = kwargs["table_name"]
        if table_name == "offline_feed.sinter_chemistry":
            return pd.DataFrame()
        if table_name == "offline_feed.ore_chemistry":
            return pd.DataFrame(
                {
                    "date_time": pd.to_datetime(
                        ["2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z"]
                    ),
                    "material_code": ["ore_1", "ore_1"],
                    "fe_t": [0.0, 62.0],
                    "tm": [4.0, 6.0],
                }
            ).set_index("date_time")
        raise AssertionError(table_name)

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    chemistry_map, _warnings = provider.get_chemistry_snapshot(
        mode="avg", window_days=180
    )

    assert chemistry_map["ore1"].fe_t_pct == 62.0
    assert chemistry_map["ore1"].moisture_pct == 5.0
    diagnostics = provider.get_data_diagnostics()["chemistry"]
    ore_row = [row for row in diagnostics["material_rows"] if row["ore_id"] == "ore1"][
        0
    ]
    assert ore_row["source"] == "offline_db_avg_non_zero"


def test_history_frame_does_not_auto_create_missing_static_dataset(tmp_path) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)
    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    history_df, warnings = provider.get_history_frame()

    assert history_df.empty
    assert any("Static dataset file not found" in warning for warning in warnings)


def test_history_frame_layers_recent_online_context_for_model_lags(
    tmp_path, monkeypatch
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)
    csv_path = tmp_path / "furnace_dataset.csv"
    pd.DataFrame(
        {
            "HOT BLAST TEMP.OC": [900.0],
            "BOSH_TEMP_A": [700.0],
        },
        index=pd.to_datetime(["2026-05-01T00:00:00Z"]),
    ).to_csv(csv_path)
    settings_path.write_text(
        settings_path.read_text(encoding="utf-8").replace(
            "static_dataset_path: missing.csv",
            f"static_dataset_path: {csv_path.as_posix()}",
        ),
        encoding="utf-8",
    )

    probe_provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )
    _, start_local, end_local = probe_provider._recent_context_window(4)
    start_utc = start_local.tz_convert("UTC").floor("h")
    end_utc = end_local.tz_convert("UTC").floor("h")
    online_index = pd.DatetimeIndex([start_utc + pd.Timedelta(hours=1), end_utc])

    class FakeOnlineDatasetService:
        def __init__(self, local_tz: str) -> None:
            self.local_tz = local_tz

        def fetch_online_process_params(
            self, _start_date, _end_date, **_kwargs
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "hot_blast_temp_c": [1000.0, 1010.0],
                    "actual_kg_thm": [150.0, 151.0],
                },
                index=online_index,
            )

        def fetch_online_temperature_params(
            self, _start_date, _end_date, **_kwargs
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "bosh_a_c": [800.0, 810.0],
                    "hearth_pad_a_c": [500.0, 510.0],
                },
                index=online_index,
            )

        def fetch_online_heatload_params(
            self, _start_date, _end_date, **_kwargs
        ) -> pd.DataFrame:
            return pd.DataFrame({"total_heat_load": [1.0, 2.0]}, index=online_index)

    monkeypatch.setattr(
        context_module, "OnlineDatasetService", FakeOnlineDatasetService
    )
    monkeypatch.setattr(
        context_module,
        "_fetch_offline_data",
        lambda **_kwargs: pd.DataFrame(
            {
                "date_time": online_index,
                "coke_2_mt": [0.0, 0.5],
                "ore_1_mt": [20.0, 30.0],
                "flux_1_mt": [2.0, 3.0],
                "sinter_3_mt": [40.0, 50.0],
            }
        ).set_index("date_time"),
    )

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    history_df, warnings = provider.get_history_frame(online_lag_hours=4)
    process_context, process_warnings = provider.get_process_context(
        history_df=history_df
    )

    assert warnings == []
    assert process_warnings == []
    assert history_df.iloc[-1]["HOT BLAST TEMP.OC"] == 1010.0
    assert history_df.iloc[-1]["PCI_KG/THM"] == 151.0
    assert history_df.iloc[-1]["BOSH_TEMP_A"] == 810.0
    assert history_df.iloc[-1]["HEARTH_TEMP_A"] == 510.0
    assert history_df.iloc[-1]["TOTAL HEAT LOAD"] == 2.0
    assert history_df.iloc[-1]["COKE_OFF_MT"] == 0.5
    assert history_df.iloc[-1]["ORE_1_CALC_MT"] == 30.0
    assert history_df.iloc[-1]["ORE_CALC_MT"] == 30.0
    assert history_df.iloc[-1]["FLUX_1_CALC_MT"] == 3.0
    assert history_df.iloc[-1]["SINTER_CALC_MT"] == 50.0
    assert process_context["HOT BLAST TEMP.OC"] == 1010.0
    assert process_context["HEARTH_TEMP_A"] == 510.0
    assert process_context["ORE_1_CALC_MT"] == 30.0


def test_recent_online_context_reports_measurement_failures(
    tmp_path,
    monkeypatch,
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    class FakeOnlineDatasetService:
        def __init__(self, local_tz: str) -> None:
            self.local_tz = local_tz

        def fetch_online_process_params(self, *_args, **_kwargs) -> pd.DataFrame:
            raise RuntimeError("process down")

        def fetch_online_temperature_params(self, *_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

        def fetch_online_heatload_params(self, *_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(
        context_module, "OnlineDatasetService", FakeOnlineDatasetService
    )

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    frame, warnings = provider._fetch_recent_online_context(4)

    assert frame.empty
    assert any("process_params fetch failed" in warning for warning in warnings)
    assert any("temperature_profile unavailable" in warning for warning in warnings)
    assert provider._last_online_context_diagnostics["issues"]

    diagnostics = provider.get_data_diagnostics()
    assert diagnostics["online_context"]["rows"] == 0
    assert diagnostics["online_context"]["enabled"] is True


def test_flux_inputs_use_charge_quantities_and_flux_chemistry(
    tmp_path, monkeypatch
) -> None:
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def fake_fetch(**kwargs):
        table_name = kwargs["table_name"]
        if table_name == "offline_feed.charge_data":
            return pd.DataFrame(
                {
                    "date_time": pd.to_datetime(
                        ["2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z"]
                    ),
                    "flux_1_mt": [0.0, 3.0],
                    "flux_2_mt": [0.0, 0.0],
                    "flux_3_mt": [0.0, 0.0],
                }
            ).set_index("date_time")
        if table_name == "offline_feed.flux_chemistry":
            return pd.DataFrame(
                {
                    "date_time": pd.to_datetime(
                        ["2026-05-01T00:00:00Z", "2026-05-02T12:00:00Z"]
                    ),
                    "material_code": ["flux_1", "flux_1"],
                    "tm": [1.5, 2.0],
                    "sio2": [4.0, None],
                    "al2o3": [6.0, 7.0],
                    "cao": [30.0, 31.0],
                    "mgo": [20.0, 21.0],
                    "fe2o3": [1.0, 2.0],
                    "loi": [40.0, 41.0],
                }
            ).set_index("date_time")
        raise AssertionError(table_name)

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)

    provider = EvonithBmoContextProvider(
        setting_path=str(settings_path),
        mapping_path=str(mapping_path),
    )

    flux_inputs, warnings = provider.get_flux_inputs(mode="latest", window_days=30)

    assert len(flux_inputs) == 1
    assert flux_inputs[0].flux_id == "flux_1"
    assert flux_inputs[0].wet_qty_mt == 3.0
    assert flux_inputs[0].moisture_pct == 1.5
    assert flux_inputs[0].sio2_pct == 4.0
    assert flux_inputs[0].cao_pct == 30.0
    assert warnings == []
    diagnostics = provider.get_data_diagnostics()["flux"]
    assert diagnostics["source"] == (
        "offline_feed.charge_data+offline_feed.flux_chemistry"
    )
    assert diagnostics["rows"][0]["rows_used"] == 1
    assert diagnostics["rows"][0]["sample_timestamp"] == "2026-05-01T00:00:00+00:00"
