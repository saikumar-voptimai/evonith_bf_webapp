from datetime import date

import pandas as pd

from furnace_data.dataset.fetcher import DatasetFetcher
from furnace_data.dataset.service import DatasetService


def test_dataset_fetcher_routes_interactive_ml_to_offline_db() -> None:
    class FakeService:
        cutoff_date = date(2025, 8, 24)

        def __init__(self) -> None:
            self.calls = []
            self.step1_called = False

        def fetch_step1(self, *args, **kwargs):
            self.step1_called = True
            return pd.DataFrame()

        def fetch_step2(self, start, end, mode, allowed_columns=None):
            self.calls.append(("rm", mode))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame(
                {"ore_1_mt": [10.0], "flux_1_mt": [2.0], "offline_only": [1.0]},
                index=idx,
            )

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None):
            self.calls.append(("hm", None))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"chem_pct_si": [0.4]}, index=idx)

        def fetch_distribution_data(self, start, end):
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"TOTAL_COKE_PORTIONS": [10.0]}, index=idx)

        def fetch_online_process_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_temperature_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_heatload_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_misc_params(self, start, end, **_kwargs):
            return pd.DataFrame()

    service = FakeService()
    fetcher = DatasetFetcher(service=service)

    df = fetcher.get_ml_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
    )

    assert service.step1_called is False
    assert service.calls == [("rm", "charge"), ("hm", None)]
    assert "ORE_1_CALC_MT" in df.columns
    assert "FLUX_1_CALC_MT" in df.columns
    assert "CHEM_PCT_SI" in df.columns
    assert "offline_only" in df.columns


def test_dataset_fetcher_default_source_is_offline_db_for_static_manager() -> None:
    class FakeService:
        cutoff_date = date(2025, 8, 24)

        def __init__(self) -> None:
            self.calls = []

        def fetch_step1(self, *args, **kwargs):
            return pd.DataFrame()

        def fetch_step2(self, start, end, mode, allowed_columns=None):
            self.calls.append(("rm", mode))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"ore_1_mt": [10.0]}, index=idx)

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None):
            self.calls.append(("hm", None))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"chem_pct_si": [0.4]}, index=idx)

        def fetch_distribution_data(self, start, end):
            return pd.DataFrame()

        def fetch_online_process_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_temperature_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_heatload_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_misc_params(self, start, end, **_kwargs):
            return pd.DataFrame()

    service = FakeService()
    fetcher = DatasetFetcher(service=service)

    fetcher.get_ml_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
    )

    assert service.calls == [("rm", "charge"), ("hm", None)]


def test_dataset_fetcher_slices_datetime_index_with_date_bounds() -> None:
    class FakeService:
        cutoff_date = date(2025, 8, 24)

        def fetch_step2(self, start, end, mode, allowed_columns=None):
            idx = pd.DatetimeIndex(["2026-01-01T06:00:00"], name="time")
            return pd.DataFrame({"ore_1_mt": [10.0]}, index=idx)

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None):
            return pd.DataFrame()

        def fetch_distribution_data(self, start, end):
            return pd.DataFrame()

        def fetch_online_process_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_temperature_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_heatload_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_misc_params(self, start, end, **_kwargs):
            return pd.DataFrame()

    output = DatasetFetcher(service=FakeService()).get_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
    )

    assert len(output) == 1
    assert output.index.name == "time"


def test_dataset_fetcher_includes_topbar_and_heatload_in_post_cutoff_dataset() -> None:
    idx = pd.DatetimeIndex(["2026-01-01"], name="time")

    class FakeService:
        cutoff_date = date(2025, 8, 24)

        def fetch_step2(self, start, end, mode, allowed_columns=None):
            return pd.DataFrame({"ore_1_mt": [10.0]}, index=idx)

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None):
            return pd.DataFrame()

        def fetch_distribution_data(self, start, end):
            return pd.DataFrame()

        def fetch_online_process_params(self, start, end, **_kwargs):
            return pd.DataFrame({"top_bar": [1.25]}, index=idx)

        def fetch_online_temperature_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_heatload_params(self, start, end, **_kwargs):
            return pd.DataFrame({"total_heat_load": [42.0]}, index=idx)

        def fetch_online_misc_params(self, start, end, **_kwargs):
            return pd.DataFrame({"stock_rod_level": [3.2]}, index=idx)

    output = DatasetFetcher(service=FakeService()).get_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
    )

    assert output.loc[idx[0], "TOPBAR"] == 1.25
    assert output.loc[idx[0], "TOTAL HEAT LOAD"] == 42.0
    assert output.loc[idx[0], "STOCKRODLEVEL"] == 3.2


def test_dataset_fetcher_local_delta_can_raise_online_failures() -> None:
    class FakeService:
        cutoff_date = date(2025, 8, 24)

        def fetch_step2(self, start, end, mode, allowed_columns=None):
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"ore_1_mt": [10.0]}, index=idx)

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None):
            return pd.DataFrame()

        def fetch_distribution_data(self, start, end):
            return pd.DataFrame()

        def fetch_online_process_params(self, start, end, **_kwargs):
            raise RuntimeError("process fetch failed")

        def fetch_online_temperature_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_heatload_params(self, start, end, **_kwargs):
            return pd.DataFrame()

        def fetch_online_misc_params(self, start, end, **_kwargs):
            return pd.DataFrame()

    fetcher = DatasetFetcher(service=FakeService())

    try:
        fetcher.build_local_delta(
            date(2026, 1, 1),
            date(2026, 1, 1),
            raise_on_error=True,
        )
    except RuntimeError as exc:
        assert "process fetch failed" in str(exc)
    else:
        raise AssertionError("Expected build_local_delta to raise online failures")


def test_dataset_service_process_params_maps_body_dp_top_to_topbar(monkeypatch) -> None:
    class FakeFetcher:
        def __init__(self, measurement):
            self.measurement = measurement

        def fetch_averaged_data(self, **kwargs):
            assert self.measurement == "process_params"
            return pd.DataFrame(
                {
                    "time": ["2026-01-01T00:00:00Z"],
                    "body_dp_top": [1.4],
                    "oxygen_flow": [5200.0],
                    "charges_per_hour": [5.5],
                    "steam_injection": [1200.0],
                    "top_temp_1": [210.0],
                    "top_temp_2": [211.0],
                    "top_temp_3": [212.0],
                    "top_temp_4": [213.0],
                    "top_temp_avg": [211.5],
                }
            )

    monkeypatch.setattr("furnace_data.influx.base.BaseDataFetcher", FakeFetcher)

    df = DatasetService().fetch_online_process_params(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
    )

    assert float(df.iloc[0]["top_bar"]) == 1.4
    assert float(df.iloc[0]["oxygen_flow_nm3hr"]) == 5200.0
    assert float(df.iloc[0]["charges_per_hr"]) == 5.5
    assert float(df.iloc[0]["steam_kgs_hr"]) == 1200.0
    assert float(df.iloc[0]["ftg_uptake_cat16_c"]) == 210.0
    assert float(df.iloc[0]["ftg_uptake_bt12_c"]) == 211.0
    assert float(df.iloc[0]["ftg_uptake_ct08_c"]) == 212.0
    assert float(df.iloc[0]["ftg_uptake_dt04_c"]) == 213.0
    assert float(df.iloc[0]["ftg_uptake_avg_c"]) == 211.5


def test_dataset_service_extracts_pellet_chemistry_from_ore_table(monkeypatch) -> None:
    def fake_offline_fetch(table_name, time_range):
        idx = pd.DatetimeIndex(["2026-01-02T00:00:00Z"], name="time")
        if table_name == "offline_feed.charge_data":
            return pd.DataFrame({"pellet_1_mt": [5.0]}, index=idx)
        if table_name == "offline_feed.raw_material_strength_analysis":
            return pd.DataFrame()
        if table_name == "offline_feed.v_charge_material_quantities":
            return pd.DataFrame(
                {
                    "material_code": ["pellet_1"],
                    "quantity": [5.0],
                    "source_column_name": ["pellet_1_mt"],
                },
                index=idx,
            )
        if table_name == "offline_feed.ore_chemistry":
            return pd.DataFrame(
                {
                    "material_code": ["pellet_1"],
                    "sio2": [4.2],
                    "al2o3": [2.4],
                    "cao": [1.1],
                    "tm": [0.7],
                },
                index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="time"),
            )
        if table_name in {
            "offline_feed.sinter_chemistry",
            "offline_feed.fuel_chemistry",
            "offline_feed.flux_chemistry",
        }:
            return pd.DataFrame()
        raise AssertionError(table_name)

    monkeypatch.setattr(
        "furnace_data.dataset.service.fetch_database_offline_data",
        fake_offline_fetch,
    )

    df = DatasetService().fetch_rm_data(
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 2),
        mode="charge",
    )

    assert float(df.iloc[0]["pellet_sio2_pct"]) == 4.2
    assert float(df.iloc[0]["pellet_al2o3_pct"]) == 2.4
    assert float(df.iloc[0]["pellet_cao_pct"]) == 1.1
    assert float(df.iloc[0]["pellet_tm_pct"]) == 0.7


def test_dataset_service_maps_fuel_moisture_to_moisture_and_im(monkeypatch) -> None:
    def fake_offline_fetch(table_name, time_range):
        idx = pd.DatetimeIndex(["2026-01-02T00:00:00Z"], name="time")
        if table_name == "offline_feed.charge_data":
            return pd.DataFrame({"coke_1_mt": [4.0], "nut_coke_1_mt": [1.0]}, index=idx)
        if table_name == "offline_feed.raw_material_strength_analysis":
            return pd.DataFrame()
        if table_name == "offline_feed.v_charge_material_quantities":
            return pd.DataFrame(
                {
                    "material_code": ["coke_1", "nut_coke_1"],
                    "quantity": [4.0, 1.0],
                    "source_column_name": ["coke_1_mt", "nut_coke_1_mt"],
                },
                index=pd.DatetimeIndex(
                    ["2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"],
                    name="time",
                ),
            )
        if table_name == "offline_feed.fuel_chemistry":
            return pd.DataFrame(
                {
                    "material_code": ["coke_1", "nut_coke_1", "pci_1"],
                    "moisture": [5.0, 7.0, 1.5],
                    "ash": [12.0, 13.0, 8.0],
                    "vm": [2.0, 3.0, 18.0],
                    "fc": [80.0, 78.0, 72.0],
                },
                index=pd.DatetimeIndex(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                    ],
                    name="time",
                ),
            )
        if table_name in {
            "offline_feed.ore_chemistry",
            "offline_feed.sinter_chemistry",
            "offline_feed.flux_chemistry",
        }:
            return pd.DataFrame()
        raise AssertionError(table_name)

    monkeypatch.setattr(
        "furnace_data.dataset.service.fetch_database_offline_data",
        fake_offline_fetch,
    )

    df = DatasetService().fetch_rm_data(
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 2),
        mode="charge",
    )

    assert float(df.iloc[0]["coke_moist_pct"]) == 5.0
    assert float(df.iloc[0]["coke_im_pct"]) == 5.0
    assert float(df.iloc[0]["nutcoke_moist_pct"]) == 7.0
    assert float(df.iloc[0]["nutcoke_im_pct"]) == 7.0
    assert float(df.iloc[0]["pci2_im_pct"]) == 1.5
    assert float(df.iloc[0]["pci2_ash_pct"]) == 8.0
    assert float(df.iloc[0]["pci2_vm_pct"]) == 18.0
    assert float(df.iloc[0]["pci2_fc_pct"]) == 72.0


def test_dataset_service_temperature_params_computes_hearth_average(monkeypatch) -> None:
    class FakeFetcher:
        def __init__(self, measurement):
            self.measurement = measurement

        def fetch_averaged_data(self, **kwargs):
            assert self.measurement == "temperature_profile"
            return pd.DataFrame(
                {
                    "time": ["2026-01-01T00:00:00Z"],
                    "temp_4373_a": [100.0],
                    "temp_5411_b": [200.0],
                    "temp_5757_c": [300.0],
                    "temp_6103_d": [400.0],
                }
            )

    monkeypatch.setattr("furnace_data.influx.base.BaseDataFetcher", FakeFetcher)

    df = DatasetService().fetch_online_temperature_params(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
    )

    assert float(df.iloc[0]["hearth_pad_avg_c"]) == 250.0


def test_dataset_service_misc_params_maps_stock_rod_radar_level(monkeypatch) -> None:
    class FakeFetcher:
        def __init__(self, measurement):
            self.measurement = measurement

        def fetch_averaged_data(self, **kwargs):
            assert self.measurement == "miscellaneous"
            return pd.DataFrame(
                {
                    "time": ["2026-01-01T00:00:00Z"],
                    "stock_rod_radar_level": [4.2],
                    "stock_rod1_pos": [3.9],
                    "stock_rod2_pos": [4.1],
                }
            )

    monkeypatch.setattr("furnace_data.influx.base.BaseDataFetcher", FakeFetcher)

    df = DatasetService().fetch_online_misc_params(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
    )

    assert float(df.iloc[0]["stock_rod_level"]) == 4.2


def test_dataset_service_offline_rm_fetch_combines_charge_and_rm_hm(monkeypatch) -> None:
    calls = []

    def fake_offline_fetch(table_name, time_range):
        calls.append(table_name)
        if table_name == "offline_feed.charge_data":
            idx = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="time")
            return pd.DataFrame(
                {
                    "sinter_1_mt": [10.0],
                    "sinter_2_mt": [5.0],
                    "ore_1_mt": [4.0],
                    "pci_mt": [2.0],
                },
                index=idx,
            )
        if table_name == "offline_feed.raw_material_strength_analysis":
            idx = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="time")
            return pd.DataFrame({"ri": [70.0], "rdi": [30.0]}, index=idx)
        if table_name == "offline_feed.v_charge_material_quantities":
            idx = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="time")
            return pd.DataFrame(
                {
                    "material_code": ["ore_1"],
                    "quantity": [4.0],
                    "source_column_name": ["ore_1_mt"],
                },
                index=idx,
            )
        if table_name in {
            "offline_feed.ore_chemistry",
            "offline_feed.sinter_chemistry",
            "offline_feed.fuel_chemistry",
            "offline_feed.flux_chemistry",
        }:
            return pd.DataFrame()
        raise AssertionError(table_name)

    monkeypatch.setattr(
        "furnace_data.dataset.service.fetch_database_offline_data",
        fake_offline_fetch,
    )

    service = DatasetService()
    df = service.fetch_rm_data(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        mode="charge",
    )

    assert calls == [
        "offline_feed.charge_data",
        "offline_feed.raw_material_strength_analysis",
        "offline_feed.v_charge_material_quantities",
        "offline_feed.ore_chemistry",
        "offline_feed.sinter_chemistry",
        "offline_feed.fuel_chemistry",
        "offline_feed.flux_chemistry",
    ]
    assert float(df.iloc[0]["sinter_mt"]) == 15.0
    assert float(df.iloc[0]["pci2_mt"]) == 2.0
    assert float(df.iloc[0]["sinter_hot_strength_ri"]) == 70.0
    assert float(df.iloc[0]["sinter_hot_strength_rdi"]) == 30.0


def test_dataset_service_offline_weighted_chemistry_uses_latest_before(monkeypatch) -> None:
    def fake_offline_fetch(table_name, time_range):
        if table_name == "offline_feed.charge_data":
            idx = pd.DatetimeIndex(["2026-01-02T00:00:00Z"], name="time")
            return pd.DataFrame(
                {"ore_2_mt": [12.0], "ore_6_mt": [99.0]},
                index=idx,
            )
        if table_name == "offline_feed.raw_material_strength_analysis":
            return pd.DataFrame()
        if table_name == "offline_feed.v_charge_material_quantities":
            idx = pd.DatetimeIndex(
                ["2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"],
                name="time",
            )
            return pd.DataFrame(
                {
                    "material_code": ["ore_1", "ore_2"],
                    "quantity": [4.0, 12.0],
                    "source_column_name": ["ore_1_mt", "ore_2_mt"],
                },
                index=idx,
            )
        if table_name == "offline_feed.ore_chemistry":
            idx = pd.DatetimeIndex(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T12:00:00Z",
                    "2026-01-03T00:00:00Z",
                ],
                name="time",
            )
            return pd.DataFrame(
                {
                    "material_code": ["ore_1", "ore_2", "ore_2"],
                    "fe_t": [60.0, 64.0, 70.0],
                },
                index=idx,
            )
        if table_name in {
            "offline_feed.sinter_chemistry",
            "offline_feed.fuel_chemistry",
            "offline_feed.flux_chemistry",
        }:
            return pd.DataFrame()
        raise AssertionError(table_name)

    monkeypatch.setattr(
        "furnace_data.dataset.service.fetch_database_offline_data",
        fake_offline_fetch,
    )

    service = DatasetService()
    df = service.fetch_rm_data(
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 2),
        mode="charge",
    )

    assert float(df.iloc[0]["ore_6_mt"]) == 99.0
    assert float(df.iloc[0]["ore_mt"]) == 111.0
    assert float(df.iloc[0]["ore_fe_total_pct"]) == 63.0
