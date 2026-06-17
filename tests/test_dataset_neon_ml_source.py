from datetime import date

import pandas as pd

from furnace_data.dataset.fetcher import DatasetFetcher
from furnace_data.dataset.service import DatasetService
from furnace_data.neon_db.offline import NEON_OFFLINE_REPORT_MAP


def test_dataset_fetcher_routes_interactive_ml_to_neon() -> None:
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
            return pd.DataFrame({"ore_1_mt": [10.0], "neon_only": [1.0]}, index=idx)

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None):
            self.calls.append(("hm", None))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"chem_pct_si": [0.4]}, index=idx)

        def fetch_distribution_data(self, start, end):
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"TOTAL_COKE_PORTIONS": [10.0]}, index=idx)

    service = FakeService()
    fetcher = DatasetFetcher(service=service)

    df = fetcher.get_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
    )

    assert service.step1_called is False
    assert service.calls == [("rm", "charge"), ("hm", None)]
    assert "ORE_1_CALC_MT" in df.columns
    assert "CHEM_PCT_SI" in df.columns
    assert "neon_only" in df.columns


def test_dataset_fetcher_default_source_is_neon_for_static_manager() -> None:
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

    service = FakeService()
    fetcher = DatasetFetcher(service=service)

    fetcher.get_dataset(
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

    output = DatasetFetcher(service=FakeService()).get_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
    )

    assert len(output) == 1
    assert output.index.name == "time"


def test_dataset_service_neon_rm_fetch_combines_charge_and_rm_hm(monkeypatch) -> None:
    calls = []

    def fake_neon_fetch(table_name, time_range):
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
        "furnace_data.dataset.service.fetch_neon_offline_data",
        fake_neon_fetch,
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


def test_dataset_service_maps_new_strength_property_slots(monkeypatch) -> None:
    calls = []

    def fake_neon_fetch(table_name, time_range):
        calls.append(table_name)
        if table_name == "offline_feed.charge_data":
            idx = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="time")
            return pd.DataFrame({"sinter_3_mt": [10.0]}, index=idx)
        if table_name == "offline_feed.raw_material_strength_analysis":
            idx = pd.DatetimeIndex(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ],
                name="time",
            )
            return pd.DataFrame(
                {
                    "material_code": ["coke_1", "sinter_1", "sinter_3"],
                    "property_1": [78.46, 9.99, 5.08],
                    "property_2": [7.2, 70.0, 80.02],
                    "property_3": [25.44, 31.0, None],
                    "property_4": [65.62, 61.0, 66.54],
                },
                index=idx,
            )
        if table_name == "plant_master.material_property_mapping":
            return pd.DataFrame(
                {
                    "material_code": ["coke_1", "sinter_1", "sinter_3"],
                    "property_1_name": ["M-40", "AI", "AI"],
                    "property_2_name": ["M-10", "TI", "TI"],
                    "property_3_name": ["CRI", "RDI", "RDI"],
                    "property_4_name": ["CSR", "RI", "RI"],
                }
            )
        if table_name == "offline_feed.v_charge_material_quantities":
            idx = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="time")
            return pd.DataFrame(
                {
                    "material_code": ["sinter_3"],
                    "quantity": [10.0],
                    "source_column_name": ["sinter_3_mt"],
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
        "furnace_data.dataset.service.fetch_neon_offline_data",
        fake_neon_fetch,
    )

    service = DatasetService()
    df = service.fetch_rm_data(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        mode="charge",
        allowed_columns={
            "sinter_cold_strength_ai": "SINTER_COLD_STRENGTH_AI",
            "sinter_cold_strength_ti": "SINTER_COLD_STRENGTH_TI",
            "sinter_hot_strength_rdi": "SINTER_HOT_STRENGTH_RDI",
            "sinter_hot_strength_ri": "SINTER_HOT_STRENGTH_RI",
            "coke_m40": "COKE_M-40",
            "coke_m10": "COKE_M-10",
            "coke_csr": "CSR",
            "coke_cri": "CRI",
        },
    )

    assert "plant_master.material_property_mapping" in calls
    assert float(df.iloc[0]["sinter_cold_strength_ai"]) == 5.08
    assert float(df.iloc[0]["sinter_cold_strength_ti"]) == 80.02
    assert float(df.iloc[0]["sinter_hot_strength_rdi"]) == 31.0
    assert float(df.iloc[0]["sinter_hot_strength_ri"]) == 66.54
    assert float(df.iloc[0]["coke_m40"]) == 78.46
    assert float(df.iloc[0]["coke_m10"]) == 7.2
    assert float(df.iloc[0]["coke_cri"]) == 25.44
    assert float(df.iloc[0]["coke_csr"]) == 65.62
    assert "property_1" not in df.columns


def test_dataset_service_neon_weighted_chemistry_uses_latest_before(monkeypatch) -> None:
    def fake_neon_fetch(table_name, time_range):
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
        "furnace_data.dataset.service.fetch_neon_offline_data",
        fake_neon_fetch,
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
