from datetime import date

import pandas as pd

from furnace_data.dataset.fetcher import DatasetFetcher
from furnace_data.dataset.service import DatasetService
from furnace_data.neon_db.offline import NEON_OFFLINE_REPORT_MAP


def test_neon_report_map_exposes_rm_charge_and_dpr_tables() -> None:
    assert NEON_OFFLINE_REPORT_MAP["RM_CHARGE"] == ["charge_data", "rm_hm"]
    assert NEON_OFFLINE_REPORT_MAP["RM_DPR"] == ["dpr_data", "rm_hm"]


def test_dataset_fetcher_routes_interactive_ml_to_neon() -> None:
    class FakeService:
        cutoff_date = date(2025, 8, 24)

        def __init__(self) -> None:
            self.sources = []
            self.step1_called = False

        def fetch_step1(self, *args, **kwargs):
            self.step1_called = True
            return pd.DataFrame()

        def fetch_step2(self, start, end, mode, allowed_columns=None, source="influx"):
            self.sources.append(("rm", source))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"ore_1_mt": [10.0], "neon_only": [1.0]}, index=idx)

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None, source="influx"):
            self.sources.append(("hm", source))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"chem_pct_si": [0.4]}, index=idx)

        def fetch_distribution_data(self, start, end):
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"TOTAL_COKE_PORTIONS": [10.0]}, index=idx)

    service = FakeService()
    fetcher = DatasetFetcher(service=service)

    df = fetcher.get_ml_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
        source="neon_db",
    )

    assert service.step1_called is False
    assert service.sources == [("rm", "neon_db"), ("hm", "neon_db")]
    assert "ORE_1_CALC_MT" in df.columns
    assert "CHEM_PCT_SI" in df.columns
    assert "neon_only" in df.columns


def test_dataset_fetcher_default_source_remains_influx_for_static_manager() -> None:
    class FakeService:
        cutoff_date = date(2025, 8, 24)

        def __init__(self) -> None:
            self.sources = []

        def fetch_step1(self, *args, **kwargs):
            return pd.DataFrame()

        def fetch_step2(self, start, end, mode, allowed_columns=None, source="influx"):
            self.sources.append(("rm", source))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"ore_1_mt": [10.0]}, index=idx)

        def fetch_hotmetal_hourly(self, start, end, keep_columns=None, source="influx"):
            self.sources.append(("hm", source))
            idx = pd.DatetimeIndex(["2026-01-01"], name="time")
            return pd.DataFrame({"chem_pct_si": [0.4]}, index=idx)

        def fetch_distribution_data(self, start, end):
            return pd.DataFrame()

    service = FakeService()
    fetcher = DatasetFetcher(service=service)

    fetcher.get_ml_dataset(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 1),
        rm_choice="RM Charge",
    )

    assert service.sources == [("rm", "influx"), ("hm", "influx")]


def test_dataset_service_neon_rm_fetch_combines_charge_and_rm_hm(monkeypatch) -> None:
    calls = []

    def fake_neon_fetch(table_name, time_range):
        calls.append(table_name)
        if table_name == "charge_data":
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
        if table_name == "rm_hm":
            idx = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="time")
            return pd.DataFrame({"ri": [70.0], "rdi": [30.0]}, index=idx)
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
        source="neon_db",
    )

    assert calls == ["charge_data", "rm_hm"]
    assert float(df.iloc[0]["sinter_mt"]) == 15.0
    assert float(df.iloc[0]["pci2_mt"]) == 2.0
    assert float(df.iloc[0]["sinter_hot_strength_ri"]) == 70.0
    assert float(df.iloc[0]["sinter_hot_strength_rdi"]) == 30.0
