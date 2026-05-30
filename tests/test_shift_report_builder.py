from __future__ import annotations

from datetime import date, timedelta, timezone
from math import isclose

import pandas as pd

from reports.furnace_report.builder import ShiftBuilder
from reports.furnace_report.data import ShiftRawData
from reports.furnace_report.renderer import as_markdown
from utils.shift_windows import shift_window


class TestShiftBuilder:
    SHIFT_DATE = date(2026, 5, 8)
    SHIFT_LABEL = "A"
    SHIFT_START, SHIFT_END = shift_window(SHIFT_DATE, SHIFT_LABEL)
    PRODUCTION_COLUMN = "Process Params - BF2_PRODUCTION TONNES PER HR"
    CHARGES_PER_HOUR_COLUMN = "Process Params - BF2_CHARGES PER HR"
    TOTAL_O2_FLOW_COLUMN = "Process Params - BF2_TOTAL OXYGEN FLOW"

    @classmethod
    def raw_shift(
        cls,
        charge_df: pd.DataFrame,
        *,
        online_df: pd.DataFrame | None = None,
        total_o2_flow_max: float | None = None,
        ore_chemistry_df: pd.DataFrame | None = None,
        fuel_chemistry_df: pd.DataFrame | None = None,
        flux_chemistry_df: pd.DataFrame | None = None,
        material_fines_df: pd.DataFrame | None = None,
        report_type: str = "Shift",
    ) -> ShiftRawData:
        return ShiftRawData(
            shift_date=cls.SHIFT_DATE,
            shift_label=cls.SHIFT_LABEL,
            shift_start_ist=cls.SHIFT_START,
            shift_end_ist=cls.SHIFT_END,
            online_df=online_df if online_df is not None else pd.DataFrame(),
            total_o2_flow_max=total_o2_flow_max,
            hm_slag_df=pd.DataFrame(),
            charge_df=charge_df,
            ore_chemistry_df=(
                ore_chemistry_df if ore_chemistry_df is not None else pd.DataFrame()
            ),
            fuel_chemistry_df=(
                fuel_chemistry_df if fuel_chemistry_df is not None else pd.DataFrame()
            ),
            flux_chemistry_df=(
                flux_chemistry_df if flux_chemistry_df is not None else pd.DataFrame()
            ),
            material_fines_df=(
                material_fines_df if material_fines_df is not None else pd.DataFrame()
            ),
            report_type=report_type,
        )

    def test_sums_neon_charge_data_consumption_columns(self) -> None:
        report = ShiftBuilder().build(
            self.raw_shift(
                pd.DataFrame(
                    {
                        "coke_1_mt": [10.0, 11.0],
                        "coke_2_mt": [2.0, None],
                        "nut_coke_1_mt": [1.5, 2.5],
                        "nut_coke_2_mt": [0.5, 1.0],
                        "sinter_1_mt": [4.0, 5.0],
                        "sinter_4_mt": [6.0, 7.0],
                        "ore_1_mt": [8.0, 9.0],
                        "ore_12_mt": [10.0, 11.0],
                        "pellet_1_mt": [3.0, 4.0],
                        "pellet_2_mt": [1.0, 2.0],
                        "flux_1_mt": [2.0, 3.0],
                        "flux_3_mt": [4.0, 5.0],
                    }
                )
            )
        )

        assert report.coke_t == 23.0
        assert report.nut_coke_t == 5.5
        assert report.sinter_t == 22.0
        assert report.ore_t == 38.0
        assert report.pellet_t == 10.0
        assert report.flux_t == 14.0

    def test_reports_zero_when_charge_columns_are_zero(self) -> None:
        report = ShiftBuilder().build(
            self.raw_shift(pd.DataFrame({"coke_1_mt": [0.0], "coke_2_mt": [0.0]}))
        )

        assert report.coke_t == 0.0

    def test_total_charges_counts_charge_rows_not_online_rate(self) -> None:
        report = ShiftBuilder().build(
            self.raw_shift(
                pd.DataFrame({"charge_no": [1, 2, 3]}),
                online_df=pd.DataFrame({self.CHARGES_PER_HOUR_COLUMN: [99.0]}),
            )
        )

        assert report.total_charges == 3

    def test_total_o2_flow_is_day_only_and_uses_timeframe_max(self) -> None:
        online_df = pd.DataFrame({self.TOTAL_O2_FLOW_COLUMN: [999.0]})
        shift_report = ShiftBuilder().build(
            self.raw_shift(
                pd.DataFrame(),
                online_df=online_df,
                total_o2_flow_max=130.0,
            )
        )
        day_report = ShiftBuilder().build(
            self.raw_shift(
                pd.DataFrame(),
                online_df=online_df,
                total_o2_flow_max=130.0,
                report_type="Day",
            )
        )

        assert shift_report.total_o2_flow is None
        assert day_report.total_o2_flow == 130.0
        assert "Total Oxygen Flow" not in as_markdown(shift_report)
        assert "| Day Total Oxygen Flow | Nm3 | 130 | - |" in as_markdown(day_report)

    def test_burden_ratio_uses_existing_consumption_totals(self) -> None:
        report = ShiftBuilder().build(
            self.raw_shift(
                pd.DataFrame(
                    {
                        "sinter_1_mt": [140.0],
                        "ore_1_mt": [56.0],
                        "pellet_1_mt": [4.0],
                    }
                )
            )
        )

        assert report.burden_ratio == "70.0: 28.0: 2.0"
        assert (
            "| Burden Ratio | Sinter:Ore:Pellet | 70.0: 28.0: 2.0 | - |"
            in as_markdown(report)
        )

    def test_burden_ratio_treats_missing_consumption_group_as_zero(self) -> None:
        report = ShiftBuilder().build(
            self.raw_shift(
                pd.DataFrame(
                    {
                        "ore_1_mt": [56.0],
                        "sinter_1_mt": [140.0],
                    }
                )
            )
        )

        assert report.burden_ratio == "71.4: 28.6: 0.0"

    def test_material_inputs_use_latest_available_analysis_before_shift_end(
        self,
    ) -> None:
        before_shift = self.SHIFT_START.astimezone(timezone.utc) - timedelta(hours=1)
        in_shift = self.SHIFT_START.astimezone(timezone.utc) + timedelta(hours=2)
        after_shift = self.SHIFT_END.astimezone(timezone.utc) + timedelta(minutes=1)

        report = ShiftBuilder().build(
            self.raw_shift(
                pd.DataFrame(
                    {
                        "coke_1_mt": [10.0],
                        "nut_coke_1_mt": [2.0],
                        "sinter_1_mt": [40.0],
                        "ore_1_mt": [20.0],
                        "pellet_1_mt": [10.0],
                        "flux_1_mt": [5.0],
                    }
                ),
                online_df=pd.DataFrame({self.PRODUCTION_COLUMN: [100.0]}),
                fuel_chemistry_df=pd.DataFrame(
                    [
                        {"date_time": in_shift, "material_code": "coke_1", "tm": 2.0},
                        {
                            "date_time": before_shift,
                            "material_code": "nut_coke_1",
                            "moisture": 3.0,
                        },
                    ]
                ),
                ore_chemistry_df=pd.DataFrame(
                    [
                        {
                            "date_time": before_shift,
                            "material_code": "ore_1",
                            "tm": 4.0,
                        },
                        {
                            "date_time": before_shift,
                            "material_code": "pellet_1",
                            "tm": 5.0,
                        },
                        {"date_time": after_shift, "material_code": "ore_1", "tm": 9.0},
                    ]
                ),
                flux_chemistry_df=pd.DataFrame(
                    [{"date_time": in_shift, "material_code": "flux_1", "tm": 1.0}]
                ),
                material_fines_df=pd.DataFrame(
                    [
                        {
                            "date_time": in_shift,
                            "material_code": "coke_1",
                            "plus_20mm": 1.0,
                            "plus_10mm": 2.0,
                            "minus_10mm": 3.0,
                        },
                        {
                            "date_time": before_shift,
                            "material_code": "nut_coke_1",
                            "plus_10mm": 4.0,
                            "plus_6mm": 1.0,
                            "minus_6mm": 5.0,
                        },
                        {
                            "date_time": before_shift,
                            "material_code": "ore_1",
                            "plus_6mm": 7.0,
                            "minus_6mm": 2.0,
                        },
                        {
                            "date_time": after_shift,
                            "material_code": "ore_1",
                            "plus_6mm": 99.0,
                            "minus_6mm": 99.0,
                        },
                        {
                            "date_time": in_shift,
                            "material_code": "flux_1",
                            "minus_10mm": 8.0,
                        },
                    ]
                ),
            )
        )

        assert isclose(report.theoretical_production or 0.0, 800.0)
        assert report.burden_moisture_input == 1.39
        assert report.ibrm == 0.09
        assert "| IBRM | - | 0.1 | - |" in as_markdown(report)
        assert report.fines_input == 3.25
