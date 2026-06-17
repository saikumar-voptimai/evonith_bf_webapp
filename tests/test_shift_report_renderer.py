"""Tests for shift report markdown rendering."""

from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

FURNACE_DATA_ROOT = Path(__file__).resolve().parents[1] / "furnace_data"
if str(FURNACE_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(FURNACE_DATA_ROOT))

from reports.shift_report.data import ParamStats, ShiftReportData, TempRow
from reports.shift_report.renderer import as_markdown


def _stats(mean: float | None = None, std: float | None = None) -> ParamStats:
    return ParamStats(mean=mean, std=std)


class ShiftReportRendererTests(unittest.TestCase):
    def test_oxygen_flow_std_dev_is_rendered(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        start = datetime(2026, 5, 8, 6, 0, tzinfo=ist)
        report = ShiftReportData(
            shift_date=date(2026, 5, 8),
            shift_label="A",
            shift_start_ist=start,
            shift_end_ist=start + timedelta(hours=8),
            status="STABLE",
            status_flags=[],
            production_rate=None,
            theoretical_production=None,
            total_charges=None,
            coke_t=None,
            nut_coke_t=None,
            sinter_t=None,
            ore_t=None,
            pellet_t=None,
            flux_t=None,
            fuel_rate=None,
            coke_rate=None,
            nut_coke_rate=None,
            pci_rate=None,
            hm_si=None,
            hm_s=None,
            hm_temp=None,
            slag_basicity=None,
            total_taps=None,
            blast_volume=_stats(),
            blast_temp=_stats(),
            blast_pressure=_stats(),
            o2_flow=_stats(2500.0, 125.5),
            o2_enrichment=_stats(),
            permeability=_stats(),
            etaco=_stats(),
            raft=_stats(),
            uptake=TempRow(None, None, None, None),
            lower_stack=TempRow(None, None, None, None),
            belly=TempRow(None, None, None, None),
            bosh=TempRow(None, None, None, None),
            hearth_4_3_a=None,
            hearth_5_4_c=None,
            hearth_5_7_c=None,
            hearth_6_1_b=None,
        )

        markdown = as_markdown(report)

        self.assertIn("| Oxygen Flow | Nm3/hr | 2500.00 | 125.50 |", markdown)


if __name__ == "__main__":
    unittest.main()
