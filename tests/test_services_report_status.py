"""KPI status of the shift reports (SV-2.4, #296): thresholds read from config/shift_report.yml, one rule for the
report builder and the twin-api's KPI view, with the Reports tab's flag texts unchanged."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest

from reports.furnace_report import builder
from reports.furnace_report import status as report_status
from services import reports

CONFIG = {
    "status_thresholds": {
        "etaco": {"warn": 42.0, "crit": 40.0},
        "fuel_rate": {"warn": 530.0, "crit": 570.0},
        "raft_lo": 2100.0,
        "raft_hi": 2350.0,
        "perm_lo": 1000.0,
        "perm_hi": 1600.0,
    }
}


def legacy_status(etaco: Any, fuel_rate: Any, raft: Any, perm: Any) -> tuple[str, list[str]]:
    """The builder's status before SV-2.4 (thresholds of the shipped configuration, texts hard-coded)."""
    flags: list[str] = []
    if etaco is not None:
        if etaco < 40.0:
            flags.append(f"ETA CO {etaco:.2f}% (critical < 40%)")
        elif etaco < 42.0:
            flags.append(f"ETA CO {etaco:.2f}% (< 42% normal)")
    if fuel_rate is not None:
        if fuel_rate > 570.0:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (critical > 570)")
        elif fuel_rate > 530.0:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (> 530 attention)")
    if raft is not None and (raft < 2100.0 or raft > 2350.0):
        flags.append(f"RAFT {raft:.2f}°C (normal 2100–2350)")
    if perm is not None and (perm < 1000.0 or perm > 1600.0):
        flags.append(f"Permeability {perm:.2f} (normal 1000–1600)")
    n = len(flags)
    return ("STABLE" if n == 0 else "ATTENTION REQUIRED" if n <= 2 else "UNSTABLE"), flags


GRID = list(
    itertools.product(
        (None, 39.5, 40.0, 41.2, 42.0, 46.3, math.nan),
        (None, 512.0, 530.0, 548.7, 570.0, 571.3),
        (None, 2050.0, 2100.0, 2210.0, 2350.0, 2380.5),
        (None, 990.0, 1000.0, 1320.0, 1600.0, 1650.2),
    )
)


def test_the_shipped_thresholds_are_read_from_the_configuration() -> None:
    t = report_status.load_thresholds()
    assert t == report_status.Thresholds.from_config(CONFIG["status_thresholds"])
    assert builder._THRESH == t  # the builder evaluates against the same file


def test_the_builder_keeps_the_reports_tab_flags_and_status() -> None:
    """Every combination of the grid (1,512 points, edges included): the same status and the same flag texts."""
    for etaco, fuel, raft, perm in GRID:
        expected = legacy_status(None if etaco is not None and math.isnan(etaco) else etaco, fuel, raft, perm)
        assert builder._status(etaco, fuel, raft, perm) == expected, (etaco, fuel, raft, perm)


def test_the_kpi_levels_agree_with_the_report_status() -> None:
    t = report_status.load_thresholds(CONFIG)
    for etaco, fuel, raft, perm in GRID:
        checks = report_status.evaluate(etaco=etaco, fuel_rate=fuel, raft=raft, perm=perm, thresholds=t)
        status, flags = report_status.overall(checks)
        assert [c.flag for c in checks if c.level in ("warning", "critical")] == flags
        assert all((c.level == "no_data") == (c.value is None) for c in checks)
        assert status == builder._status(etaco, fuel, raft, perm)[0]


def test_levels_edges_and_missing_values() -> None:
    t = report_status.load_thresholds(CONFIG)
    by_key = {
        c.key: c for c in report_status.evaluate(etaco=39.99, fuel_rate=530.0, raft=None, perm=1600.01, thresholds=t)
    }
    assert by_key["etaco"].level == "critical" and by_key["etaco"].flag == "ETA CO 39.99% (critical < 40%)"
    assert by_key["fuel_rate"].level == "ok" and by_key["fuel_rate"].flag is None  # 530 is not above 530
    assert by_key["raft"].level == "no_data" and by_key["raft"].value is None
    assert by_key["permeability"].level == "warning" and by_key["permeability"].flag.startswith("Permeability 1600.01")
    assert by_key["etaco"].rule == "warning below 42 %, critical below 40 %"
    assert by_key["fuel_rate"].rule == "warning above 530 kg/tHM, critical above 570 kg/tHM"
    assert by_key["raft"].rule == "normal 2100–2350 °C" and by_key["permeability"].rule == "normal 1000–1600"
    assert (by_key["etaco"].warn, by_key["etaco"].crit, by_key["raft"].lo, by_key["raft"].hi) == (42, 40, 2100, 2350)


def test_a_changed_configuration_changes_the_levels_and_the_texts(monkeypatch) -> None:
    """Thresholds read from configuration: the service reads the file at every call."""
    import config.config_loader as loader

    stricter = {"status_thresholds": {**CONFIG["status_thresholds"], "etaco": {"warn": 47.5, "crit": 45.0}}}
    data = {"etaco": {"mean": 46.3, "std": 0.8}, "fuel_rate": 528.4, "raft": {"mean": 2210.0}, "permeability": {}}
    assert [c["level"] for c in reports.kpi_checks(data)] == ["ok", "ok", "ok", "no_data"]
    monkeypatch.setattr(loader, "load_config", lambda name="": stricter if name == "shift_report.yml" else {})
    checks = reports.kpi_checks(data)
    assert checks[0]["level"] == "warning" and checks[0]["flag"] == "ETA CO 46.30% (< 47.5% normal)"
    assert reports.thresholds_data()["etaco_warn_pct"] == 47.5 and reports.thresholds_data()["unstable_from"] == 3


def test_incomplete_thresholds_are_refused() -> None:
    with pytest.raises(ValueError, match="status_thresholds"):
        report_status.load_thresholds({"status_thresholds": {"etaco": {"warn": 42.0}}})


def test_the_detail_carries_the_kpis_of_its_data() -> None:
    row = {
        "id": "8f6c1f1e-6d3f-4a35-9b1e-0b0a3f0c9a11",
        "shift_key": "2026-10-27_A",
        "narrative_status": "ok",
        "data": {"etaco": {"mean": 39.2, "std": 1.0}, "fuel_rate": 575.0, "raft": {"mean": 2400.0, "std": 20.0},
                 "permeability": {"mean": None, "std": None}},
    }  # fmt: skip
    detail = reports.detail_data(row, report_status.load_thresholds(CONFIG))
    assert [(k["key"], k["level"]) for k in detail["kpis"]] == [
        ("etaco", "critical"),
        ("fuel_rate", "critical"),
        ("raft", "warning"),
        ("permeability", "no_data"),
    ]
    statuses = report_status.overall(
        [report_status.KpiCheck(**{**k}) for k in detail["kpis"]]  # the view round-trips into checks
    )
    assert statuses[0] == "UNSTABLE"
