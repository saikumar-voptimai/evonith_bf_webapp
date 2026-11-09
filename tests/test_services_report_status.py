"""KPI status of the shift reports (SV-2.4, #296): thresholds read from config/shift_report.yml, one rule for the
report builder and the twin-api's KPI view, with the Reports tab's flag texts unchanged. Thresholds of 3 Nov 2026 (set
from the Jun-Nov 2026 shift record); hot-metal Si, hot-metal temperature and slag basicity are checked as well."""

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
        "etaco": {"warn": 41.0, "crit": 40.0},
        "fuel_rate": {"warn": 570.0, "crit": 590.0},
        "raft_lo": 2100.0,
        "raft_hi": 2350.0,
        "perm_lo": 1250.0,
        "perm_hi": 1500.0,
        "hm_si": {"lo": 0.25, "hi": 0.60, "crit_lo": 0.20, "crit_hi": 0.80},
        "hm_temp_lo": 1440.0,
        "hm_temp_hi": 1520.0,
        "basicity_lo": 1.05,
        "basicity_hi": 1.15,
    }
}


def reference_status(
    etaco: Any,
    fuel_rate: Any,
    raft: Any,
    perm: Any,
    hm_si: Any = None,
    hm_temp: Any = None,
    basicity: Any = None,
) -> tuple[str, list[str]]:
    """The Reports tab's rule written out: its flag texts, the thresholds of CONFIG."""
    flags: list[str] = []
    if etaco is not None:
        if etaco < 40.0:
            flags.append(f"ETA CO {etaco:.2f}% (critical < 40%)")
        elif etaco < 41.0:
            flags.append(f"ETA CO {etaco:.2f}% (< 41% normal)")
    if fuel_rate is not None:
        if fuel_rate > 590.0:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (critical > 590)")
        elif fuel_rate > 570.0:
            flags.append(f"Fuel Rate {fuel_rate:.2f} kg/tHM (> 570 attention)")
    if raft is not None and (raft < 2100.0 or raft > 2350.0):
        flags.append(f"RAFT {raft:.2f}°C (normal 2100–2350)")
    if perm is not None and (perm < 1250.0 or perm > 1500.0):
        flags.append(f"Permeability {perm:.2f} (normal 1250–1500)")
    if hm_si is not None:
        if hm_si < 0.2:
            flags.append(f"HM Si {hm_si:.2f}% (critical < 0.2%)")
        elif hm_si > 0.8:
            flags.append(f"HM Si {hm_si:.2f}% (critical > 0.8%)")
        elif hm_si < 0.25 or hm_si > 0.6:
            flags.append(f"HM Si {hm_si:.2f}% (normal 0.25–0.6)")
    if hm_temp is not None and (hm_temp < 1440.0 or hm_temp > 1520.0):
        flags.append(f"HM temperature {hm_temp:.2f}°C (normal 1440–1520)")
    if basicity is not None and (basicity < 1.05 or basicity > 1.15):
        flags.append(f"Slag basicity {basicity:.2f} (normal 1.05–1.15)")
    n = len(flags)
    return ("STABLE" if n == 0 else "ATTENTION REQUIRED" if n <= 2 else "UNSTABLE"), flags


# The four KPIs of the tab, edges included (the other three absent).
GRID = list(
    itertools.product(
        (None, 39.5, 40.0, 40.8, 41.0, 42.3, math.nan),
        (None, 548.7, 570.0, 575.3, 590.0, 591.2),
        (None, 2050.0, 2100.0, 2210.0, 2350.0, 2380.5),
        (None, 1240.0, 1250.0, 1320.0, 1500.0, 1520.4),
    )
)
# The three hot-metal and slag KPIs, edges included, with the four above at normal values.
NORMAL = (42.3, 555.0, 2250.0, 1350.0)
GRID_HM = list(
    itertools.product(
        (None, 0.19, 0.2, 0.24, 0.25, 0.43, 0.6, 0.61, 0.8, 0.81, math.nan),
        (None, 1439.9, 1440.0, 1481.0, 1520.0, 1520.1),
        (None, 1.04, 1.05, 1.09, 1.15, 1.151),
    )
)


def _value(x: Any) -> Any:
    return None if x is not None and isinstance(x, float) and math.isnan(x) else x


def test_the_shipped_thresholds_are_read_from_the_configuration() -> None:
    t = report_status.load_thresholds()
    assert t == report_status.Thresholds.from_config(CONFIG["status_thresholds"])
    assert builder._THRESH == t  # the builder evaluates against the same file


def test_the_builder_keeps_the_reports_tab_flags_and_status() -> None:
    """Every combination of the grids (1,512 + 396 points, edges included): the same status and flag texts."""
    for etaco, fuel, raft, perm in GRID:
        expected = reference_status(_value(etaco), fuel, raft, perm)
        assert builder._status(etaco, fuel, raft, perm) == expected, (etaco, fuel, raft, perm)
    for si, hmt, b2 in GRID_HM:
        expected = reference_status(*NORMAL, _value(si), hmt, b2)
        assert builder._status(*NORMAL, si, hmt, b2) == expected, (si, hmt, b2)


def test_the_kpi_levels_agree_with_the_report_status() -> None:
    t = report_status.load_thresholds(CONFIG)
    points = [(e, f, r, p, None, None, None) for e, f, r, p in GRID] + [(*NORMAL, s, h, b) for s, h, b in GRID_HM]
    for etaco, fuel, raft, perm, si, hmt, b2 in points:
        checks = report_status.evaluate(
            etaco=etaco, fuel_rate=fuel, raft=raft, perm=perm, thresholds=t, hm_si=si, hm_temp=hmt, basicity=b2
        )
        status, flags = report_status.overall(checks)
        assert [c.flag for c in checks if c.level in ("warning", "critical")] == flags
        assert all((c.level == "no_data") == (c.value is None) for c in checks)
        assert status == builder._status(etaco, fuel, raft, perm, si, hmt, b2)[0]


def test_levels_edges_and_missing_values() -> None:
    t = report_status.load_thresholds(CONFIG)
    by_key = {
        c.key: c
        for c in report_status.evaluate(
            etaco=39.99,
            fuel_rate=570.0,
            raft=None,
            perm=1500.01,
            thresholds=t,
            hm_si=0.81,
            hm_temp=1432.5,
            basicity=1.09,
        )
    }
    assert list(by_key) == ["etaco", "fuel_rate", "raft", "permeability", "hm_si", "hm_temp", "slag_basicity"]
    assert by_key["etaco"].level == "critical" and by_key["etaco"].flag == "ETA CO 39.99% (critical < 40%)"
    assert by_key["fuel_rate"].level == "ok" and by_key["fuel_rate"].flag is None  # 570 is not above 570
    assert by_key["raft"].level == "no_data" and by_key["raft"].value is None
    assert by_key["permeability"].level == "warning" and by_key["permeability"].flag.startswith("Permeability 1500.01")
    assert by_key["hm_si"].level == "critical" and by_key["hm_si"].flag == "HM Si 0.81% (critical > 0.8%)"
    assert (
        by_key["hm_temp"].level == "warning" and by_key["hm_temp"].flag == "HM temperature 1432.50°C (normal 1440–1520)"
    )
    assert by_key["slag_basicity"].level == "ok" and by_key["slag_basicity"].unit == ""
    assert by_key["etaco"].rule == "warning below 41 %, critical below 40 %"
    assert by_key["fuel_rate"].rule == "warning above 570 kg/tHM, critical above 590 kg/tHM"
    assert by_key["raft"].rule == "normal 2100–2350 °C" and by_key["permeability"].rule == "normal 1250–1500"
    assert by_key["hm_si"].rule == "normal 0.25–0.6 %, critical below 0.2 % or above 0.8 %"
    assert by_key["hm_temp"].rule == "normal 1440–1520 °C" and by_key["slag_basicity"].rule == "normal 1.05–1.15"
    assert (by_key["etaco"].warn, by_key["etaco"].crit, by_key["raft"].lo, by_key["raft"].hi) == (41, 40, 2100, 2350)
    si = by_key["hm_si"]
    assert (si.lo, si.hi, si.warn, si.crit) == (0.25, 0.6, 0.2, 0.8)  # normal band, then the critical band's edges
    low_si = report_status.evaluate(etaco=None, fuel_rate=None, raft=None, perm=None, thresholds=t, hm_si=0.24)[4]
    assert low_si.level == "warning" and low_si.flag == "HM Si 0.24% (normal 0.25–0.6)"


def test_a_changed_configuration_changes_the_levels_and_the_texts(monkeypatch) -> None:
    """Thresholds read from configuration: the service reads the file at every call."""
    import config.config_loader as loader

    stricter = {"status_thresholds": {**CONFIG["status_thresholds"], "etaco": {"warn": 47.5, "crit": 45.0}}}
    data = {"etaco": {"mean": 46.3, "std": 0.8}, "fuel_rate": 528.4, "raft": {"mean": 2210.0}, "permeability": {}}
    assert [c["level"] for c in reports.kpi_checks(data)] == ["ok", "ok", "ok"] + ["no_data"] * 4
    monkeypatch.setattr(loader, "load_config", lambda name="": stricter if name == "shift_report.yml" else {})
    checks = reports.kpi_checks(data)
    assert checks[0]["level"] == "warning" and checks[0]["flag"] == "ETA CO 46.30% (< 47.5% normal)"
    assert reports.thresholds_data()["etaco_warn_pct"] == 47.5 and reports.thresholds_data()["unstable_from"] == 3
    assert reports.thresholds_data()["hm_si_crit_hi_pct"] == 0.8


def test_incomplete_thresholds_are_refused() -> None:
    with pytest.raises(ValueError, match="status_thresholds"):
        report_status.load_thresholds({"status_thresholds": {"etaco": {"warn": 42.0}}})
    without_hot_metal = {k: v for k, v in CONFIG["status_thresholds"].items() if not k.startswith(("hm_", "basicity"))}
    with pytest.raises(ValueError, match="status_thresholds"):  # the hot-metal and slag bands are required
        report_status.load_thresholds({"status_thresholds": without_hot_metal})


def test_the_detail_carries_the_kpis_of_its_data() -> None:
    row = {
        "id": "8f6c1f1e-6d3f-4a35-9b1e-0b0a3f0c9a11",
        "shift_key": "2026-10-27_A",
        "narrative_status": "ok",
        "data": {"etaco": {"mean": 39.2, "std": 1.0}, "fuel_rate": 575.0, "raft": {"mean": 2400.0, "std": 20.0},
                 "permeability": {"mean": None, "std": None}, "hm_si": 0.85, "hm_temp": 1455.0, "slag_basicity": 1.02},
    }  # fmt: skip
    detail = reports.detail_data(row, report_status.load_thresholds(CONFIG))
    assert [(k["key"], k["level"]) for k in detail["kpis"]] == [
        ("etaco", "critical"),
        ("fuel_rate", "warning"),
        ("raft", "warning"),
        ("permeability", "no_data"),
        ("hm_si", "critical"),
        ("hm_temp", "ok"),
        ("slag_basicity", "warning"),
    ]
    statuses = report_status.overall(
        [report_status.KpiCheck(**{**k}) for k in detail["kpis"]]  # the view round-trips into checks
    )
    assert statuses[0] == "UNSTABLE"
    # A report stored before the hot-metal and slag checks existed: those KPIs read no_data, never a flag.
    old = {**row, "data": {k: v for k, v in row["data"].items() if k not in ("hm_si", "hm_temp", "slag_basicity")}}
    levels = [k["level"] for k in reports.detail_data(old, report_status.load_thresholds(CONFIG))["kpis"]]
    assert levels[4:] == ["no_data"] * 3
