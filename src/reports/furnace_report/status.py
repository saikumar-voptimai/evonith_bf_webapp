"""Rule-based status of a shift report against ``config/shift_report.yml -> status_thresholds`` (SV-2.4, #296).

One check per KPI that has thresholds (ETA CO, fuel rate, RAFT, permeability, hot-metal Si, hot-metal temperature,
slag basicity): ``evaluate`` gives each its level (ok, warning, critical; no_data without a value) and, when it
breaches, the Reports tab's flag text; ``overall`` counts the flags (none STABLE, one or two ATTENTION REQUIRED, three
or more UNSTABLE). The report builder (``builder._status``) and the twin-api's KPI view
(``services.reports.kpi_checks``) both evaluate through this module, so the stored status and the levels on the Shift
Reports page follow one rule. The thresholds were set from the plant's Jun-Nov 2026 shift record (the YAML says how).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Level = Literal["ok", "warning", "critical", "no_data"]
FurnaceStatus = Literal["STABLE", "ATTENTION REQUIRED", "UNSTABLE"]

#: Flags from which the status is ATTENTION REQUIRED, and from which it is UNSTABLE, count.
ATTENTION_FROM = 1
UNSTABLE_FROM = 3
CONFIG_FILE = "shift_report.yml"


@dataclass(frozen=True)
class Thresholds:
    """The thresholds of ``status_thresholds``: ETA CO in %, fuel rate in kg/tHM, RAFT in °C, permeability index,
    hot-metal Si in % (normal band and critical band), hot-metal temperature in °C, slag basicity CaO/SiO₂."""

    etaco_warn: float
    etaco_crit: float
    fuel_rate_warn: float
    fuel_rate_crit: float
    raft_lo: float
    raft_hi: float
    perm_lo: float
    perm_hi: float
    si_lo: float
    si_hi: float
    si_crit_lo: float
    si_crit_hi: float
    hm_temp_lo: float
    hm_temp_hi: float
    basicity_lo: float
    basicity_hi: float

    @classmethod
    def from_config(cls, block: Mapping[str, Any]) -> Thresholds:
        try:
            return cls(
                etaco_warn=float(block["etaco"]["warn"]),
                etaco_crit=float(block["etaco"]["crit"]),
                fuel_rate_warn=float(block["fuel_rate"]["warn"]),
                fuel_rate_crit=float(block["fuel_rate"]["crit"]),
                raft_lo=float(block["raft_lo"]),
                raft_hi=float(block["raft_hi"]),
                perm_lo=float(block["perm_lo"]),
                perm_hi=float(block["perm_hi"]),
                si_lo=float(block["hm_si"]["lo"]),
                si_hi=float(block["hm_si"]["hi"]),
                si_crit_lo=float(block["hm_si"]["crit_lo"]),
                si_crit_hi=float(block["hm_si"]["crit_hi"]),
                hm_temp_lo=float(block["hm_temp_lo"]),
                hm_temp_hi=float(block["hm_temp_hi"]),
                basicity_lo=float(block["basicity_lo"]),
                basicity_hi=float(block["basicity_hi"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{CONFIG_FILE} -> status_thresholds is incomplete or not numeric: {exc!r}") from exc


def load_thresholds(config: Mapping[str, Any] | None = None) -> Thresholds:
    """The thresholds of ``config`` (default: ``config/shift_report.yml``, read now)."""
    if config is None:
        from config.config_loader import load_config

        config = load_config(CONFIG_FILE) or {}
    return Thresholds.from_config(config.get("status_thresholds") or {})


@dataclass(frozen=True)
class KpiCheck:
    """One KPI against its thresholds; ``flag`` is the Reports tab's text when the KPI breaches."""

    key: Literal["etaco", "fuel_rate", "raft", "permeability", "hm_si", "hm_temp", "slag_basicity"]
    label: str
    unit: str
    value: float | None
    level: Level
    rule: str
    flag: str | None = None
    warn: float | None = None
    crit: float | None = None
    lo: float | None = None
    hi: float | None = None


def _g(value: float) -> str:
    return f"{value:g}"


def _finite(value: Any) -> float | None:
    """A number, or None for a missing, non-numeric or non-finite value (no data rather than a flag)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _etaco(value: float | None, t: Thresholds) -> KpiCheck:
    rule = f"warning below {_g(t.etaco_warn)} %, critical below {_g(t.etaco_crit)} %"
    level: Level = "no_data"
    flag = None
    if value is not None:
        level = "ok"
        if value < t.etaco_crit:
            level, flag = "critical", f"ETA CO {value:.2f}% (critical < {_g(t.etaco_crit)}%)"
        elif value < t.etaco_warn:
            level, flag = "warning", f"ETA CO {value:.2f}% (< {_g(t.etaco_warn)}% normal)"
    return KpiCheck("etaco", "ETA CO", "%", value, level, rule, flag, warn=t.etaco_warn, crit=t.etaco_crit)


def _fuel_rate(value: float | None, t: Thresholds) -> KpiCheck:
    rule = f"warning above {_g(t.fuel_rate_warn)} kg/tHM, critical above {_g(t.fuel_rate_crit)} kg/tHM"
    level: Level = "no_data"
    flag = None
    if value is not None:
        level = "ok"
        if value > t.fuel_rate_crit:
            level, flag = "critical", f"Fuel Rate {value:.2f} kg/tHM (critical > {_g(t.fuel_rate_crit)})"
        elif value > t.fuel_rate_warn:
            level, flag = "warning", f"Fuel Rate {value:.2f} kg/tHM (> {_g(t.fuel_rate_warn)} attention)"
    return KpiCheck(
        "fuel_rate", "Fuel rate", "kg/tHM", value, level, rule, flag, warn=t.fuel_rate_warn, crit=t.fuel_rate_crit
    )


def _band(
    key: Literal["raft", "permeability", "hm_temp", "slag_basicity"],
    label: str,
    unit: str,
    value: float | None,
    lo: float,
    hi: float,
) -> KpiCheck:
    """Outside [lo, hi]: warning, with the tab's flag (``RAFT 2050.00°C (normal 2100–2350)``)."""
    rule = f"normal {_g(lo)}–{_g(hi)}{f' {unit}' if unit else ''}"
    if value is None:
        return KpiCheck(key, label, unit, None, "no_data", rule, lo=lo, hi=hi)
    if lo <= value <= hi:
        return KpiCheck(key, label, unit, value, "ok", rule, lo=lo, hi=hi)
    flag = f"{label} {value:.2f}{unit} (normal {_g(lo)}–{_g(hi)})"
    return KpiCheck(key, label, unit, value, "warning", rule, flag, lo=lo, hi=hi)


def _hm_si(value: float | None, t: Thresholds) -> KpiCheck:
    """Hot-metal Si, the thermal-state proxy: outside the normal band a warning, outside the critical band (a chilled
    hearth below, a hot and coke-hungry furnace above) critical. ``lo`` / ``hi`` carry the normal band, ``warn`` /
    ``crit`` the critical band's low and high edge."""
    rule = f"normal {_g(t.si_lo)}–{_g(t.si_hi)} %, critical below {_g(t.si_crit_lo)} % or above {_g(t.si_crit_hi)} %"
    level: Level = "no_data"
    flag = None
    if value is not None:
        level = "ok"
        if value < t.si_crit_lo:
            level, flag = "critical", f"HM Si {value:.2f}% (critical < {_g(t.si_crit_lo)}%)"
        elif value > t.si_crit_hi:
            level, flag = "critical", f"HM Si {value:.2f}% (critical > {_g(t.si_crit_hi)}%)"
        elif not t.si_lo <= value <= t.si_hi:
            level, flag = "warning", f"HM Si {value:.2f}% (normal {_g(t.si_lo)}–{_g(t.si_hi)})"
    return KpiCheck(
        "hm_si", "HM Si", "%", value, level, rule, flag, warn=t.si_crit_lo, crit=t.si_crit_hi, lo=t.si_lo, hi=t.si_hi
    )


def evaluate(
    *,
    etaco: float | None,
    fuel_rate: float | None,
    raft: float | None,
    perm: float | None,
    thresholds: Thresholds,
    hm_si: float | None = None,
    hm_temp: float | None = None,
    basicity: float | None = None,
) -> tuple[KpiCheck, ...]:
    """ETA CO and fuel rate (warning, critical), RAFT, permeability, hot-metal temperature and slag basicity (outside
    the normal band: warning), hot-metal Si (normal band: warning; critical band: critical); a missing or non-finite
    value is no_data."""
    t = thresholds
    return (
        _etaco(_finite(etaco), t),
        _fuel_rate(_finite(fuel_rate), t),
        _band("raft", "RAFT", "°C", _finite(raft), t.raft_lo, t.raft_hi),
        _band("permeability", "Permeability", "", _finite(perm), t.perm_lo, t.perm_hi),
        _hm_si(_finite(hm_si), t),
        _band("hm_temp", "HM temperature", "°C", _finite(hm_temp), t.hm_temp_lo, t.hm_temp_hi),
        _band("slag_basicity", "Slag basicity", "", _finite(basicity), t.basicity_lo, t.basicity_hi),
    )


def overall(checks: tuple[KpiCheck, ...] | list[KpiCheck]) -> tuple[FurnaceStatus, list[str]]:
    """The report's status and flags: none STABLE, ``ATTENTION_FROM`` up to ``UNSTABLE_FROM`` - 1 ATTENTION REQUIRED,
    from ``UNSTABLE_FROM`` on UNSTABLE."""
    flags = [c.flag for c in checks if c.flag]
    if len(flags) >= UNSTABLE_FROM:
        return "UNSTABLE", flags
    if len(flags) >= ATTENTION_FROM:
        return "ATTENTION REQUIRED", flags
    return "STABLE", flags


__all__ = [
    "ATTENTION_FROM",
    "UNSTABLE_FROM",
    "KpiCheck",
    "Thresholds",
    "evaluate",
    "load_thresholds",
    "overall",
]
