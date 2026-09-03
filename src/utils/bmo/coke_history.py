"""Daily predicted-vs-realised history for coke rate and hot-metal silicon.

WHY THIS EXISTS.

Three things in the Blend Optimizer need the same underlying frame:

  * the retrain button, which refits the coke-rate bias offset on recent history
  * the predicted-vs-realised coke rate chart
  * the predicted-vs-realised silicon chart

Building it once, here, keeps them consistent. Before this, the only assembler
lived in ``scripts/energy_balance_phase0.py``, which the app cannot import.

WHAT "PREDICTED" MEANS FOR EACH.

Coke rate comes from inverting the energy balance: given the burden actually
charged, the blast actually blown and the PCI actually injected, what coke rate
closes the heat balance? It uses NO knowledge of the coke actually charged, so
comparing it against the charge reports is a fair test.

Silicon comes from the shipped Si model bundle, scored on the historical feature
rows it was trained to consume.

WHY THE CALIBRATION MUST BE REFRESHED OFTEN.

The energy balance has the right shape and the wrong level, and the level drifts
by about 3.3 kg/tHM per month. Measured over 281 days
(scripts/coke_calibration_cadence.py):

    held for     MAE   MAPE%      R2
        0 d     13.9    4.59   +0.428
       30 d     18.9    6.34   +0.054
       90 d     33.0   11.12   -1.232

A quarter-old calibration is worse than no calibration at all, which is why the
page refreshes automatically rather than waiting to be asked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Fields the energy balance needs from the hourly static ML dataset, mapped to
# the names EnergyBalanceInputs expects.
_STATIC_FIELDS = {
    "HOT BLAST VOLUMENM3/HR.": "blast_volume_nm3_per_hr",
    "HOT BLAST TEMP.OC": "blast_temperature_c",
    "O2 ENRICHMENT %": "oxygen_enrichment_pct",
    "FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%": "top_gas_co_pct",
    "FURNACE TOP GAS ANALYSISCO2%": "top_gas_co2_pct",
    "FURNACE TOP GAS ANALYSISH2%": "top_gas_h2_pct",
    "CHEM_PCT_C": "hm_carbon_pct",
    "CHEM_PCT_FE": "hm_iron_pct",
    "CHEM_PCT_SI": "hm_silicon_pct",
    "CHEM_PCT_MN": "hm_manganese_pct",
    "SLAG_PCT_FEO": "slag_feo_pct",
    "FLUX_LOI%": "flux_loi_pct",
    "PRODUCTIONTONNESPERHR": "production_per_hour",
    "COKE RATE KG/THM": "coke_setpoint_kg_thm",
    "PCI_KG/THM": "pci_kg_thm",
}
_MOISTURE_FIELDS = {
    "ORE_TM%": "ore", "PELLET_PCT_TM": "pellet", "FLUX_TM%": "flux",
    "COKE_MOIST%": "coke", "NUTCOKE_MOIST%": "nut_coke",
}
# Top gas temperature is not in every export of the static dataset, and the
# fallback is expensive: the balance moves 2.4 kg/THM per 20 C of it, so a
# constant costs about 0.12 kg/THM for every degree it is wrong by.
#
# FTG_UPTAKE_TEMP_AVG is the uptake temperature and IS present, averaging 191 C
# with sd 30. Defaulting to 140 C instead of reading it under-predicted the coke
# rate by roughly 6 kg/THM and made this pipeline's fitted offset disagree with
# the shipped one by that much. Read the column.
_TOP_TEMP_CANDIDATES = (
    "TOP TEMPERATUREOC", "TOP_TEMP_AVG", "top_temp_avg", "FTG_UPTAKE_TEMP_AVG",
)
_DEFAULT_TOP_TEMP_C = 140.0


@dataclass
class HistoryResult:
    """Daily history plus a note on anything that had to be defaulted."""

    frame: pd.DataFrame
    warnings: list[str]

    @property
    def usable_days(self) -> int:
        if self.frame.empty or "predicted_coke" not in self.frame:
            return 0
        return int(self.frame[["predicted_coke", "actual_coke"]].dropna().shape[0])


def _window(days: int) -> tuple[datetime, datetime]:
    """Explicit UTC range. The offline presets stop at three months."""

    end = datetime.now(timezone.utc)
    return end - timedelta(days=days + 2), end


def _static_daily(days: int) -> tuple[pd.DataFrame, list[str]]:
    """Daily means of the hourly static ML dataset."""

    from data.ml.static_csv import load_static_dataset

    warnings: list[str] = []
    raw = load_static_dataset()
    if raw is None or raw.empty:
        return pd.DataFrame(), ["static ML dataset unavailable"]

    frame = raw.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        time_col = next((c for c in frame.columns if c.lower() in ("time", "date_time")), None)
        if time_col is None:
            return pd.DataFrame(), ["static dataset has no time index"]
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[time_col]).set_index(time_col)

    out = pd.DataFrame(index=frame.index)
    for source, target in _STATIC_FIELDS.items():
        if source in frame.columns:
            out[target] = pd.to_numeric(frame[source], errors="coerce")
        else:
            out[target] = np.nan
            warnings.append(f"static column missing: {source}")
    for source, key in _MOISTURE_FIELDS.items():
        out[f"moist_{key}"] = (
            pd.to_numeric(frame[source], errors="coerce")
            if source in frame.columns else np.nan
        )
    top = next((c for c in _TOP_TEMP_CANDIDATES if c in frame.columns), None)
    out["top_gas_temperature_c"] = (
        pd.to_numeric(frame[top], errors="coerce") if top
        else _DEFAULT_TOP_TEMP_C
    )
    if top is None:
        warnings.append(
            f"top gas temperature absent; assumed {_DEFAULT_TOP_TEMP_C:.0f} C"
        )

    daily = out.resample("1D").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index.date)
    cutoff = daily.index.max() - pd.Timedelta(days=days)
    return daily[daily.index >= cutoff], warnings


def _charge_daily(days: int) -> pd.DataFrame:
    """Charged tonnes per day from the charge reports.

    Charge reports, not DPR: DPR under-reports coke by about 13%, and the static
    CSV's COKE_CALC_MT correlates only +0.16 with actual dumps.
    """

    from furnace_data.offline import fetch_offline_data

    # An explicit range, not a preset: the presets stop at "last 3 months" and
    # the calibration window alone is 90 days.
    raw = fetch_offline_data("charge_data", time_range=_window(days),
                             query_type="raw")
    if raw is None or raw.empty:
        return pd.DataFrame()

    def total(prefix: str, count: int) -> pd.Series:
        cols = [f"{prefix}_{i}_mt" for i in range(1, count + 1)]
        present = [c for c in cols if c in raw.columns]
        if not present:
            return pd.Series(0.0, index=raw.index)
        return raw[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)

    frame = pd.DataFrame({
        "coke_mt": total("coke", 2),
        "nut_coke_mt": total("nut_coke", 2),
        "flux_mt": total("flux", 3),
        "sinter_mt": total("sinter", 4),
        "ore_mt": total("ore", 12),
        "pellet_mt": total("pellet", 2),
    }, index=raw.index)
    daily = frame.resample("1D").sum(numeric_only=True)
    daily.index = pd.to_datetime(daily.index.tz_localize(None).date)
    # A partial day of charge reports would look like a low-coke day.
    charges = frame.resample("1D").size()
    charges.index = daily.index
    return daily[charges.between(120, 190)]


def _dpr_daily(days: int) -> pd.DataFrame:
    """Hot metal, slag, PCI and the two top-leaving dust streams."""

    from furnace_data.offline import fetch_offline_data

    raw = fetch_offline_data("dpr_data", time_range=_window(days),
                             query_type="raw")
    if raw is None or raw.empty:
        return pd.DataFrame()
    wanted = {
        "total_hot_metal_mt": "hot_metal_mt", "slag_generation_mt": "slag_mt",
        "pci_mt": "pci_mt", "flue_dust_mt": "flue_dust_mt",
        "gcp_dust_mt": "gcp_dust_mt",
    }
    frame = pd.DataFrame(index=raw.index)
    for source, target in wanted.items():
        frame[target] = (
            pd.to_numeric(raw[source], errors="coerce")
            if source in raw.columns else np.nan
        )
    daily = frame.resample("1D").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index.tz_localize(None).date)
    return daily[daily["hot_metal_mt"].between(1200, 3200)]


def _shell_loss_daily(days: int) -> pd.Series:
    """Stave heat load in GJ/hr, rows 6-10.

    The tags read in MW, so x3.6 gives GJ/hr. The x3600 "GW.hr" conversion that
    appears elsewhere is a thousand times too large.

    NOTE this is the stave-only figure, not scaled to all cooling circuits. Which
    basis is correct is an open question worth 11% on the coke rate - see
    docs/energy_balance_findings_and_open_decisions.md section 5.
    """

    try:
        from furnace_data.influx.online import fetch_online_df

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 2)
        raw = fetch_online_df(
            selected_measurements=["heatload_delta_t"], time_range="last 1 week",
            request_type="windowed-average", window_by="1 hour",
            start_time_override=start, end_time_override=end,
            column_naming="field",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("shell loss unavailable: %s", exc)
        return pd.Series(dtype=float)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    quads = [f"heat_load_r{r}_q{q}" for r in range(6, 11) for q in range(1, 5)]
    have = [c for c in quads if c in raw.columns]
    if not have:
        return pd.Series(dtype=float)
    total_mw = raw[have].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    total_mw = total_mw.where(total_mw.between(2.0, 12.0))
    daily = (total_mw * 3.6).resample("1D").mean()
    daily.index = pd.to_datetime(daily.index.tz_localize(None).date)
    return daily


def build_daily_history(days: int = 120) -> HistoryResult:
    """Assemble daily actuals and run the energy balance against each day.

    Args:
         - days: int - Lookback in days.

    Returns:
         - return HistoryResult - Frame indexed by date with predicted_coke,
           actual_coke, actual_si, pci_kg_thm, nut_coke_kg_thm and the residual,
           plus warnings about anything defaulted.
    """

    from utils.energy_balance import EnergyBalanceInputs
    from utils.energy_balance.constants import load_config
    from utils.energy_balance.solve import solve_coke_rate_kg_per_thm

    static, warnings = _static_daily(days)
    charge, dpr = _charge_daily(days), _dpr_daily(days)
    if static.empty or charge.empty or dpr.empty:
        missing = [n for n, f in (("static", static), ("charge", charge),
                                  ("dpr", dpr)) if f.empty]
        return HistoryResult(pd.DataFrame(),
                             warnings + [f"no data from: {', '.join(missing)}"])

    frame = charge.join(dpr, how="inner").join(static, how="inner")
    shell = _shell_loss_daily(days)
    frame["shell_gj_per_hr"] = shell.reindex(frame.index) if len(shell) else np.nan
    if frame["shell_gj_per_hr"].isna().all():
        warnings.append("shell heat load unavailable; balance uses its default")

    cfg = load_config()
    rows = []
    for when, row in frame.iterrows():
        hm = float(row.get("hot_metal_mt") or 0.0)
        if hm <= 0:
            continue
        f = lambda key, default=0.0: (  # noqa: E731
            float(row[key]) if key in row and pd.notna(row[key]) else default
        )
        try:
            inputs = EnergyBalanceInputs(
                hot_metal_mt=hm, slag_mt=f("slag_mt"),
                coke_mt=f("coke_mt"), nut_coke_mt=f("nut_coke_mt"),
                pci_mt=f("pci_mt"),
                blast_volume_nm3_per_hr=f("blast_volume_nm3_per_hr"),
                blast_temperature_c=f("blast_temperature_c"),
                oxygen_enrichment_pct=f("oxygen_enrichment_pct"),
                top_gas_co_pct=f("top_gas_co_pct"),
                top_gas_co2_pct=f("top_gas_co2_pct"),
                top_gas_h2_pct=f("top_gas_h2_pct"),
                top_gas_temperature_c=f("top_gas_temperature_c", _DEFAULT_TOP_TEMP_C),
                hm_carbon_pct=f("hm_carbon_pct", 4.3),
                hm_iron_pct=f("hm_iron_pct", 94.5),
                hm_silicon_pct=f("hm_silicon_pct", 0.5),
                hm_manganese_pct=f("hm_manganese_pct", 0.2),
                slag_feo_pct=f("slag_feo_pct", 0.4),
                flue_dust_mt=f("flue_dust_mt"), gcp_dust_mt=f("gcp_dust_mt"),
                flux_mt=f("flux_mt"), sinter_mt=f("sinter_mt"),
                ore_mt=f("ore_mt"), pellet_mt=f("pellet_mt"),
                flux_loi_pct=f("flux_loi_pct", 40.0),
                fuel_vm_pct={"coke": 0.9, "nut_coke": 1.0, "pci": 19.9},
                moisture_pct={
                    key: f(f"moist_{key}") for key in
                    ("ore", "pellet", "flux", "coke", "nut_coke")
                },
                shell_loss_gj_per_hr=(
                    f("shell_gj_per_hr") if pd.notna(row.get("shell_gj_per_hr"))
                    else None
                ),
            )
            predicted = solve_coke_rate_kg_per_thm(inputs, cfg)
        except Exception:  # noqa: BLE001 - a bad day is a missing prediction
            predicted = np.nan
        rows.append({
            "date": when,
            "predicted_coke": predicted,
            "actual_coke": f("coke_mt") / hm * 1000.0,
            "actual_si": f("hm_silicon_pct", np.nan),
            "pci_kg_thm": f("pci_mt") / hm * 1000.0,
            "nut_coke_kg_thm": f("nut_coke_mt") / hm * 1000.0,
            "coke_setpoint_kg_thm": f("coke_setpoint_kg_thm", np.nan),
        })

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["residual"] = out["predicted_coke"] - out["actual_coke"]
    return HistoryResult(out, warnings)


def refit_calibration(days: int = 120, window: int = 90):
    """Refit the bias offset from recent history and persist it.

    Args:
         - days: int - History to assemble.
         - window: int - Trailing days the offset averages over.

    Returns:
         - return tuple - (CokeCalibration, HistoryResult).
    """

    from utils.bmo.coke_calibration import fit_offset, save_calibration

    history = build_daily_history(days)
    usable = history.frame[["predicted_coke", "actual_coke"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna() if not history.frame.empty else pd.DataFrame()
    if usable.empty:
        from utils.bmo.coke_calibration import NO_CALIBRATION
        return NO_CALIBRATION, history

    recent = usable.tail(window)
    calibration = fit_offset(
        recent["predicted_coke"].tolist(), recent["actual_coke"].tolist(),
        window_days=window,
        days=[d.date().isoformat() if hasattr(d, "date") else str(d)
              for d in recent.index],
    )
    save_calibration(calibration)
    return calibration, history
