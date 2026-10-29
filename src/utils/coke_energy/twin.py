"""Rolling-window twin evaluation (SV-1.1, #281): the Coke Energy pipeline applied to one window W = [t - 8 h, t).

The window is aggregated by the statements that aggregate a shift (``build_shift_inputs`` with ``window_key``) and
balanced by the statements that balance a shift (``shift_balance``). Every 14-day quantity comes from the completed
shifts of a pipeline result (the pool) with the pipeline's windowing: stable shifts dated in [d - 14 d, d), at least
``min_shifts`` of them, d the date of the window's start. A window that coincides with a balanced shift therefore
returns that shift's values from the pipeline table:

    K_TRACE    coke the metered blast burnt (oxygen balance at the analyser's eta CO) plus the 14-day median of
               (charged - burnt) of the pool                                          = pipeline ``K_burnt``
    K_FORGE    two-zone balance with omega, Q_LZ, Q_UZ, r_H at their 14-day medians   = pipeline ``K_energy``
    omega_imp  two-zone calibration of W on the reconciled basis (charged coke, O-balance air, analyser eta CO)
                                                                                      = pipeline ``omega_shift``
    omega_cal  the 14-day median omega that K_FORGE uses                              = pipeline ``omega_fixed``
    K_charged  dry coke charged per tonne of hot metal                                = pipeline ``K_charged``
    K_STRATA   the K_FORGE balance with omega = omega_STRATA, STRATA 1.5 for the window's blend and operation
               against the same 14 days (``utils.burden_engine.strata_window``: anchor A = those shifts, network
               solves at the window's blend, blend + production and full operation); 1 sigma = |dK/domega| x
               sigma(omega_STRATA), dK/domega of the balance at fixed omega

Status: ``complete`` when K_TRACE, K_FORGE, omega_imp and (when requested) K_STRATA were evaluated; ``partial`` when
the window balanced but a 14-day quantity is missing (fewer than ``min_shifts`` stable shifts in the pool, or a date
before ``predict_from``) or STRATA did not evaluate; ``failed`` when the window has too little data, a known data error,
or a balance that does not solve. Pure numerics: no I/O, no Streamlit (``services.twin`` loads the sources and stores
the result).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from utils.burden_engine import strata_window
from utils.burden_engine.strata_network import solve_stats
from utils.coke_energy import burden_column, hmt_forecast
from utils.coke_energy.build_shifts import build_shift_inputs, window_key
from utils.coke_energy.burnt import k_burnt
from utils.coke_energy.pipeline import (
    PipelineResult,
    PipelineSettings,
    _case,
    _with_dates,
    build_cases,
    excluded_shifts,
    rolling_pool_median,
    run_shift_balances,
    shift_params,
    valid_shift_mask,
)
from utils.coke_energy.rist import calibrate_2z, solve_2z
from utils.coke_energy.whatif import stable_mask

WINDOW_HOURS = 8
SHIFT_START_HOURS = {"A": 6, "B": 14, "C": 22}
STATES = ("complete", "partial", "failed")
DOMEGA = 0.005  # omega step of dK/domega (fixed-omega balance)

# Window row (pipeline name) -> quantity key with its unit.
QUANTITIES = {
    "prod_th": "production_t_h",
    "hm_t": "hot_metal_t",
    "HMT": "hmt_c",
    "hm_si": "hm_si_pct",
    "hm_c": "hm_c_pct",
    "humidity_g": "blast_moisture_g_nm3",
    "eta_CO": "eta_co_pct",
    "eta_H2": "eta_h2_pct",
    "pci_kgthm_dry": "pci_kg_thm",
    "nut_kgthm_dry": "nut_coke_kg_thm",
    "fuel_kgthm_dry": "fuel_kg_thm",
    "blast_air_nm3thm": "blast_air_nm3_thm",
    "V_air_reconciled": "blast_air_reconciled_nm3_thm",
    "O2_nm3thm": "oxygen_nm3_thm",
    "HBT": "hot_blast_c",
    "T_top": "top_gas_c",
    "slag_kgthm_Al2O3tracer": "slag_kg_thm",
    "coke_kgthm_calc": "coke_wet_kg_thm",
}


@dataclass(frozen=True)
class Window:
    """[start, end) in IST, tz-naive; the date of a window is the date of its start (a C shift's date)."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not pd.Timestamp(self.start) < pd.Timestamp(self.end):
            raise ValueError(f"window start {self.start} is not before its end {self.end}")

    @classmethod
    def ending(cls, t: Any, hours: int = WINDOW_HOURS) -> Window:
        """The window of ``hours`` that ends at the full hour of ``t`` (evaluated on data up to HH:00)."""
        end = pd.Timestamp(t).floor("h")
        return cls(end - pd.Timedelta(hours=hours), end)

    @classmethod
    def of_shift(cls, skey: str) -> Window:
        """The window of shift ``YYYY-MM-DD_A|B|C`` (A 06-14, B 14-22, C 22-06 of the next day)."""
        day, letter = skey.rsplit("_", 1)
        start = pd.Timestamp(day) + pd.Timedelta(hours=SHIFT_START_HOURS[letter])
        return cls(start, start + pd.Timedelta(hours=WINDOW_HOURS))

    @property
    def key(self) -> str:
        return f"W{self.end:%Y-%m-%dT%H:%M}"

    @property
    def date(self) -> pd.Timestamp:
        return pd.Timestamp(self.start).normalize()

    @property
    def hours(self) -> float:
        return (pd.Timestamp(self.end) - pd.Timestamp(self.start)) / pd.Timedelta(hours=1)


@dataclass
class WindowEvaluation:
    """One evaluation of a window; coke rates in kg/tHM dry, omega dimensionless; None where not evaluated."""

    window: Window
    status: str
    k_trace: float | None = None
    k_forge: float | None = None
    k_charged: float | None = None
    omega_imp: float | None = None
    omega_cal: float | None = None
    k_strata: float | None = None
    k_strata_sd: float | None = None
    omega_strata: float | None = None
    omega_strata_sd: float | None = None
    quantities: dict[str, float] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _count(row: pd.Series, col: str) -> int:
    """A count column of the window row; 0 when the source had no row in the window (NaN after the join)."""
    v = _num(row.get(col))
    return int(v) if v is not None else 0


def window_inputs(
    fd: pd.DataFrame, m10: pd.DataFrame, hm: pd.DataFrame, dpr: pd.DataFrame, window: Window, history_start: str
) -> pd.DataFrame:
    """The window's input row (the columns of a shift row of ``build_shift_inputs``); empty without hourly rows.

    The whole hourly dataset from ``history_start`` is passed, not only the window's rows: the ash analyses are filled
    with the median of that history, exactly as for the shifts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_shift_inputs(
            fd, m10, hm, dpr, start=history_start, key_fn=window_key(window.start, window.end, window.key)
        )


def _invalid_reasons(row: pd.Series) -> list[str]:
    """Which of the pipeline's data checks (``valid_shift_mask``) the window fails, in words."""
    hours, ok10, taps = _count(row, "n_hours"), _count(row, "n_ok10"), _count(row, "n_taps")
    checks = [
        (hours >= 7, f"{hours} hourly rows (need 7)"),
        (ok10 >= 36, f"{ok10} valid 10-min samples (need 36)"),
        (taps >= 2, f"{taps} hot-metal taps (need 2)"),
        ((_num(row.get("PRODUCTIONTONNESPERHR")) or 0.0) / max(hours, 1) > 60, "production below 60 t/h"),
        (pd.notna(row.get("chem_pct_fe")), "no hot-metal analysis"),
        (pd.notna(row.get("slag_pct_al2o3")), "no slag analysis"),
        (pd.notna(row.get("dpr_dc_dust_kgthm")), "no DPR dust for the date"),
        ((_num(row.get("COKE_CALC_MT")) or 0.0) > 0, "no coke charged"),
    ]
    return [msg for ok, msg in checks if not bool(ok)]


def _pool_medians(
    pool: PipelineResult, target: pd.Series, frames: Mapping[str, list[str]], settings: PipelineSettings
) -> dict[str, float]:
    """14-day medians over the pool's stable shifts for one target date, as the pipeline computes them."""
    out: dict[str, float] = {}
    R = _with_dates(pool.R)
    for name, cols in frames.items():
        source = pool.B[["gap"]] if name == "burnt" else pool.Z
        Rs = R.join(source, rsuffix="_2z")
        med = rolling_pool_median(Rs[Rs.stable], target, cols, settings.window_days, settings.min_shifts)
        out.update({c: float(v) for c, v in med.iloc[0].items()})
    return out


def evaluate_window(
    fd: pd.DataFrame,
    m10: pd.DataFrame,
    hm: pd.DataFrame,
    dpr: pd.DataFrame,
    window: Window,
    pool: PipelineResult,
    settings: PipelineSettings | None = None,
    provisional_dates: Collection[date] = (),
    strata: bool = True,
    transit: burden_column.TransitReference | None = None,
) -> WindowEvaluation:
    """Evaluate ``window`` on the four source frames with the 14-day quantities of ``pool`` (see the module notes).

    ``strata=False`` skips K_STRATA (3 network solves per window and 1 per 14-day anchor, about 1 s each).
    ``transit``: reference of the burden column (``burden_column.TransitReference``); the column at the window's end
    adds the transit quantities (descent times, coke by zone)."""

    settings = settings or pool.settings
    key = window.key
    flags: dict[str, Any] = {"provisional_dpr": window.date.date() in set(provisional_dates)}
    ev = WindowEvaluation(window=window, status="failed", flags=flags)

    S = window_inputs(fd, m10, hm, dpr, window, settings.history_start)
    if S.empty or key not in S.index:
        ev.notes.append("no hourly data in the window")
        return ev
    row = S.loc[key]
    flags.update(n_hours=_count(row, "n_hours"), n_ok10=_count(row, "n_ok10"), n_taps=_count(row, "n_taps"))
    transit_q = _transit_quantities(fd, window, transit, ev)
    if not bool(valid_shift_mask(S).loc[key]):
        ev.notes.extend(_invalid_reasons(row))
        return ev
    if key in excluded_shifts(S, settings):
        ev.notes.append(
            f"data error: ore moisture {row.get('ORE_TM%'):.1f} % above {settings.max_ore_moisture_pct:g} %"
        )
        return ev

    params = shift_params(S, S.index, settings)
    p = params[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R, errors = run_shift_balances(S, S.index, None, params)
    if errors or R.empty:
        ev.notes.extend(errors or ["the window balance did not solve"])
        return ev
    rb = R.loc[key]
    cases = build_cases(S, R.index)
    flags["stable"] = bool(stable_mask(R).loc[key])
    ev.k_charged = _num(rb.coke_kgthm_dry)
    ev.quantities = {name: v for col, name in QUANTITIES.items() if (v := _num(rb.get(col))) is not None}
    ev.quantities.update(transit_q)
    ev.quantities.update({f"{g}_share_pct": v for g, v in burden_column.ferrous_shares(row).items()})
    ev.status = "partial"

    target = pd.Series([window.date], index=[key])
    med = _pool_medians(pool, target, {"burnt": ["gap"], "energy": ["omega", "Q_LZ_GJh", "Q_UZ_GJh", "r_H"]}, settings)

    # K_TRACE: the metered blast's coke plus the pool's 14-day (charged - burnt).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_raw, _, _ = k_burnt(_case(cases, key, rb.eta_H2), rb.eta_CO / 100, p)
    ev.quantities["k_burnt_raw_kg_thm"] = float(k_raw)
    offset = _num(med["gap_roll"])
    if offset is None:
        ev.notes.append(
            f"TRACE: fewer than {settings.min_shifts} stable shifts in the {settings.window_days} days "
            "before the window"
        )
    else:
        ev.k_trace = float(k_raw + offset)
        ev.quantities["burnt_offset_kg_thm"] = offset

    # omega_imp: two-zone calibration of the window on the reconciled basis.
    try:
        cal = calibrate_2z(_case(cases, key, rb.eta_H2), rb.coke_kgthm_dry, rb.V_air_reconciled, rb.eta_CO / 100, p)
        ev.omega_imp = _num(cal["omega"])
        ev.quantities.update(
            q_lz_gj_h=float(cal["Q_LZ"] * rb.prod_th / 1000),
            q_uz_gj_h=float(cal["Q_UZ"] * rb.prod_th / 1000),
            r_h=float(cal["r_H"]),
        )
    except Exception as exc:  # noqa: BLE001 - as the pipeline, a calibration that does not solve leaves omega empty
        ev.notes.append(f"omega_imp: two-zone calibration did not solve ({exc})")

    # K_FORGE: two-zone balance at the pool's 14-day medians.
    omega_roll = _num(med["omega_roll"])
    prod = float(rb.prod_th)

    def balance(omega: float, hmt: float | None = None) -> dict:
        case = _case(cases, key)
        if hmt is not None:
            case["HMT"] = float(hmt)
        return solve_2z(
            case,
            p,
            omega,
            med["Q_LZ_GJh_roll"] * 1000 / prod,
            med["Q_UZ_GJh_roll"] * 1000 / prod,
            med["r_H_roll"],
        )

    if omega_roll is None:
        ev.notes.append(
            f"FORGE: fewer than {settings.min_shifts} stable shifts in the {settings.window_days} days "
            "before the window"
        )
    elif window.date < pd.Timestamp(settings.predict_from):
        ev.notes.append(f"FORGE: window before predict_from {settings.predict_from}")
    else:
        try:
            s = balance(omega_roll)
            ev.k_forge = _num(s["K"])
            ev.omega_cal = omega_roll
            ev.quantities.update(
                eta_co_forge_pct=float(100 * s["eta"]),
                top_gas_forge_c=float(s["T_top"]),
                q_lz_cal_gj_h=float(med["Q_LZ_GJh_roll"]),
                q_uz_cal_gj_h=float(med["Q_UZ_GJh_roll"]),
                r_h_cal=float(med["r_H_roll"]),
            )
        except Exception as exc:  # noqa: BLE001 - reported, the rest of the evaluation stands
            ev.notes.append(f"FORGE: two-zone balance did not solve ({exc})")

    # HMT one descent time ahead (SV-1.4; shown only after its gate): the FORGE balance at the latest tap HMT.
    if ev.k_forge is not None:
        _hmt_forecast(ev, fd, hm, window, S.loc[key], balance)

    # K_STRATA: the same balance with omega_STRATA of the window against the same 14 days.
    if strata and ev.k_forge is not None:
        _evaluate_strata(ev, pool, S.loc[key], rb, window, settings, balance)

    if (
        ev.k_trace is not None
        and ev.k_forge is not None
        and ev.omega_imp is not None
        and (ev.k_strata is not None or not strata)
    ):
        ev.status = "complete"
    return ev


def _transit_quantities(
    fd: pd.DataFrame, window: Window, ref: burden_column.TransitReference | None, ev: WindowEvaluation
) -> dict[str, float]:
    """Descent times and coke by zone of the burden column at the window's end (a failure is a note)."""
    try:
        st = burden_column.column(fd, window.end, ref)
    except Exception as exc:  # noqa: BLE001 - reported, the balances stand
        ev.notes.append(f"transit: burden column not evaluated ({exc})")
        return {}
    if not st.complete:
        ev.notes.append("transit: the charge record does not reach the oldest charge of the column")
    ev.flags["transit_missing_hours"] = st.missing_hours
    return burden_column.quantities(st)


def _hmt_forecast(
    ev: WindowEvaluation,
    fd: pd.DataFrame,
    hm: pd.DataFrame,
    window: Window,
    row: pd.Series,
    balance: Callable[..., dict],
) -> None:
    """HMT_f(t + tau_RW) of ``hmt_forecast`` into the quantities (nothing when a part is missing)."""
    now = hmt_forecast.latest(hmt_forecast.taps(hm), window.end)
    if now is None:
        return
    hmt_now, age = now
    hm_t = _num(row.get("PRODUCTIONTONNESPERHR"))
    fer = sum(_num(row.get(c)) or 0.0 for c in ("SINTER_CALC_MT", "ORE_CALC_MT", "TOTAL_PELLET_CALC_MT"))
    karr = hmt_forecast.k_arr(fd, window.end, 1000.0 * fer / hm_t) if hm_t else None
    horizon = ev.quantities.get("tau_raceway_h")
    if karr is None or horizon is None:
        return
    omega = ev.omega_cal
    try:
        k0 = _num(balance(omega, hmt_now)["K"])
        k1 = _num(balance(omega, hmt_now + hmt_forecast.DHMT)["K"])
    except Exception as exc:  # noqa: BLE001 - reported, the evaluation stands
        ev.notes.append(f"HMT forecast: balance at the tap HMT did not solve ({exc})")
        return
    if k0 is None or k1 is None:
        return
    slope = (k1 - k0) / hmt_forecast.DHMT
    f = hmt_forecast.forecast(hmt_now, karr, k0, slope)
    ev.quantities.update(
        hmt_now_c=hmt_now,
        hmt_now_age_h=age,
        k_arr_kg_thm=karr,
        k_forge_hmt_now_kg_thm=k0,
        dk_dhmt_kg_thm_per_c=slope,
        **({"hmt_forecast_c": f, "hmt_forecast_horizon_h": float(horizon)} if f is not None else {}),
    )


def _evaluate_strata(
    ev: WindowEvaluation,
    pool: PipelineResult,
    row: pd.Series,
    rb: pd.Series,
    window: Window,
    settings: PipelineSettings,
    balance: Callable[[float], dict],
) -> None:
    """omega_STRATA, K_STRATA and their 1 sigma into ``ev`` (a failure leaves them empty with a note)."""
    R = _with_dates(pool.R)
    keys = strata_window.anchor_keys(R, R.stable, window.date, settings.window_days)
    n0, s0 = solve_stats()
    try:
        anchor = strata_window.build_anchor(pool.S, pool.R, pool.Z, keys, ev.omega_cal, window.date)
        sw = strata_window.evaluate_window(anchor, row, humidity=float(rb.humidity_g))
        k = _num(balance(sw.omega)["K"])
        k_up = _num(balance(sw.omega + DOMEGA)["K"])
    except Exception as exc:  # noqa: BLE001 - reported, TRACE and FORGE stand
        ev.notes.append(f"STRATA: not evaluated ({exc})")
        return
    if k is None or k_up is None:
        ev.notes.append("STRATA: the balance at omega_STRATA did not solve")
        return
    n1, s1 = solve_stats()
    dk = (k_up - k) / DOMEGA
    t = sw.terms
    sigma_k = abs(dk) * sw.sigma
    wet = 1 - (_num(row.get("COKE_MOIST%")) or 0.0) / 100  # the certainty tiers are set on wet coke (engine sig_K)
    ev.omega_strata, ev.omega_strata_sd = sw.omega, sw.sigma
    ev.k_strata, ev.k_strata_sd = k, sigma_k
    ev.quantities.update(
        sw.channels(),
        strata_capacity_ratio=anchor.R,
        dk_domega_kg_thm=float(dk),
        strata_anchor_coke_kg_thm=float(anchor.state.coke),
    )
    ev.flags["strata"] = {
        "anchor_shifts": len(anchor.keys),
        "network_solves": n1 - n0,  # this evaluation: the anchor (first use) and the window's new states
        "network_seconds": round(s1 - s0, 2),
        "support": t.support,
        "outside": list(t.outside),
        "investigate": list(t.investigate),
        "certainty": anchor.model.certainty(sigma_k / wet, t.support, t.investigate),
        "unmapped_ores": sorted(set(anchor.unmapped) | set(sw.unmapped)),
    }
    ev.versions.update(strata_window.versions(anchor))
