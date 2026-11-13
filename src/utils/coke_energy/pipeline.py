"""The web model's shift pipeline, run in memory.

The Claude-web scripts each read the previous step's pickle and wrote the next one. Their ``__main__`` blocks are
reproduced here, in the same order and with the same arithmetic:

    build_shifts.py      build_shift_inputs      S    shift inputs (hourly dataset + 10-min data + HM/slag + DPR)
    run_shifts.py        run_shift_balances      R    measured-gas balance and reconciled basis, every valid shift
    rolling_predict.py   oxygen_balance_model    P    eta CO fixed at the analyser's value, heat-loss allowance U from
                                                      the previous 14 days
    burnt.py             burnt_coke              B    coke the metered blast burnt, plus the calibrated (charged - burnt)
    twozone_shifts.py    calibrate_shifts,       Z    two-zone calibration of every shift (omega, Q_LZ, Q_UZ, r_H)
                         energy_balance_model    O2   omega, Q_LZ, Q_UZ, r_H fixed at their calibration

Calibration of the burnt-coke offset and of the energy model (``PipelineSettings.calibration``):

    median       the web scripts: median over the stable shifts dated in the ``window_days`` before the shift's date,
                 at least ``min_shifts`` of them (``rolling_pool_median``); the dataclass default
    exponential  the tracker (user decision 12 Nov 2026; ``coke_energy.yml``): mean over the stable shifts that ended
                 before the shift began and are at most ``window_days`` old, each weighted 0.5 ** (age / half-life),
                 age between the shifts' midpoints, at least ``tracker_min_shifts`` of them (``rolling_pool_tracker``).
                 Half-life 3 d: on the stable shifts of 2026 FORGE - TRACE SD 6.9 -> 6.4 kg/tHM, TRACE against 3-day
                 charged coke SD 4.7 -> 3.7 (claude/model_check_oct_nov_2026.md, section 6.2)

The heat-loss allowance U of the oxygen-balance model stays on the 14-day median in both modes.

``case_from_shift`` is the slow part of a shift (a wide row read field by field), and the web scripts rebuilt the same
case in every step. Here each shift's case is built once (``build_cases``) and every step gets a deep copy, so the
results are identical (``calibrate_shifts`` is ``twozone_shifts.calibrate_all`` on the cached cases; a test holds them
equal). A completed shift's inputs do not change, so every per-shift result is also memoised in-process, keyed by a
fingerprint of the shift's input row and the step's parameters: a refresh recomputes only new or revised shifts.

Every prediction is out of sample: its calibration comes from STABLE shifts that ended before it began. The web scripts
predicted stable shifts only; here every balanced shift gets a prediction (the page needs the latest shift even when it
was not steady), from the same calibration pool, so the stable-shift numbers are identical to the web's
(tests/test_coke_energy_pipeline.py checks this against the vendored ``rolling_predict``).

Slag temperature. The web model put slag at HMT + 40 degC on every shift (``DEFAULTS['slag_T_above_HM']``). Plant
practice (15 Oct 2026): slag runs 30 degC above hot metal when its Al2O3 is low (~16 %) and hotter as Al2O3 rises, up to
about 60 degC. ``PipelineSettings.slag_superheat`` sets that line; each shift's balance then takes its superheat from
the shift's measured slag Al2O3 (``shift_params``). ``None`` keeps the web's constant 40 degC.

``shift_table`` joins them into one row per shift for the page.
"""

from __future__ import annotations

import copy
import hashlib
import warnings
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from utils.coke_energy.bf_balance import DEFAULTS, solve_coke
from utils.coke_energy.build_shifts import build_shift_inputs
from utils.coke_energy.burnt import k_burnt
from utils.coke_energy.rist import TZ, calibrate_2z, solve_2z
from utils.coke_energy.run_shifts import case_from_shift, shift_balance
from utils.coke_energy.whatif import stable_mask

Cases = Mapping[str, dict]

# Hours from the shift date to the middle of the shift (C runs 22:00-06:00, so its middle is 02:00 next day).
SHIFT_MID_HOURS = {"A": 10, "B": 18, "C": 26}
SHIFT_HOURS = 8.0
# A pool shift enters the tracker when its midpoint lies at least this long before the target's: for 8-hour shifts and
# windows, the pool shift ended before the target began (6 min of tolerance).
TRACKER_MIN_AGE_H = SHIFT_HOURS - 0.1
CALIBRATION_METHODS = ("median", "exponential")


@dataclass(frozen=True)
class PipelineSettings:
    """Rolling-window choices the web scripts hard-coded; defaults are the web values (``calibration`` "median")."""

    history_start: str = "2025-08-01"
    predict_from: str = "2025-12-01"
    window_days: int = 14
    min_shifts: int = 10
    bad_shifts: tuple[str, ...] = ("2026-10-07_C",)
    max_ore_moisture_pct: float = 15.0
    # Dry fuel rate of a shift's charge record (coke + nut coke + PCI, kg/tHM) outside this band = a record error
    # (``fuel_rate_kg_thm``); None = no check (the web model had none)
    fuel_rate_valid_kg_thm: tuple[float, float] | None = None
    # ((Al2O3 %, superheat degC) at the low end, (Al2O3 %, superheat degC) at the high end); None = web constant 40 degC
    slag_superheat: tuple[tuple[float, float], tuple[float, float]] | None = None
    # calibration of the burnt-coke offset and of omega / Q_LZ / Q_UZ / r_H: "median" (web) or "exponential" (tracker)
    calibration: str = "median"
    half_life_days: float = 3.0
    tracker_min_shifts: int = 4

    def __post_init__(self) -> None:
        if self.calibration not in CALIBRATION_METHODS:
            raise ValueError(f"calibration {self.calibration!r}: one of {', '.join(CALIBRATION_METHODS)}")
        if not self.half_life_days > 0:
            raise ValueError("half_life_days must be positive")
        band = self.fuel_rate_valid_kg_thm
        if band is not None and not 0 < band[0] < band[1]:
            raise ValueError(f"fuel_rate_valid_kg_thm {band}: low and high, 0 < low < high")

    @property
    def calibration_min_shifts(self) -> int:
        """Stable shifts the calibration of a shift needs (median: ``min_shifts``, tracker: ``tracker_min_shifts``)."""
        return self.min_shifts if self.calibration == "median" else self.tracker_min_shifts

    @property
    def calibration_label(self) -> str:
        """The calibration in words, for notes and views."""
        if self.calibration == "median":
            return f"median over the stable shifts of the {self.window_days} days before"
        return (f"exponentially weighted mean (half-life {self.half_life_days:g} d) over the stable shifts of the "
                f"{self.window_days} days before")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PipelineSettings":
        """Build from the ``coke_energy`` block of ``coke_energy.yml``."""

        tracker = dict(cfg.get("calibration_tracker") or {})
        return cls(
            history_start=str(cfg.get("history_start", cls.history_start)),
            predict_from=str(cfg.get("predict_from", cls.predict_from)),
            window_days=int(cfg.get("rolling_window_days", cls.window_days)),
            min_shifts=int(cfg.get("rolling_min_shifts", cls.min_shifts)),
            bad_shifts=tuple(cfg.get("bad_shifts", cls.bad_shifts) or ()),
            max_ore_moisture_pct=float(cfg.get("max_ore_moisture_pct", cls.max_ore_moisture_pct)),
            fuel_rate_valid_kg_thm=_band_from_config(cfg.get("fuel_rate_valid_kg_thm")),
            slag_superheat=_superheat_from_config(cfg.get("slag_superheat")),
            calibration=str(tracker.get("method", cls.calibration)),
            half_life_days=float(tracker.get("half_life_days", cls.half_life_days)),
            tracker_min_shifts=int(tracker.get("min_shifts", cls.tracker_min_shifts)),
        )


def _band_from_config(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    lo, hi = value
    return (float(lo), float(hi))


def _superheat_from_config(block: Mapping[str, Any] | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not block:
        return None
    lo, hi = block["low"], block["high"]
    return ((float(lo["al2o3_pct"]), float(lo["superheat_c"])), (float(hi["al2o3_pct"]), float(hi["superheat_c"])))


def slag_superheat_c(al2o3_pct: float | None, rule: tuple[tuple[float, float], tuple[float, float]] | None) -> float:
    """Slag temperature above hot metal, degC: linear in slag Al2O3 between the rule's two points, flat outside them.

    Without a rule, or without an Al2O3 analysis, the web model's constant (``DEFAULTS['slag_T_above_HM']``, 40 degC)."""

    if rule is None or al2o3_pct is None or not np.isfinite(al2o3_pct):
        return float(DEFAULTS["slag_T_above_HM"])
    (a0, t0), (a1, t1) = rule
    return float(np.interp(al2o3_pct, [a0, a1], [t0, t1]))


def shift_params(S: pd.DataFrame, keys: pd.Index, settings: "PipelineSettings") -> dict[str, dict]:
    """Model parameters per shift: the web DEFAULTS with the slag superheat of that shift's slag Al2O3."""

    al2o3 = S["slag_pct_al2o3"] if "slag_pct_al2o3" in S.columns else pd.Series(np.nan, index=S.index)
    return {k: {**DEFAULTS, "slag_T_above_HM": slag_superheat_c(al2o3.get(k), settings.slag_superheat)} for k in keys}


def _p(params: Mapping[str, dict] | None, k: str) -> dict:
    return params[k] if params is not None else dict(DEFAULTS)


@dataclass
class PipelineResult:
    """All intermediate frames (web names) plus the joined per-shift table the page reads."""

    S: pd.DataFrame
    R: pd.DataFrame
    P: pd.DataFrame
    B: pd.DataFrame
    Z: pd.DataFrame
    O2: pd.DataFrame
    shifts: pd.DataFrame
    settings: PipelineSettings
    errors: list[str] = field(default_factory=list)


def valid_shift_mask(S: pd.DataFrame) -> pd.Series:
    """Shifts with enough data for a balance (run_shifts.py ``__main__``)."""

    return ((S.n_hours >= 7) & (S.n_ok10 >= 36) & (S.n_taps >= 2) & (S.PRODUCTIONTONNESPERHR / S.n_hours > 60) &
            S.chem_pct_fe.notna() & S.slag_pct_al2o3.notna() & S.dpr_dc_dust_kgthm.notna() & (S.COKE_CALC_MT > 0))


FUEL_COLUMNS = ("COKE_DRY", "NUT_DRY", "PCI_DRY")


def fuel_rate_kg_thm(S: pd.DataFrame) -> pd.Series:
    """Dry fuel rate of each shift's charge record, kg/tHM: (coke + nut coke + PCI, dry t) / hot metal (t) x 1000.

    NaN where a fuel or the production is missing. A blast furnace at BF-02's output runs between about 490 and 630
    kg/tHM (balanced shifts Aug 2025 - Nov 2026 without the record errors); a rate far outside that band is a
    weighing record that is stuck or missing (16-21 Oct 2026: PCI 420-470 kg/tHM against about 200 after the stop of
    15 Oct; 31 Oct - 5 Dec 2025: charged coke 140-220 kg/tHM against 250-280 in the DPR), not a furnace state."""

    if any(c not in S.columns for c in (*FUEL_COLUMNS, "PRODUCTIONTONNESPERHR")):
        return pd.Series(np.nan, index=S.index, dtype=float)
    hm = S["PRODUCTIONTONNESPERHR"].where(S["PRODUCTIONTONNESPERHR"] > 0)
    return (S["COKE_DRY"] + S["NUT_DRY"] + S["PCI_DRY"]) / hm * 1000.0


def excluded_reasons(S: pd.DataFrame, settings: PipelineSettings) -> dict[str, str]:
    """Shifts with a known data error and the reason: the web BAD list, an ore-moisture reading that cannot be real,
    and a fuel record outside ``settings.fuel_rate_valid_kg_thm`` (``fuel_rate_kg_thm``). The first reason holds."""

    out = {k: "listed as a data error (bad_shifts)" for k in settings.bad_shifts}
    if "ORE_TM%" in S.columns:
        tm = S["ORE_TM%"]
        for k, v in tm[tm > settings.max_ore_moisture_pct].items():
            out.setdefault(str(k), f"ore moisture {v:.1f} % above {settings.max_ore_moisture_pct:g} %")
    band = settings.fuel_rate_valid_kg_thm
    if band is not None:
        lo, hi = band
        fuel = fuel_rate_kg_thm(S)
        for k, v in fuel[(fuel < lo) | (fuel > hi)].items():
            out.setdefault(str(k), f"fuel record {v:.0f} kg/tHM dry, outside {lo:g}-{hi:g} kg/tHM")
    return out


def excluded_shifts(S: pd.DataFrame, settings: PipelineSettings) -> set[str]:
    """Shifts with a known data error (``excluded_reasons``)."""

    return set(excluded_reasons(S, settings))


# Per-shift memo. Values are never mutated by the callers (rows go into DataFrames, cases are deep-copied).
_MEMO: dict[tuple, Any] = {}
_MEMO_LIMIT = 60_000
_MISS = object()


def _memo(key: tuple | None, compute: Any) -> Any:
    if key is None:
        return compute()
    value = _MEMO.get(key, _MISS)
    if value is _MISS:
        value = compute()
        if len(_MEMO) >= _MEMO_LIMIT:
            _MEMO.clear()
        _MEMO[key] = value
    return value


def clear_memo() -> None:
    """Forget every memoised per-shift result."""

    _MEMO.clear()


def row_fingerprints(S: pd.DataFrame, keys: pd.Index) -> dict[str, str]:
    """Digest of each shift's input row (all columns, NaN-safe): equal digests mean equal inputs."""

    cols = "|".join(map(str, S.columns)).encode()
    rows = S.loc[keys]
    num = rows.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    text = rows.select_dtypes(exclude="number").astype(str).agg("|".join, axis=1)
    out = {}
    for i, k in enumerate(keys):
        h = hashlib.blake2b(num[i].tobytes(), digest_size=16)
        h.update(str(k).encode())            # the balance row carries its shift key
        h.update(cols)
        h.update(str(text.iloc[i]).encode())
        out[k] = h.hexdigest()
    return out


def _fp(fps: Mapping[str, str] | None, k: str) -> str | None:
    return None if fps is None else fps.get(k)


def run_shift_balances(S: pd.DataFrame, keys: pd.Index, fps: Mapping[str, str] | None = None,
                       params: Mapping[str, dict] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """``shift_balance`` for each key (run_shifts.py ``__main__``); a shift that raises is reported and skipped."""

    rows, errors = [], []
    for key in keys:
        fp = _fp(fps, key)
        p = _p(params, key)
        try:
            rows.append(_memo(None if fp is None else ("balance", fp, p["slag_T_above_HM"]), lambda: shift_balance(S.loc[key], p)))
        except Exception as exc:  # noqa: BLE001 - one bad shift must not stop the rest
            errors.append(f"{key}: {exc}")
    R = pd.DataFrame(rows)
    if not R.empty:
        R = R.set_index("skey")
    return R, errors


def shift_midpoints(sdate: pd.Series, shift: pd.Series) -> pd.Series:
    """Middle of each shift (IST, tz-naive) from its date and letter; NaT for a letter other than A, B, C."""
    return pd.to_datetime(sdate) + pd.to_timedelta(shift.map(SHIFT_MID_HOURS), unit="h")


def _with_dates(R: pd.DataFrame) -> pd.DataFrame:
    return R.assign(stable=stable_mask(R), date=pd.to_datetime(R.sdate), t_mid=shift_midpoints(R.sdate, R["shift"]))


def rolling_pool_median(pool: pd.DataFrame, target_dates: pd.Series, cols: list[str], window_days: int,
                        min_shifts: int) -> pd.DataFrame:
    """For each target date d, the median of ``cols`` over pool rows dated in [d - window_days, d).

    The web scripts' windowing (``st[(st.date < row.date) & (st.date >= row.date - 14 days)]``, at least
    ``min_shifts`` rows); here the pool and the targets may differ."""

    pool = pool.sort_values("date")
    pdates = pool["date"].to_numpy()
    values = pool[cols].to_numpy(dtype=float)
    out = np.full((len(target_dates), len(cols)), np.nan)
    window = np.timedelta64(window_days, "D")
    for i, d in enumerate(pd.to_datetime(target_dates).to_numpy()):
        lo = np.searchsorted(pdates, d - window, side="left")
        hi = np.searchsorted(pdates, d, side="left")
        if hi - lo >= min_shifts:
            block = values[lo:hi]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[i] = np.nanmedian(block, axis=0)
    return pd.DataFrame(out, index=target_dates.index, columns=[c + "_roll" for c in cols])


def tracker_weights(age_h: np.ndarray, window_days: float, half_life_days: float) -> np.ndarray:
    """Weight of pool shifts ``age_h`` hours (midpoint to midpoint) before a target: 0.5 ** (age / half-life) when the
    shift ended before the target began and is at most ``window_days`` old, else 0."""
    age_h = np.asarray(age_h, dtype=float)
    inside = (age_h >= TRACKER_MIN_AGE_H) & (age_h <= 24.0 * window_days)
    with np.errstate(invalid="ignore"):
        return np.where(inside, 0.5 ** (age_h / (24.0 * half_life_days)), 0.0)


def rolling_pool_tracker(pool: pd.DataFrame, target_times: pd.Series, cols: list[str], window_days: float,
                         half_life_days: float, min_shifts: int) -> pd.DataFrame:
    """For each target time t (a shift's or window's midpoint), the weighted mean of ``cols`` over the pool rows with
    the weights of ``tracker_weights`` (pool midpoints in column ``t_mid``). Each column needs at least ``min_shifts``
    rows with a value and a weight; NaN otherwise."""

    pool = pool.sort_values("t_mid")
    ptimes = pool["t_mid"].to_numpy(dtype="datetime64[ns]")
    values = pool[cols].to_numpy(dtype=float)
    have = np.isfinite(values)
    filled = np.where(have, values, 0.0)
    out = np.full((len(target_times), len(cols)), np.nan)
    for i, t in enumerate(pd.to_datetime(target_times).to_numpy(dtype="datetime64[ns]")):
        if np.isnat(t):
            continue
        w = tracker_weights((t - ptimes) / np.timedelta64(1, "h"), window_days, half_life_days)
        wv = w[:, None] * have
        n = ((w > 0)[:, None] & have).sum(axis=0)
        total = wv.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (wv * filled).sum(axis=0) / total
        out[i] = np.where((n >= min_shifts) & (total > 0), mean, np.nan)
    return pd.DataFrame(out, index=target_times.index, columns=[c + "_roll" for c in cols])


def calibration_values(pool: pd.DataFrame, target: pd.DataFrame, cols: list[str],
                       settings: PipelineSettings) -> pd.DataFrame:
    """The calibration of ``cols`` for each target row (``<col>_roll``), by ``settings.calibration``.

    ``pool``: stable shifts with ``date``, ``t_mid`` and ``cols``; ``target``: rows with ``date`` (median: the pool's
    dates before it) and ``t_mid`` (tracker: the pool's shifts that ended before it began)."""
    if settings.calibration == "median":
        return rolling_pool_median(pool, target["date"], cols, settings.window_days, settings.min_shifts)
    return rolling_pool_tracker(pool, target["t_mid"], cols, settings.window_days, settings.half_life_days,
                                settings.tracker_min_shifts)


def calibration_pool(pool: pd.DataFrame, date: Any, t_mid: Any, settings: PipelineSettings,
                     value_col: str | None = None) -> pd.DataFrame:
    """The pool rows the calibration of one target uses, oldest first, with ``age_h`` (target midpoint minus the row's,
    h) and ``weight``: the row's share of the calibration, summing to 1 over the rows with a ``value_col`` (all rows
    without it). Median: every stable shift dated in the window, equal shares; tracker: the shifts of
    ``tracker_weights``."""
    t = pd.Timestamp(t_mid)
    if settings.calibration == "median":
        day = pd.Timestamp(date).normalize()
        used = pool[(pool["date"] >= day - pd.Timedelta(days=settings.window_days)) & (pool["date"] < day)].copy()
        raw = pd.Series(1.0, index=used.index)
    else:
        age = (t - pd.to_datetime(pool["t_mid"])) / pd.Timedelta(hours=1)
        w = pd.Series(tracker_weights(age.to_numpy(dtype=float), settings.window_days, settings.half_life_days),
                      index=pool.index)
        used = pool[w > 0].copy()
        raw = w[w > 0]
    used["age_h"] = (t - pd.to_datetime(used["t_mid"])) / pd.Timedelta(hours=1)
    counted = raw.where(used[value_col].notna(), 0.0) if value_col is not None else raw
    total = float(counted.sum())
    used["weight"] = counted / total if total > 0 else np.nan
    return used.sort_values("t_mid")


def build_cases(S: pd.DataFrame, keys: pd.Index, fps: Mapping[str, str] | None = None) -> dict[str, dict]:
    """``case_from_shift`` once per shift; steps take ``copy.deepcopy`` of it, as the web rebuilt it fresh each time."""

    return {k: _memo(None if _fp(fps, k) is None else ("case", _fp(fps, k)), lambda: case_from_shift(S.loc[k]))
            for k in keys}


def _case(cases: Cases, k: str, eta_H2_pct: float | None = None) -> dict:
    case = copy.deepcopy(cases[k])
    if eta_H2_pct is not None:
        case["eta_H2"] = eta_H2_pct / 100 if 10 < eta_H2_pct < 70 else DEFAULTS["eta_H2"]
    return case


def calibrate_shifts(cases: Cases, R: pd.DataFrame, p: Mapping[str, Any], T_R: float | None = None,
                     fps: Mapping[str, str] | None = None, params: Mapping[str, dict] | None = None) -> pd.DataFrame:
    """Two-zone calibration of every shift: ``twozone_shifts.calibrate_all`` on the cached cases.

    ``p`` applies to every shift unless ``params`` gives each shift its own."""

    def one(key: str, rb: pd.Series) -> dict | None:
        case = _case(cases, key, rb.eta_H2)
        try:
            cal = calibrate_2z(case, rb.coke_kgthm_dry, rb.V_air_reconciled, rb.eta_CO / 100,
                               params[key] if params is not None else p, T_R=T_R)
        except Exception:  # noqa: BLE001 - the web skipped (and printed) such shifts
            return None
        z = cal["z"]; prod = rb.prod_th
        return dict(skey=key, omega=cal["omega"], y_d=cal["y_d"], r_d=cal["r_d"], Q_LZ=cal["Q_LZ"], Q_UZ=cal["Q_UZ"], Q_tot=cal["Q_tot"],
                    Q_LZ_GJh=cal["Q_LZ"] * prod / 1000, Q_UZ_GJh=cal["Q_UZ"] * prod / 1000, r_H=cal["r_H"], eta_f=cal["eta_f"],
                    eta=cal["eta"], eta_H2=cal["eta_H2"], O_TRZ=z["O_TRZ"], O_max=z["O_max_TRZ"], nC=z["nC"], nH2=z["nH2"],
                    y_top=z["y_top"], O_err=cal["O_err"])

    rows = []
    tz = tuple(sorted(TZ.items()))
    for key, rb in R[["eta_H2", "coke_kgthm_dry", "V_air_reconciled", "eta_CO", "prod_th"]].iterrows():
        fp = _fp(fps, key)
        superheat = (params[key] if params is not None else p)["slag_T_above_HM"]
        mkey = None if fp is None else ("calibrate_2z", fp, superheat, *rb.to_numpy().tolist(), T_R, tz)
        row = _memo(mkey, lambda: one(key, rb))
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows).set_index("skey")


def oxygen_balance_model(S: pd.DataFrame, R: pd.DataFrame, settings: PipelineSettings, cases: Cases | None = None,
                         fps: Mapping[str, str] | None = None, params: Mapping[str, dict] | None = None) -> pd.DataFrame:
    """Coke from the O-balance + enthalpy balance at the shift's measured eta CO (rolling_predict.py)."""

    cases = cases if cases is not None else build_cases(S, R.index)
    Rs = _with_dates(R)
    Rs["U_GJh"] = (Rs.R_available_for_losses - Rs.R_out_cooling_total) * Rs.prod_th / 1000
    U = rolling_pool_median(Rs[Rs.stable], Rs["date"], ["U_GJh"], settings.window_days, settings.min_shifts)["U_GJh_roll"]
    out = []
    cols = ["date", "eta_H2", "eta_CO", "coke_kgthm_dry", "R_out_cooling_total", "prod_th"]
    for k, rb in Rs.loc[U.notna() & (Rs.date >= settings.predict_from), cols].iterrows():
        Q = rb.R_out_cooling_total + U[k] * 1000 / rb.prod_th
        fp = _fp(fps, k)
        p = _p(params, k)
        K = _memo(None if fp is None else ("solve_coke", fp, p["slag_T_above_HM"], rb.eta_H2, rb.eta_CO, Q),
                  lambda: solve_coke(_case(cases, k, rb.eta_H2), p, eta_CO=rb.eta_CO / 100, Q_loss=Q)["K"])
        out.append(dict(skey=k, m=str(rb.date.to_period("M")), K_act=rb.coke_kgthm_dry, K_roll=K, Q_loss=Q, U_roll_GJh=U[k]))
    P = pd.DataFrame(out, columns=["skey", "m", "K_act", "K_roll", "Q_loss", "U_roll_GJh"]).set_index("skey")
    P["err"] = P.K_roll - P.K_act
    return P


def burnt_coke(S: pd.DataFrame, R: pd.DataFrame, settings: PipelineSettings, cases: Cases | None = None,
               fps: Mapping[str, str] | None = None, params: Mapping[str, dict] | None = None) -> pd.DataFrame:
    """Coke burnt by the metered blast per shift and its offset-corrected value (burnt.py ``__main__``)."""

    cases = cases if cases is not None else build_cases(S, R.index)
    rows = []
    for key, rb in R[["eta_H2", "eta_CO", "coke_kgthm_dry", "blast_air_nm3thm", "V_air_reconciled", "hm_t", "prod_th"]].iterrows():
        fp = _fp(fps, key)
        p = _p(params, key)
        Kb, dKdO, chk = _memo(None if fp is None else ("k_burnt", fp, p["slag_T_above_HM"], rb.eta_H2, rb.eta_CO),
                              lambda: k_burnt(_case(cases, key, rb.eta_H2), rb.eta_CO / 100, p))
        rows.append(dict(skey=key, K_act=rb.coke_kgthm_dry, K_burnt=Kb, V_meter=rb.blast_air_nm3thm, V_rec=rb.V_air_reconciled,
                         hm_t=rb.hm_t, prod_th=rb.prod_th, O_chk=chk))
    B = pd.DataFrame(rows).set_index("skey")
    B["gap"] = B.K_act - B.K_burnt                   # + = coke charged ahead of what the blast burnt
    B["dK_per_pct_air"] = (B.K_burnt - B.K_act) / (100 * (B.V_meter / B.V_rec - 1))
    Rs = _with_dates(R).join(B[["gap"]])
    # the offset's name keeps the web's "14"; its value follows settings.calibration (median or tracker)
    B["offset14"] = calibration_values(Rs[Rs.stable], Rs, ["gap"], settings)["gap_roll"]
    B["K_burnt_corr"] = B.K_burnt + B.offset14
    return B


def energy_balance_model(S: pd.DataFrame, R: pd.DataFrame, Z: pd.DataFrame, settings: PipelineSettings,
                         cases: Cases | None = None, fps: Mapping[str, str] | None = None,
                         params: Mapping[str, dict] | None = None) -> pd.DataFrame:
    """Two-zone coke with omega, Q_LZ, Q_UZ, r_H at their calibration (twozone_shifts.py ``__main__``; the web's 14-day
    medians, or the tracker, per ``settings.calibration``)."""

    cases = cases if cases is not None else build_cases(S, R.index)
    Rs = _with_dates(R).join(Z, rsuffix="_2z")
    cols = ["omega", "Q_LZ_GJh", "Q_UZ_GJh", "r_H"]
    Rs = Rs.join(calibration_values(Rs[Rs.stable], Rs, cols, settings))
    out = []
    cols = ["date", "prod_th", "coke_kgthm_dry", "eta_CO", "T_top", "blast_air_nm3thm",
            "omega_roll", "Q_LZ_GJh_roll", "Q_UZ_GJh_roll", "r_H_roll"]
    tz = tuple(sorted(TZ.items()))
    keep = ("K", "eta", "T_top", "V", "y_d")
    for k, rb in Rs.loc[Rs.omega_roll.notna() & (Rs.date >= settings.predict_from), cols].iterrows():
        prod = rb.prod_th
        fp = _fp(fps, k)
        p = _p(params, k)
        mkey = None if fp is None else ("solve_2z", fp, p["slag_T_above_HM"], rb.omega_roll, rb.Q_LZ_GJh_roll,
                                        rb.Q_UZ_GJh_roll, rb.r_H_roll, tz)

        def solve(k=k, rb=rb, prod=prod, p=p) -> dict:
            # the web did not set eta_H2 on this case
            s = solve_2z(_case(cases, k), p, rb.omega_roll, rb.Q_LZ_GJh_roll * 1000 / prod, rb.Q_UZ_GJh_roll * 1000 / prod,
                         rb.r_H_roll)
            return {name: s[name] for name in keep}

        s = _memo(mkey, solve)
        out.append(dict(skey=k, m=str(rb.date.to_period("M")), K_act=rb.coke_kgthm_dry, K_2z=s["K"], eta_2z=100 * s["eta"], eta_meas=rb.eta_CO,
                        T_top_2z=s["T_top"], T_top_meas=rb.T_top, V_2z=s["V"], V_meter=rb.blast_air_nm3thm, y_d_2z=s["y_d"],
                        omega_roll=rb.omega_roll, Q_LZ_GJh_roll=rb.Q_LZ_GJh_roll, Q_UZ_GJh_roll=rb.Q_UZ_GJh_roll, r_H_roll=rb.r_H_roll))
    columns = ["skey", "m", "K_act", "K_2z", "eta_2z", "eta_meas", "T_top_2z", "T_top_meas", "V_2z", "V_meter", "y_d_2z",
               "omega_roll", "Q_LZ_GJh_roll", "Q_UZ_GJh_roll", "r_H_roll"]
    return pd.DataFrame(out, columns=columns).set_index("skey")


def shift_table(R: pd.DataFrame, P: pd.DataFrame, B: pd.DataFrame, Z: pd.DataFrame, O2: pd.DataFrame,
                provisional_dates: Collection[date] = ()) -> pd.DataFrame:
    """One row per balanced shift with the four coke rates and the quantities the page shows."""

    T = pd.DataFrame(index=R.index)
    T["sdate"] = pd.to_datetime(R.sdate)
    T["shift"] = R["shift"]
    T["t_mid"] = T.sdate + pd.to_timedelta(T["shift"].map(SHIFT_MID_HOURS), unit="h")
    T["stable"] = stable_mask(R)
    T["provisional"] = T.sdate.dt.date.isin(set(provisional_dates))
    T["K_charged"] = R.coke_kgthm_dry
    T["K_obal"] = P["K_roll"]
    T["K_energy"] = O2["K_2z"]
    T["K_burnt"] = B["K_burnt_corr"]
    T["K_burnt_raw"] = B["K_burnt"]
    T["burnt_offset"] = B["offset14"]
    T["eta_meas"] = R.eta_CO
    T["eta_energy"] = O2["eta_2z"]
    T["T_top_meas"] = R.T_top
    T["T_top_energy"] = O2["T_top_2z"]
    T["omega_shift"] = Z["omega"]
    T["omega_fixed"] = O2["omega_roll"]
    T["Q_LZ_GJh_shift"] = Z["Q_LZ_GJh"]
    T["Q_UZ_GJh_shift"] = Z["Q_UZ_GJh"]
    T["Q_LZ_GJh_fixed"] = O2["Q_LZ_GJh_roll"]
    T["Q_UZ_GJh_fixed"] = O2["Q_UZ_GJh_roll"]
    T["r_H_fixed"] = O2["r_H_roll"]
    T["y_d_shift"] = Z["y_d"]
    T["y_d_energy"] = O2["y_d_2z"]
    T["U_roll_GJh"] = P["U_roll_GJh"]
    T["Q_loss_obal"] = P["Q_loss"]
    T["V_energy"] = O2["V_2z"]
    for c in ["prod_th", "hm_t", "pci_kgthm_dry", "nut_kgthm_dry", "fuel_kgthm_dry", "HMT", "hm_si", "hm_c", "HBT", "T_top",
              "humidity_g", "blast_air_nm3thm", "V_air_reconciled", "blast_meter_vs_reconciled_pct", "O2_nm3thm", "eta_H2",
              "slag_kgthm_model", "slag_kgthm_Al2O3tracer", "coke_ash", "coke_FC", "R_out_cooling_total", "Q_loss_calibrated"]:
        if c in R.columns:
            T[c] = R[c]
    return T.sort_values("t_mid")


def run_pipeline(fd: pd.DataFrame, m10: pd.DataFrame, hm: pd.DataFrame, dpr: pd.DataFrame,
                 settings: PipelineSettings | None = None, provisional_dates: Collection[date] = ()) -> PipelineResult:
    """Build every shift's balance and the two models' out-of-sample predictions from the four source frames.

    ``provisional_dates``: DPR dates that were filled in (dust carried forward) because the DPR was not entered yet;
    their shifts are flagged in the table."""

    settings = settings or PipelineSettings()
    with warnings.catch_warnings():
        # The web scripts ran with all warnings silenced (pandas/numpy noise on all-NaN shifts); keep that local.
        warnings.simplefilter("ignore")
        S = build_shift_inputs(fd, m10, hm, dpr, start=settings.history_start)
        good = valid_shift_mask(S)
        keys = S.index[good & ~S.index.isin(excluded_shifts(S, settings))]
        fps = row_fingerprints(S, keys)
        params = shift_params(S, keys, settings)
        R, errors = run_shift_balances(S, keys, fps, params)
        if R.empty:
            empty = pd.DataFrame()
            return PipelineResult(S, R, empty, empty, empty, empty, empty, settings, errors)
        cases = build_cases(S, R.index, fps)
        P = oxygen_balance_model(S, R, settings, cases, fps, params)
        B = burnt_coke(S, R, settings, cases, fps, params)
        Z = calibrate_shifts(cases, R, dict(DEFAULTS), fps=fps, params=params)
        O2 = energy_balance_model(S, R, Z, settings, cases, fps, params)
        T = shift_table(R, P, B, Z, O2, provisional_dates)
        T["slag_superheat_c"] = pd.Series({k: params[k]["slag_T_above_HM"] for k in T.index})
        T["slag_al2o3_pct"] = S["slag_pct_al2o3"].reindex(T.index)
    return PipelineResult(S, R, P, B, Z, O2, T, settings, errors)
