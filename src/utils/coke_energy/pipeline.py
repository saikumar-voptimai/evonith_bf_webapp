"""The web model's shift pipeline, run in memory.

The Claude-web scripts each read the previous step's pickle and wrote the next one. Their ``__main__`` blocks are
reproduced here, in the same order and with the same arithmetic:

    build_shifts.py      build_shift_inputs      S    shift inputs (hourly dataset + 10-min data + HM/slag + DPR)
    run_shifts.py        run_shift_balances      R    measured-gas balance and reconciled basis, every valid shift
    rolling_predict.py   oxygen_balance_model    P    eta CO fixed at the analyser's value, heat-loss allowance U from
                                                      the previous 14 days
    burnt.py             burnt_coke              B    coke the metered blast burnt, less the 14-day (charged - burnt)
    twozone_shifts.py    calibrate_shifts,       Z    two-zone calibration of every shift (omega, Q_LZ, Q_UZ, r_H)
                         energy_balance_model    O2   omega, Q_LZ, Q_UZ, r_H fixed at their 14-day medians

``case_from_shift`` is the slow part of a shift (a wide row read field by field), and the web scripts rebuilt the same
case in every step. Here each shift's case is built once (``build_cases``) and every step gets a deep copy, so the
results are identical (``calibrate_shifts`` is ``twozone_shifts.calibrate_all`` on the cached cases; a test holds them
equal). A completed shift's inputs do not change, so every per-shift result is also memoised in-process, keyed by a
fingerprint of the shift's input row and the step's parameters: a refresh recomputes only new or revised shifts.

Every prediction is out of sample: its calibration is the median over STABLE shifts on EARLIER dates. The web scripts
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


@dataclass(frozen=True)
class PipelineSettings:
    """Rolling-window choices the web scripts hard-coded; defaults are the web values."""

    history_start: str = "2025-08-01"
    predict_from: str = "2025-12-01"
    window_days: int = 14
    min_shifts: int = 10
    bad_shifts: tuple[str, ...] = ("2026-10-07_C",)
    max_ore_moisture_pct: float = 15.0
    # ((Al2O3 %, superheat degC) at the low end, (Al2O3 %, superheat degC) at the high end); None = web constant 40 degC
    slag_superheat: tuple[tuple[float, float], tuple[float, float]] | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PipelineSettings":
        """Build from the ``coke_energy`` block of ``coke_energy.yml``."""

        return cls(
            history_start=str(cfg.get("history_start", cls.history_start)),
            predict_from=str(cfg.get("predict_from", cls.predict_from)),
            window_days=int(cfg.get("rolling_window_days", cls.window_days)),
            min_shifts=int(cfg.get("rolling_min_shifts", cls.min_shifts)),
            bad_shifts=tuple(cfg.get("bad_shifts", cls.bad_shifts) or ()),
            max_ore_moisture_pct=float(cfg.get("max_ore_moisture_pct", cls.max_ore_moisture_pct)),
            slag_superheat=_superheat_from_config(cfg.get("slag_superheat")),
        )


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


def excluded_shifts(S: pd.DataFrame, settings: PipelineSettings) -> set[str]:
    """Shifts with a known data error: the web BAD list plus an ore-moisture reading that cannot be real."""

    bad = set(settings.bad_shifts)
    if "ORE_TM%" in S.columns:
        bad |= set(S.index[S["ORE_TM%"] > settings.max_ore_moisture_pct])
    return bad


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


def _with_dates(R: pd.DataFrame) -> pd.DataFrame:
    return R.assign(stable=stable_mask(R), date=pd.to_datetime(R.sdate))


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
    B["offset14"] = rolling_pool_median(Rs[Rs.stable], Rs["date"], ["gap"], settings.window_days, settings.min_shifts)["gap_roll"]
    B["K_burnt_corr"] = B.K_burnt + B.offset14
    return B


def energy_balance_model(S: pd.DataFrame, R: pd.DataFrame, Z: pd.DataFrame, settings: PipelineSettings,
                         cases: Cases | None = None, fps: Mapping[str, str] | None = None,
                         params: Mapping[str, dict] | None = None) -> pd.DataFrame:
    """Two-zone coke with omega, Q_LZ, Q_UZ, r_H at their 14-day medians (twozone_shifts.py ``__main__``)."""

    cases = cases if cases is not None else build_cases(S, R.index)
    Rs = _with_dates(R).join(Z, rsuffix="_2z")
    cols = ["omega", "Q_LZ_GJh", "Q_UZ_GJh", "r_H"]
    Rs = Rs.join(rolling_pool_median(Rs[Rs.stable], Rs["date"], cols, settings.window_days, settings.min_shifts))
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
