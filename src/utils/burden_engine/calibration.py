"""Two-zone calibration and operating regime for the burden engine, from the Coke Energy pipeline's shift tables (#252).

The engine needs (a) the two-zone calibration - shaft efficiency, lower/upper-zone heat losses (GJ/h), r_H, the fuel and
hot-metal analyses, dust - in the layout of ``assets/burden_engine/cal_2z_jul_oct_2026.json``, and (b) today's
operating regime (controls, pressures, thermal state) to start from.

Calibration method (the Claude-web ``ibrm_model.load_calibration`` + ``ibrm_2z.load_calibration_2z``, ported):
  1. average the input rows of the chosen stable shifts (``coke_energy.whatif`` baseline method) and build one case;
  2. size the burden from the Fe balance at the actual coke rate (``size_burden``) so calibration and blends use the
     same method; calibrate the heat loss at the measured eta CO (fixed-eta model) to get the blast volume;
  3. two-zone calibration (``coke_energy.rist.calibrate_2z``): omega, direct reduction, lower/upper-zone losses, r_H;
  4. re-solve at the calibration burden with basicity held (fixed omega) for the model's own blast and coke at the
     calibration point (``hb_per_t_model_cal``, ``coke_model_cal``).

Pure numerical code: takes the pipeline frames (S = shift inputs, R = shift balances); no I/O.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from utils.coke_energy.bf_balance import DEFAULTS, calibrate_loss, flux_needed, slag_and_flux
from utils.coke_energy.bf_thermo import MW
from utils.coke_energy.weather import blast_moisture, mean_blast_moisture
from utils.coke_energy.rist import TZ, calibrate_2z, solve_2z
from utils.coke_energy.run_shifts import case_from_shift
from utils.coke_energy.whatif import stable_mask

MATS = ("sin", "ore", "pel")
CASE_KEYS = ("coke", "nut", "pci", "nut_kg", "pci_kg", "hm", "HMT", "slag_FeO_pct", "T_top", "HBT", "humidity_g",
             "O2_per_air", "dust", "prod_th", "eta_H2")


class CalibrationError(ValueError):
    """Too few or unusable shifts for a calibration."""


def size_burden(case, K, shares, chem, p, slag_ref):
    """Copy of case with sinter/ore/pellet dry kg set from the Fe balance at coke rate K (ibrm_model, verbatim)."""
    c = copy.deepcopy(case)
    tot = sum(shares.values())
    fe_avg = sum(shares[m] / tot * chem[m]["Fe"] for m in MATS) / 100.0
    du = c["dust"]
    fe_dust = du["dc_kg"] * du["dc_Fe_pct"] / 100 + du["gcp_kg"] * du["gcp_Fe_pct"] / 100
    fe_slag = slag_ref * c["slag_FeO_pct"] / 100 * MW["Fe"] / MW["FeO"]
    fe_fuel = 0.0
    for kg, fu in [(K, c["coke"]), (c["nut_kg"], c["nut"]), (c["pci_kg"], c["pci"])]:
        fe_fuel += kg * fu["ash"] / 100 * fu["ash_comp"].get("Fe2O3", 0) / 100 * 2 * MW["Fe"] / MW["Fe2O3"]
    flux = [m for m in c["burden"] if m["name"] == "flux"][0]
    fe_flux = flux["dry_kg"] * flux["comp"].get("Fe", 0) / 100
    W = (c["hm"]["Fe"] * 10 + fe_slag + fe_dust - fe_fuel - fe_flux) / fe_avg
    for m in c["burden"]:
        if m["name"] in ("sinter", "ore", "pellet"):
            key = {"sinter": "sin", "ore": "ore", "pellet": "pel"}[m["name"]]
            m["dry_kg"] = W * shares[key] / tot
            m["comp"] = {**m["comp"], **{k: v for k, v in chem[key].items() if k != "moist"}}
            if "moist" in chem[key]:
                m["moist_pct"] = chem[key]["moist"]
    c["W_ibrm"] = W
    return c


def chem_from_case(case):
    out = {}
    for m in case["burden"]:
        key = {"sinter": "sin", "ore": "ore", "pellet": "pel"}.get(m["name"])
        if key:
            out[key] = {**m["comp"], "moist": m.get("moist_pct", 0.0)}
    return out


def select_shifts(S: pd.DataFrame, R: pd.DataFrame, *, start=None, end=None, pci: float | None = None,
                  nut: float | None = None, tol_pci: float = 10.0, tol_nut: float = 6.0,
                  exclude: Iterable[str] = ()) -> pd.Index:
    """Stable shifts (``whatif.stable_mask``) with start <= date < end, optionally within a PCI / nut-coke regime
    (kg/tHM as charged, as ``whatif.baseline_case``)."""
    keys = R.index[stable_mask(R)].intersection(S.index).difference(pd.Index(list(exclude)))
    sel = S.loc[keys]
    d = pd.to_datetime(sel["sdate"])
    if start is not None:
        sel = sel[d >= pd.Timestamp(start)]
    if end is not None:
        sel = sel[pd.to_datetime(sel["sdate"]) < pd.Timestamp(end)]
    if pci is not None or nut is not None:
        prod = sel["PRODUCTIONTONNESPERHR"]
        if pci is not None:
            sel = sel[((sel["PCI_CALC_MT"] / prod * 1000) - pci).abs() <= tol_pci]
        if nut is not None:
            sel = sel[((sel["NUTCOKE_CALC_MT"] / sel["PRODUCTIONTONNESPERHR"] * 1000) - nut).abs() <= tol_nut]
    return sel.index


def calibrate_from_shifts(S: pd.DataFrame, R: pd.DataFrame, keys: Iterable[str], p: Mapping[str, Any] | None = None,
                          T_R: float | None = None, min_shifts: int = 1) -> dict[str, Any]:
    """Two-zone calibration over the given shifts, in the engine's ``cal`` layout."""
    keys = pd.Index(list(keys))
    if len(keys) < min_shifts:
        raise CalibrationError(f"{len(keys)} stable shifts in the window; at least {min_shifts} are needed")
    p = dict(p or DEFAULTS)
    sel = S.loc[keys]
    row = sel.select_dtypes("number").mean()
    row["sdate"] = pd.to_datetime(sel["sdate"]).max()
    row["shift"] = "avg"
    case = case_from_shift(row)
    case["humidity_g"] = mean_blast_moisture(pd.to_datetime(sel["sdate"]))[0]     # window mean (DATA-11)
    Rb = R.loc[keys]
    case["eta_H2"] = float(Rb["eta_H2"].mean()) / 100
    eta = float(Rb["eta_CO"].mean()) / 100
    chem = chem_from_case(case)
    kg = {k: m["dry_kg"] for k, m in zip(("sin", "ore", "pel", "flux"), case["burden"])}
    W = kg["sin"] + kg["ore"] + kg["pel"]
    if W <= 0:
        raise CalibrationError("no iron-bearing burden in the selected shifts")
    shares = {k: 100 * kg[k] / W for k in MATS}
    for k in MATS:                      # a material not charged in the window: its chemistry is not needed
        if shares[k] <= 0:
            chem.setdefault(k, {"Fe": 60.0})
            chem[k].setdefault("Fe", 60.0)
    K = float(case["K_actual"])
    slag_ref = slag_and_flux(K, case, p)["slag_kg"]
    c = size_burden(case, K, shares, chem, p, slag_ref)
    Ql, V, rr = calibrate_loss(c, K, p, eta_CO=eta)
    B2_target = rr["slag"]["B2"]
    c["eta_H2"] = case["eta_H2"]
    z = calibrate_2z(c, K, V, eta, p, T_R=T_R)
    prod = float(case["prod_th"])
    T_R = TZ["T_R"] if T_R is None else T_R
    Q_LZ_GJh = z["Q_LZ"] * prod / 1000
    Q_UZ_GJh = z["Q_UZ"] * prod / 1000
    # the model's own blast and coke at the calibration burden (fixed omega, basicity held)
    case_fn = lambda Kx: size_burden(case, Kx, shares, chem, p, slag_ref)
    flux_fn = lambda Kx, cx: flux_needed(Kx, cx, p, B2_target)
    s = solve_2z(case, p, z["omega"], z["Q_LZ"], z["Q_UZ"], z["r_H"], T_R=T_R, case_fn=case_fn, flux_fn=flux_fn,
                 x0=(K, V, z["y_d"]))
    sized = s["case"]
    keep_case = {k: copy.deepcopy(case[k]) for k in CASE_KEYS}
    dates = pd.to_datetime(sel["sdate"])
    # Packed-bed anchor (FLOW-01 base point) and meter basis for the same shifts: the blast-limit rules are then
    # applied as changes from this window's burden and controls ("re-anchored"), not from the Jul-Oct 2026 averages.
    hbv = pd.to_numeric(sel.get("hot_blast_vol_nm3h"), errors="coerce")
    o2 = pd.to_numeric(sel.get("oxygen_flow"), errors="coerce")
    wet = {m["name"]: m["dry_kg"] / max(1e-9, 1 - m.get("moist_pct", 0.0) / 100) for m in sized["burden"]}
    fer = sum(wet.get(n, 0.0) for n in ("sinter", "ore", "pellet"))
    anchor = dict(V=float(hbv.mean()), Pb=float(pd.to_numeric(sel.get("hot_blast_press"), errors="coerce").mean()),
                  Pt=float(pd.to_numeric(sel.get("top_press_avg"), errors="coerce").mean()),
                  pellet=float(100 * wet.get("pellet", 0.0) / fer) if fer > 0 else 0.0,
                  o2_pct=float((100 * o2 / hbv).mean()), gomti=0.0, ore6_on=0.0)
    anchor = {k: v for k, v in anchor.items() if np.isfinite(v)}
    return dict(
        case=keep_case, p=p, omega=float(z["omega"]), Q_LZ_GJh=float(Q_LZ_GJh), Q_UZ_GJh=float(Q_UZ_GJh),
        r_H=float(z["r_H"]), T_R=float(T_R), y_d=float(z["y_d"]), K=K, V=float(V), B2=float(B2_target),
        shares_dry=shares, chem=chem, hb_per_t_model_cal=float(s["V"] + s["ro"]["V_O2"]), coke_model_cal=float(s["K"]),
        eta_CO_measured=100 * eta, hb_meter_nm3h=anchor.get("V"), anchor=anchor,
        period=[str(dates.min().date()), str(dates.max().date())], n_shifts=int(len(keys)),
        shifts=list(map(str, keys)),
        burden_cal=[dict(name=m["name"], dry_kg=m["dry_kg"], moist_pct=m.get("moist_pct", 0.0), comp=m["comp"])
                    for m in sized["burden"]],
    )


def _median(s: pd.Series | None) -> float | None:
    if s is None:
        return None
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.median()) if len(v) else None


def operating_regime(S: pd.DataFrame, R: pd.DataFrame, keys: Iterable[str], enrich_tag_per_o2_pct: float = 0.84,
                     as_of: pd.Timestamp | None = None) -> dict[str, Any]:
    """Today's operating regime: medians over the given shifts of the controls, pressures and thermal state.

    O2/air uses the hot-blast meter basis (DATA-01: the meter includes the O2). Blast moisture is the weather record's
    value for ``as_of`` (default: the last shift's date), else the month's typical ambient value (DATA-04, DATA-11)."""
    keys = pd.Index(list(keys))
    sel = S.loc[keys]
    Rb = R.loc[keys]
    hbv = pd.to_numeric(sel.get("hot_blast_vol_nm3h"), errors="coerce")
    o2 = pd.to_numeric(sel.get("oxygen_flow"), errors="coerce")
    o2_per_air = (o2 / (hbv - o2)).replace([np.inf, -np.inf], np.nan)
    day = as_of if as_of is not None else pd.to_datetime(sel["sdate"]).max()
    hum, hum_src = blast_moisture(day)
    pci = pd.to_numeric(Rb.get("pci_kgthm_dry"), errors="coerce")
    nut = pd.to_numeric(Rb.get("nut_kgthm_dry"), errors="coerce")
    out = dict(
        n_shifts=int(len(keys)),
        period=[str(pd.to_datetime(sel["sdate"]).min().date()), str(pd.to_datetime(sel["sdate"]).max().date())] if len(keys) else [],
        HBT=_median(sel.get("hot_blast_temp")), hot_blast_nm3h=_median(hbv), O2_nm3h=_median(o2),
        O2_per_air=_median(o2_per_air), Pb=_median(sel.get("hot_blast_press")), Pt=_median(sel.get("top_press_avg")),
        dP=_median(sel.get("body_dp_total")), permeability=_median(sel.get("body_perm")),
        HMT=_median(Rb.get("HMT")), hm_Si=_median(Rb.get("hm_si")), prod_th=_median(Rb.get("prod_th")),
        coke_kgthm_dry=_median(Rb.get("coke_kgthm_dry")), pci_kgthm_dry=_median(pci), nut_kgthm_dry=_median(nut),
        eta_CO=_median(Rb.get("eta_CO")), T_top=_median(Rb.get("T_top")), raft_dcs=_median(Rb.get("raft")),
        humidity_g=float(hum), humidity_source=hum_src,
    )
    if out["O2_per_air"] is not None:
        r = out["O2_per_air"]
        out["enrich_tag"] = enrich_tag_per_o2_pct * 100 * r / (1 + r)
    return out
