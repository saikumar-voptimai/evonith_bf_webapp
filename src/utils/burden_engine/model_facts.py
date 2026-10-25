"""Model and validation facts of the fact register, generated from the engine and the STRATA asset (groups MOD, VAL).

MOD  Coke response of the two shaft-efficiency models to a standard change at the reference point: Fixed-omega (Rist
     baseline, omega held at the calibration, ASM-01) and STRATA-omega (blend- and operation-dependent, ASM-18). Each
     fact carries both values, their difference and the 1-sigma of the STRATA value.
       substitutions  10 wet pt of ferrous, flux held at the reference plan (coke through chemistry + omega)
       production     +1 t/h at the same burden (heat loss GJ/h held: dilution in both; STRATA 1.4+ has no production
                      channel, so the two models differ only through the other-operation response)
       moisture       +1 g/Nm3 blast moisture at the same burden
       nut coke       +10 kg/tHM at the same burden (STRATA 1.3+, GAS-23)
     Reference point: the 16 Oct 2026 UAT yard, the shipped calibration, 2,270 t/d, a record-average ferrous blend.
VAL  Out-of-sample burnt coke by blend family (``validation.by_blend`` of the STRATA asset, oos/by_blend.py):
     realised against Fixed-omega and STRATA-omega, skill, bias and the verdict (Supported / Disputed / Neutral /
     Modelled only) as the fact status.

``python -m utils.burden_engine.model_facts`` prints the facts; ``--write`` replaces the MOD and VAL rows of
``config/facts_register.json`` (other rows untouched). Pure numerical code apart from that file write; no Streamlit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from utils.burden_engine.engine import REFERENCE_INPUTS_PATH, Engine, Limits, Settings
from utils.burden_engine.facts import REGISTER_PATH
from utils.burden_engine.omega_model import MODEL_PATH, difference_sigma

P_REF = 2270 / 24
BASE_BLEND = {"sinter_sp_02": 64.0, "lloyds_pellet": 10.0, "lloyds_clo": 13.0, "geomin_clo": 10.0, "nmdc_rom": 3.0}
SUBSTITUTIONS = (
    ("MOD-01", "Pellet for sinter", {"lloyds_pellet": 10, "sinter_sp_02": -10}),
    ("MOD-02", "Lloyds CLO for sinter", {"lloyds_clo": 10, "sinter_sp_02": -10}),
    ("MOD-03", "Geomin CLO for sinter", {"geomin_clo": 10, "sinter_sp_02": -10}),
    ("MOD-04", "NMDC ROM for sinter", {"nmdc_rom": 10, "sinter_sp_02": -10}),
    ("MOD-05", "Donimalai for sinter", {"nmdc_donimalai": 10, "sinter_sp_02": -10}),
    ("MOD-06", "Gomti for sinter", {"gomti_clo": 10, "sinter_sp_02": -10}),
    ("MOD-07", "Lloyds oversize for sinter", {"lloyds_oversize_clo": 10, "sinter_sp_02": -10}),
    ("MOD-08", "Pellet for Lloyds CLO", {"lloyds_pellet": 10, "lloyds_clo": -10}),
)


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()


def _engine() -> Engine:
    with open(REFERENCE_INPUTS_PATH, encoding="utf-8") as fh:
        inputs = json.load(fh)
    mats = [m["id"] for m in inputs["materials"]]
    bounds = {m["id"]: (0.0, 100.0) for m in inputs["materials"]}
    return Engine(inputs, materials=mats, bounds=bounds)


def _blend_z(eng: Engine, z_ref: np.ndarray, shares: dict[str, float], s: Settings, P: float) -> np.ndarray:
    z = np.array(z_ref, float)
    Y = float(np.sum(z[:eng.nf]))
    for i, v in enumerate(eng.fer):
        z[i] = Y * shares.get(v.id, 0.0) / 100.0
    return eng.size_fe(z, s, P)


def model_facts(eng: Engine | None = None) -> list[dict[str, Any]]:
    """MOD rows: coke response under Fixed-omega and STRATA-omega (kg/tHM dry)."""
    eng = eng or _engine()
    s_fix = Settings(omega_mode="fixed")
    ref = eng.min_cost(P=P_REF, s=s_fix, L=Limits(), fuel=True)
    z0 = _blend_z(eng, np.asarray(ref["z"], float), BASE_BLEND, s_fix, P_REF)
    m = eng.omega_model()
    wet = 1 - eng.case0["coke"]["moist_pct"] / 100
    version = m.version.split(":")[0]

    def pair(za, zb, sa_f, sb_f, Pa, Pb):
        """(dK fixed, dK strata, sigma strata) of b minus a."""
        fa, fb = eng.evaluate(za, sa_f, Pa), eng.evaluate(zb, sb_f, Pb)
        sa, sb = replace(sa_f, omega_mode="blend"), replace(sb_f, omega_mode="blend")
        ba, bb = eng.evaluate(za, sa, Pa, full=True), eng.evaluate(zb, sb, Pb, full=True)
        sw = difference_sigma(m, bb["omega_terms"], ba["omega_terms"])
        return fb["K"] - fa["K"], bb["K"] - ba["K"], abs(eng._dKdw) * sw / wet, bb["omega"] - ba["omega"]

    rows = []
    today = _today()
    common = dict(group="MOD", status="Model", grade="B", unit="kg/tHM dry coke", version="", date=today,
                  next_test="T5", validity_window="reference point only (linear response)")
    for fid, label, d in SUBSTITUTIONS:
        sh = dict(BASE_BLEND)
        for k, v in d.items():
            sh[k] = sh.get(k, 0.0) + v
        z1 = _blend_z(eng, z0, sh, s_fix, P_REF)
        kf, ks, sg, dw = pair(z0, z1, s_fix, s_fix, P_REF, P_REF)
        rows.append(dict(common, id=fid, statement=f"{label}, 10 wet pt of ferrous: coke response, Fixed-omega and "
                                                    "STRATA-omega", value=round(ks, 2), se=round(sg, 2),
                         basis=f"Engine at the reference point (base blend {BASE_BLEND}, flux held, {P_REF * 24:,.0f} "
                               f"t/d); Fixed-omega = chemistry only (ASM-01); STRATA-omega = chemistry + shaft "
                               f"efficiency ({version}); se = 1 sigma of the STRATA difference",
                         params=dict(fixed_omega=round(kf, 2), strata_omega=round(ks, 2), difference=round(ks - kf, 2),
                                     sigma_strata=round(sg, 2), d_omega=round(dw, 4))))
    kf, ks, sg, dw = pair(z0, z0, s_fix, s_fix, P_REF, P_REF + 1.0)
    rows.append(dict(common, id="MOD-09", statement="Production +1 t/h at the same burden: coke response, Fixed-omega "
                                                    "and STRATA-omega", value=round(ks, 2), se=round(sg, 2),
                     unit="kg/tHM dry coke per t/h",
                     basis="Heat loss GJ/h held (dilution of the loss per tonne in both models). STRATA 1.4+ has no "
                           "production channel (GAS-16 retired, GAS-25): the TRZ-residence effect of the network is not "
                           "applied, so the STRATA value differs from Fixed-omega only through the other-operation "
                           "channel. Plant data: burnt coke -1.3 +/- 0.4 kg/tHM per t/h within months (GAS-22)",
                     params=dict(fixed_omega=round(kf, 2), strata_omega=round(ks, 2), difference=round(ks - kf, 2),
                                 sigma_strata=round(sg, 2), d_omega=round(dw, 4))))
    s_h = replace(s_fix, humidity_g=s_fix.humidity_g + 1.0)
    kf, ks, sg, dw = pair(z0, z0, s_fix, s_h, P_REF, P_REF)
    rows.append(dict(common, id="MOD-10", statement="Blast moisture +1 g/Nm3 at the same burden: coke response, "
                                                    "Fixed-omega and STRATA-omega", value=round(ks, 2), se=round(sg, 2),
                     unit="kg/tHM dry coke per g/Nm3",
                     basis="Fixed-omega: endothermic water-gas reaction in the balance only; STRATA-omega adds the "
                           "network's response to the moisture through the other-operation channel. STRATA 1.5 has no "
                           "moisture data term (GAS-19 retired); the day's Open-Meteo moisture (DATA-11) acts through the "
                           "balance. Plant data 2026: burnt coke +1.28 +/- 0.56 kg/tHM per g/Nm3 within months",
                     params=dict(fixed_omega=round(kf, 2), strata_omega=round(ks, 2), difference=round(ks - kf, 2),
                                 sigma_strata=round(sg, 2), d_omega=round(dw, 4))))
    s_n = replace(s_fix, nut_plant=s_fix.nut_plant + 10.0)
    kf, ks, sg, dw = pair(z0, z0, s_fix, s_n, P_REF, P_REF)
    rows.append(dict(common, id="MOD-11", statement="Nut coke +10 kg/tHM at the same burden: coke response, "
                                                    "Fixed-omega and STRATA-omega", value=round(ks, 2), se=round(sg, 2),
                     unit="kg/tHM dry coke per 10 kg/tHM nut coke",
                     basis="Fixed-omega: nut-coke carbon and its heat in the balance only; STRATA-omega adds the nut-coke "
                           "term of shaft efficiency (GAS-23) and the network's coke-layer response",
                     params=dict(fixed_omega=round(kf, 2), strata_omega=round(ks, 2), difference=round(ks - kf, 2),
                                 sigma_strata=round(sg, 2), d_omega=round(dw, 4))))
    return rows


def validation_facts(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """VAL rows from the asset's per-family out-of-sample table."""
    if spec is None:
        with open(MODEL_PATH, encoding="utf-8") as fh:
            spec = json.load(fh)
    val = spec.get("validation") or {}
    fams = val.get("by_blend") or []
    rows = []
    today = _today()
    for i, f in enumerate(fams, start=1):
        comp = (f"pellet {f['pellet']:.1f}, CLO {f['clo']:.1f}, ROM {f['rom']:.1f}, Gomti {f['gomti']:.1f}, "
                f"oversize {f['oversize']:.1f} % of ferrous (mean)")
        rows.append(dict(
            id=f"VAL-{i:02d}", group="VAL", statement=f"Out-of-sample burnt coke, {f['family']}: realised against "
                                                     "Fixed-omega and STRATA-omega",
            value=f["strata"], unit="kg/tHM dry burnt coke (mean)", se=f["se_bias"],
            status=f["verdict"], grade="B" if f["n"] >= 10 else "D",
            basis=f"{f['n']} days, 1 Jan - 7 Oct 2026, day minus 14-day anchor, parameters from earlier days only; "
                  f"{comp}. Realised {f['realised']:.1f}, Fixed-omega {f['fixed_omega']:.1f}, STRATA {f['strata']:.1f}; "
                  f"RMSE {f['rmse_fixed']:.2f} -> {f['rmse_strata']:.2f}; skill {f['skill']:+.2f}; STRATA bias "
                  f"{f['bias']:+.2f} +/- " + (f"{f['se_bias']:.2f}" if f.get("se_bias") is not None else "n/a"),
            literature="", next_test="T5" if f["verdict"] != "Supported" else "", version="", date=today,
            note=val.get("verdict_rule", ""),
            params=dict(n=f["n"], realised=f["realised"], fixed_omega=f["fixed_omega"], strata_omega=f["strata"],
                        rmse_fixed=f["rmse_fixed"], rmse_strata=f["rmse_strata"], skill=f["skill"], bias=f["bias"],
                        se_bias=f["se_bias"], composition=dict(pellet=f["pellet"], clo=f["clo"], rom=f["rom"],
                                                               gomti=f["gomti"], oversize=f["oversize"]))))
    return rows


def write_register(rows: list[dict[str, Any]], version: str, change: str, path=REGISTER_PATH) -> dict[str, Any]:
    """Replace the MOD and VAL rows of the register, bump its version and log the change."""
    with open(path, encoding="utf-8") as fh:
        reg = json.load(fh)
    groups = {r["group"] for r in rows}
    today = _today()
    for r in rows:
        r["version"] = version
    reg["facts"] = [f for f in reg["facts"] if f["group"] not in groups] + rows
    for st in sorted({r["status"] for r in rows}):
        if st not in reg["statuses"]:
            reg["statuses"].append(st)
    if reg["version"] != version:
        reg["revision_log"].insert(0, dict(version=version, date=today, change=change))
    reg["version"], reg["date"] = version, today
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return reg


def _fmt(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="replace the MOD and VAL rows of the fact register")
    ap.add_argument("--version", default=None, help="register version to write (default: unchanged)")
    ap.add_argument("--change", default="MOD and VAL facts regenerated (utils.burden_engine.model_facts).")
    a = ap.parse_args()
    rows = model_facts() + validation_facts()
    for r in rows:
        p = r["params"]
        if r["group"] == "MOD":
            print(f"{r['id']} {r['statement'][:60]:60s} Fixed {_fmt(p['fixed_omega'])}  STRATA {_fmt(p['strata_omega'])}"
                  f" +/- {p['sigma_strata']:.2f}")
        else:
            print(f"{r['id']} {r['statement'][24:84]:60s} {r['status']:14s} realised {p['realised']:.1f} Fixed "
                  f"{p['fixed_omega']:.1f} STRATA {p['strata_omega']:.1f} (n {p['n']})")
    if a.write:
        with open(REGISTER_PATH, encoding="utf-8") as fh:
            v = a.version or json.load(fh)["version"]
        write_register(rows, v, a.change)
        print(f"written: {REGISTER_PATH} v{v}")


if __name__ == "__main__":
    main()
