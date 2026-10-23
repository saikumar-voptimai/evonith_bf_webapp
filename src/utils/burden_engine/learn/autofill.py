"""Fact-register proposals of STRATA-Learn.

Rules
  automatic   value / se / params of an existing fact when the new value lies within 2 registered standard errors of
              the registered value (GAS-14 scales, GAS-19 moisture coefficient); day counts and means of VAL facts
  approval    any status change (e.g. VAL Modelled only -> Supported, Supported -> Disputed); a value outside 2 se;
              growth of the GAS-18 support envelope; promotion of the challenger (gate)
Nothing is written by ``propose``; ``apply`` writes the approved items with a revision-log entry."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from utils.burden_engine.learn.evidence import PARAMS

GAS14_KEYS = {"lam_pellet": ("lam_pellet", "sd_pellet"), "lam_lump": ("lam_lump", "sd_lump"),
              "lam_production": ("lam_production", "sd_production"), "lam_other": ("lam_other", "sd_other")}


def _neff(e: np.ndarray) -> float:
    if len(e) < 4:
        return float(len(e))
    r = np.corrcoef(e[1:], e[:-1])[0, 1]
    r = 0.0 if not np.isfinite(r) else max(0.0, r)
    return len(e) * (1 - r) / (1 + r)


def family_table(df: pd.DataFrame, pred_col: str) -> list[dict[str, Any]]:
    """Per-family out-of-sample burnt coke with the VAL verdict rule; df needs K_burnt, K_held, dKdw, family, pred_col."""
    d = df.dropna(subset=["K_burnt", "K_held", "dKdw", pred_col]).copy()
    d["K_strata"] = d.K_held + d.dKdw * d[pred_col]
    d["e0"], d["e1"] = d.K_burnt - d.K_held, d.K_burnt - d.K_strata
    groups = list(d.groupby("family")) + [("All days", d)]
    if "sh_rom" in d:
        groups.append(("ROM >= 10 % (any family)", d[d.sh_rom >= 10]))
    if "sh_pellet" in d:
        groups.append(("Pellet >= 12 % (any family)", d[d.sh_pellet >= 12]))
    rows = []
    for fam, g in groups:
        n = len(g)
        if n == 0:
            continue
        mse0, mse1 = float((g.e0 ** 2).mean()), float((g.e1 ** 2).mean())
        bias = float(g.e1.mean())
        se = float(g.e1.std(ddof=1) / math.sqrt(max(_neff(g.e1.to_numpy()), 1.0))) if n > 1 else float("nan")
        skill = 1 - mse1 / mse0 if mse0 > 0 else float("nan")
        verdict = ("Modelled only" if n < 10 else "Supported" if skill > 0 and abs(bias) < 2 * se else
                   "Disputed" if skill < -0.05 or abs(bias) >= 2 * se else "Neutral")
        rows.append(dict(family=fam, n=n, realised=float(g.K_burnt.mean()), fixed_omega=float(g.K_held.mean()),
                         strata=float(g.K_strata.mean()), rmse_fixed=math.sqrt(mse0), rmse_strata=math.sqrt(mse1),
                         skill=skill, bias=bias, se_bias=se, verdict=verdict))
    return rows


def propose(reg_data: Mapping[str, Any], kalman_summary: Mapping[str, Any], fam_rows: Iterable[Mapping[str, Any]],
            support: Mapping[str, tuple[float, float]] | None = None, gate_result: Mapping[str, Any] | None = None,
            drift: Mapping[str, Any] | None = None, today: str = "", support_tol: float = 0.5) -> dict[str, Any]:
    facts = {f["id"]: f for f in reg_data["facts"]}
    items: list[dict[str, Any]] = []
    mean, sd = kalman_summary["mean"], kalman_summary["sd"]

    g14 = facts.get("GAS-14")
    if g14:
        p = g14.get("params") or {}
        new_p = dict(p)
        auto = True
        for k, (pk, sk) in GAS14_KEYS.items():
            old, old_sd = p.get(pk), p.get(sk)
            new_p[pk], new_p[sk] = round(mean[k], 3), round(sd[k], 3)
            if old is not None and old_sd and abs(mean[k] - old) > 2 * old_sd:
                auto = False
        items.append(dict(id="GAS-14", kind="value", automatic=auto,
                          change=dict(value=round(mean["lam_pellet"], 3), se=round(sd["lam_pellet"], 3), params=new_p),
                          reason="Kalman state of the plant-data scales" + ("" if auto else " (outside 2 se: approval)")))
    g19 = facts.get("GAS-19")
    if g19 and g19.get("se"):
        a, s = mean["a_moisture"], sd["a_moisture"]
        auto = abs(a - g19["value"]) <= 2 * g19["se"]
        items.append(dict(id="GAS-19", kind="value", automatic=auto, change=dict(value=round(a, 5), se=round(s, 5)),
                          reason="Kalman state of the moisture coefficient" + ("" if auto else " (outside 2 se)")))
    g23 = facts.get("GAS-23")
    if g23 and g23.get("se") and "a_nut" in mean:
        a, s = mean["a_nut"], sd["a_nut"]
        auto = abs(a - g23["value"]) <= 2 * g23["se"]
        items.append(dict(id="GAS-23", kind="value", automatic=auto, change=dict(value=round(a, 6), se=round(s, 6)),
                          reason="Kalman state of the nut-coke coefficient" + ("" if auto else " (outside 2 se)")))
    for r in fam_rows:
        f = next((v for v in facts.values() if v["group"] == "VAL" and f", {r['family']}:" in v["statement"]), None)
        if f is None:
            items.append(dict(id=None, kind="new", automatic=False, change=r, reason=f"new family {r['family']}"))
            continue
        params = dict(f.get("params") or {}, n=r["n"], realised=round(r["realised"], 1),
                      fixed_omega=round(r["fixed_omega"], 1), strata_omega=round(r["strata"], 1),
                      rmse_fixed=round(r["rmse_fixed"], 2), rmse_strata=round(r["rmse_strata"], 2),
                      skill=round(r["skill"], 3), bias=round(r["bias"], 2), se_bias=round(r["se_bias"], 2))
        status_change = r["verdict"] != f["status"]
        items.append(dict(id=f["id"], kind="status" if status_change else "value", automatic=not status_change,
                          change=dict(value=round(r["strata"], 1), se=round(r["se_bias"], 2), params=params,
                                      **({"status": r["verdict"]} if status_change else {})),
                          reason=(f"{f['status']} -> {r['verdict']} (n {r['n']}, skill {r['skill']:+.2f}, bias "
                                  f"{r['bias']:+.1f} +/- {r['se_bias']:.1f})") if status_change else
                                 f"n {f['params'].get('n')} -> {r['n']}"))
    if support:
        g18 = facts.get("GAS-18")
        old = (g18.get("params") or {}).get("support", {}) if g18 else {}
        grown = {k: (min(v[0], old[k][0]), max(v[1], old[k][1])) for k, v in support.items()
                 if k in old and (v[0] < old[k][0] - support_tol or v[1] > old[k][1] + support_tol)}
        if grown:
            items.append(dict(id="GAS-18", kind="support", automatic=False,
                              change=dict(params=dict(g18.get("params") or {}, support=dict(old, **grown))),
                              reason="support envelope grew: " + "; ".join(f"{k} {old[k]} -> {v}" for k, v in grown.items())))
    if gate_result and gate_result.get("promote"):
        items.append(dict(id="ASSET", kind="promotion", automatic=False, change=dict(mean=mean, sd=sd),
                          reason=gate_result["reason"]))
    if drift and drift.get("alarm"):
        items.append(dict(id="GAS-20", kind="drift", automatic=False, change={},
                          reason="CUSUM drift alarm on the champion's errors: recalibration needed"))
    return dict(date=today, items=items, n_auto=sum(i["automatic"] for i in items),
                n_approval=sum(not i["automatic"] for i in items))


def apply(reg_data: Mapping[str, Any], proposal: Mapping[str, Any], approved: Iterable[str] = (),
          version: str | None = None, today: str = "") -> dict[str, Any]:
    """Register with the automatic items and the approved ids applied (asset promotion is not a register change)."""
    reg = copy.deepcopy(dict(reg_data))
    facts = {f["id"]: f for f in reg["facts"]}
    approved = set(approved)
    done = []
    for it in proposal["items"]:
        if it["id"] in (None, "ASSET") or it["id"] not in facts:
            continue
        if not (it["automatic"] or it["id"] in approved):
            continue
        f = facts[it["id"]]
        for k, v in it["change"].items():
            f[k] = v
        f["date"] = today or f.get("date", "")
        if version:
            f["version"] = version
        done.append(it["id"])
    if done and version and version != reg["version"]:
        reg["revision_log"].insert(0, dict(version=version, date=today,
                                           change="STRATA-Learn: " + ", ".join(sorted(set(done))) + " updated"))
        reg["version"] = version
    reg["date"] = today or reg["date"]
    return reg


def params_vector(summary: Mapping[str, Any]) -> np.ndarray:
    return np.array([summary["mean"][p] for p in PARAMS])
