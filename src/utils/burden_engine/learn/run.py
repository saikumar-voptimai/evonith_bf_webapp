"""Nightly shadow run of STRATA-Learn.

    python -m utils.burden_engine.learn.run [--evidence PATH] [--add CSV] [--out DIR] [--deployed YYYY-MM-DD]

1  evidence: the store at --evidence (default: the seed), plus the rows of --add (same columns), saved back
2  challenger: random-walk Kalman filter from the priors over every evidence day (one-step predictions out of sample)
3  champion: the deployed asset's scales; gate on the days after --deployed (default: the asset date)
4  drift: CUSUM on the champion's errors after deployment, else on the challenger's one-step errors
5  register proposal (automatic items and items for approval) and per-family table from the one-step predictions
Outputs in --out: strata_learn_state.json, strata_learn_history.csv, register_proposal.json, strata_learn_report.md.
Nothing in the asset, the register or the page is changed."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.burden_engine.facts import REGISTER_PATH
from utils.burden_engine.learn.autofill import family_table, propose
from utils.burden_engine.learn.evidence import PARAMS, REGRESSORS, EvidenceStore
from utils.burden_engine.learn.gate import cusum, gate
from utils.burden_engine.learn.kalman import RandomWalkKalman
from utils.burden_engine.omega_model import MODEL_PATH

EVAL_FROM = "2026-01-01"          # first 16 days of the record are the filter's burn-in from the priors


def shadow_run(store: EvidenceStore, spec: dict, reg_data: dict, deployed: str | None = None,
               eval_from: str = EVAL_FROM) -> dict:
    cs = spec["channel_scales"]
    rho, sig = float(cs["rho"]), float(cs["sigma_e"])
    dates, X, y = store.matrix()
    kf = RandomWalkKalman.from_prior(rho, sig)
    hist = pd.DataFrame(kf.run(dates, X, y))
    hist["date"] = pd.to_datetime(hist["date"])
    theta_c = np.asarray(cs["mean"], float)
    hist["pred_champion"] = X @ theta_c
    df = store.df.dropna(subset=list(REGRESSORS) + ["dobs"]).reset_index(drop=True)
    df["pred_challenger"] = hist["pred"].to_numpy()
    df["pred_champion"] = hist["pred_champion"].to_numpy()
    if deployed is None:
        m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", spec.get("version", ""))
        deployed = m.group(1) if m else str(df["date"].max().date())
    after = df["date"] > pd.Timestamp(deployed)
    g = gate(df.loc[after, "dobs"], df.loc[after, "pred_champion"], df.loc[after, "pred_challenger"])
    if after.sum() >= 2:
        dr = cusum((df.loc[after, "dobs"] - df.loc[after, "pred_champion"]).to_numpy(), rho, sig)
        dr["basis"] = "champion errors after deployment"
    else:
        dr = cusum((df["dobs"] - df["pred_challenger"]).to_numpy(), rho, sig)
        dr["basis"] = "challenger one-step errors over the evidence (no post-deployment days yet)"
    dr.pop("path", None)
    if dr.get("first_alarm") is not None:
        base = df.loc[after] if after.sum() >= 2 else df
        dr["first_alarm_date"] = str(base["date"].iloc[dr["first_alarm"]].date())
    fam = family_table(df[df["date"] >= pd.Timestamp(eval_from)], "pred_challenger")
    summ = kf.summary()
    skill = {}
    for lab, col in (("challenger (one-step, out of sample)", "pred_challenger"),
                     ("champion (deployed asset; in sample before deployment)", "pred_champion")):
        e1, e0 = df["dobs"] - df[col], df["dobs"]
        d = df.dropna(subset=["K_burnt"])
        k1 = d.K_burnt - (d.K_held + d.dKdw * d[col])
        k0 = d.K_burnt - d.K_held
        ev = df["date"] >= pd.Timestamp(eval_from)
        skill[lab] = dict(omega_day=float(1 - (e1[ev] ** 2).mean() / (e0[ev] ** 2).mean()),
                          coke_day=float(1 - (k1[d.date >= eval_from] ** 2).mean() / (k0[d.date >= eval_from] ** 2).mean()))
    prop = propose(reg_data, summ, fam, support=store.support(), gate_result=g, drift=dr,
                   today=str(pd.Timestamp.now(tz="Asia/Kolkata").date()))
    return dict(state=summ, history=hist, gate=g, drift=dr, families=fam, proposal=prop, skill=skill,
                deployed=deployed, n_days=len(df), champion=dict(zip(PARAMS, map(float, theta_c))))


def report_md(res: dict) -> str:
    s, g, dr = res["state"], res["gate"], res["drift"]
    L = ["# STRATA-Learn shadow run", "",
         (f"Evidence: {res['n_days']} days to {s['last_date']}; deployed asset {res['deployed']}; mode: shadow (no "
          "change to the asset, the register or the page)."), "", "## Plant-data scales", "",
         "| Parameter | Champion (asset) | Challenger (Kalman) | sd |", "|---|---:|---:|---:|"]
    for p in PARAMS:
        L.append(f"| {p} | {res['champion'][p]:.4g} | {s['mean'][p]:.4g} | {s['sd'][p]:.3g} |")
    L += ["", f"## Out-of-sample skill since {EVAL_FROM} (against Fixed-omega)", "", "| Predictor | omega, day | burnt coke, day |",
          "|---|---:|---:|"]
    for k, v in res["skill"].items():
        L.append(f"| {k} | {v['omega_day']:+.3f} | {v['coke_day']:+.3f} |")
    L += ["", "## Gate and drift", "", f"* Gate: {g['status']} - {g['reason']}.",
          f"* Drift (CUSUM k 0.5, h 5; {dr['basis']}): {'ALARM' if dr['alarm'] else 'no alarm'} "
          f"(S+ {dr['s_plus']:.2f}, S- {dr['s_minus']:.2f})"
          + (f"; first excursion {dr.get('first_alarm_date')}" if dr.get("ever_alarm") else "") + ".", "",
          "## Blend families (one-step predictions of the challenger)", "",
          "| Family | n | Realised | Fixed-ω | STRATA | Skill | Bias ± SE | Verdict |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in res["families"]:
        L.append(f"| {r['family']} | {r['n']} | {r['realised']:.1f} | {r['fixed_omega']:.1f} | {r['strata']:.1f} | "
                 f"{r['skill']:+.2f} | {r['bias']:+.1f} ± {r['se_bias']:.1f} | {r['verdict']} |")
    p = res["proposal"]
    L += ["", f"## Register proposal: {p['n_auto']} automatic, {p['n_approval']} for approval", "",
          "| Fact | Kind | Automatic | Reason |", "|---|---|---|---|"]
    for it in p["items"]:
        L.append(f"| {it['id']} | {it['kind']} | {'yes' if it['automatic'] else 'approval'} | {it['reason']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="STRATA-Learn nightly shadow run")
    ap.add_argument("--evidence", default=None, help="evidence store CSV (default: the shipped seed, in memory)")
    ap.add_argument("--add", default=None, help="CSV of new day rows to append to the store")
    ap.add_argument("--out", default="strata_learn_out")
    ap.add_argument("--deployed", default=None)
    a = ap.parse_args()
    store = EvidenceStore(a.evidence)
    if a.add:
        n = store.append(pd.read_csv(a.add, parse_dates=["date"]))
        print(f"appended {n} new days")
        if a.evidence:
            store.save()
    with open(MODEL_PATH, encoding="utf-8") as fh:
        spec = json.load(fh)
    with open(REGISTER_PATH, encoding="utf-8") as fh:
        reg = json.load(fh)
    res = shadow_run(store, spec, reg, a.deployed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res["history"].to_csv(out / "strata_learn_history.csv", index=False, float_format="%.6g")
    with open(out / "strata_learn_state.json", "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in res.items() if k != "history"}, fh, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(out / "register_proposal.json", "w", encoding="utf-8") as fh:
        json.dump(res["proposal"], fh, indent=1, default=str)
    (out / "strata_learn_report.md").write_text(report_md(res), encoding="utf-8")
    print(report_md(res))


if __name__ == "__main__":
    main()
