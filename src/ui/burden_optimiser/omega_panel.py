"""Shaft efficiency on the Burden & Fuel Optimiser page: the settings (model switch Fixed-ω / STRATA-ω, certainty
filter, risk aversion, auto mode), the shaft-efficiency panel of a plan with its critical assumptions, and the auto-mode
candidates with the language-model review.

The model is ``utils/burden_engine/omega_model.py``; the rule and the review checks are
``utils/burden_engine/auto_review.py``. As with the commentary, the model call is on a button, never automatic.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from utils.burden_engine.auto_review import REVIEW_SYSTEM, parse_review, review_context, ungrounded_numbers

log = logging.getLogger(__name__)

FILTERS = {"low": "Exclude Investigate", "any": "Any (incl. Investigate)", "medium": "Medium or better",
           "high": "High only", "auto": "Auto: rule + review"}
MODELS = {"blend": "STRATA-ω (blend-dependent)", "fixed": "Fixed-ω (Rist baseline)"}
RISK = {0.0: "None (expected cost)", 1.0: "Moderate (+1 σ)", 1.28: "Cautious (+1.28 σ)", 1.64: "Strict (+1.64 σ)"}
BADGE = {"High": ":green-badge[High certainty]", "Medium": ":orange-badge[Medium certainty]",
         "Low": ":red-badge[Low certainty]", "Investigate": ":violet-badge[Investigate: trial before adoption]",
         "Held": ":gray-badge[ω held]"}
N_FLOAT = 3                     # critical assumptions shown on the face of a Low / Investigate / extrapolated result


def render_omega_settings(key: str = "bfo_om") -> dict[str, Any]:
    """Model choice, certainty filter and risk aversion. Returns Settings fields plus ``auto``."""
    c1, c2, c3 = st.columns([2, 2, 2])
    mode = c1.radio("Shaft-efficiency model", list(MODELS), format_func=MODELS.get, index=0, key=f"{key}_mode",
                    horizontal=True,
                    help="STRATA-ω: ω responds to the blend's reducibility (plant RI tests) and size, residence time in "
                         "the thermal reserve zone, the coke layers, production, oxygen, nut coke and blast moisture. "
                         "The STRATA-Flow gas-flow network of BF-02 is solved for every plan (no surrogate; about 1 s "
                         "per solve, one solve per optimiser iteration), scaled by the plant data and tested out of "
                         "sample (STRATA 1.3, GAS-13..23). Fixed-ω: ω held at the calibration for every blend (Rist "
                         "baseline, ASM-01).")
    if mode == "fixed":
        c2.caption("Fixed-ω: ω held at the calibration; a blend changes coke through its chemistry only.")
        return dict(omega_mode="fixed", certainty="any", risk_k=0.0, auto=False)
    filt = c2.selectbox("Certainty filter", list(FILTERS), format_func=FILTERS.get, index=0, key=f"{key}_filter",
                        help="High: σ(coke) ≤ 2 kg/tHM and blend and production inside what BF-02 ran. Medium: σ ≤ 5 and "
                             "within the margin. Investigate: pellet above 25 % of ferrous or σ(coke) above 8 kg/tHM "
                             "(beyond common practice; trial before adoption). Exclude Investigate (default) keeps "
                             "every other plan. Auto: the plan under three settings, a fixed rule picks, a language "
                             "model reviews the pick.")
    risk = c3.select_slider("Risk aversion", options=list(RISK), value=0.0, format_func=RISK.get, key=f"{key}_risk",
                            disabled=(filt == "auto"),
                            help="Objective = cost + k × σ(cost). k = 1.28 judges each plan at the cost it beats nine "
                                 "times in ten.")
    return dict(omega_mode="blend", certainty="any" if filt == "auto" else filt, risk_k=float(risk), auto=filt == "auto")


def render_omega_result(eng: Any, r: Mapping[str, Any], today: Mapping[str, Any] | None = None) -> None:
    """ω of the plan, its parts, σ(coke) and the certainty class; the saving against today with its σ."""
    om = r.get("omega") or {}
    if om.get("mode") != "blend":
        st.caption(f"Shaft efficiency held at the calibration, ω = {om.get('value', float('nan')):.3f} (ASM-01). "
                   "Switch on blend-dependent shaft efficiency to see the blend's effect on gas utilisation and the "
                   "uncertainty of each plan.")
        return
    with st.container(border=True):
        st.markdown(f"###### 🌀 &nbsp;Shaft efficiency &nbsp;{BADGE.get(om['certainty'], '')}", unsafe_allow_html=True)
        k = st.columns(4)
        k[0].metric("ω (plan)", f"{om['value']:.3f}", f"{om['value'] - om['omega_cal']:+.3f} vs calibration",
                    delta_color="normal", help=f"Calibration ω {om['omega_cal']:.3f}.")
        k[1].metric("σ coke (kg/tHM)", f"± {om['sigma_K']:.1f}",
                    help="1 sigma of the plan's coke relative to the calibration state: blend scales (data posterior), "
                         "operating scales, blast-moisture and nut-coke coefficients. "
                         + ", ".join(f"{p} {v:.1f}" for p, v in om["sigma_parts_K"].items()) + " kg/tHM.")
        k[2].metric("σ cost (Rs/tHM)", f"± {r['cost'].get('sigma', 0.0):,.0f}")
        if today is not None and (today.get("omega") or {}).get("mode") == "blend":
            sg = eng.saving_sigma(r, today)
            sv = today["cost"]["total"] - r["cost"]["total"]
            z = sv / sg["sigma_cost"] if sg and sg["sigma_cost"] > 0 else float("nan")
            k[3].metric("Saving vs today (Rs/tHM)", f"{sv:+,.0f} ± {sg['sigma_cost']:,.0f}" if sg else f"{sv:+,.0f}",
                        help="Positive = the plan saves. σ of the difference: parameters shared by both plans cancel.")
            if math.isfinite(z):
                k[3].caption(f"{z:.1f} σ" + (" — inside the error band" if abs(z) < 1 else ""))
        else:
            k[3].metric("Day-to-day spread (kg/tHM)", f"± {om['day_sd_K']:.1f}",
                        help="Daily scatter of ω around the model, as coke; it averages out over days.")
        rows = [("Blend: reducibility, size, burden volume (model)", om["d_blend"]),
                ("Blend: plant-data scale (pellet, lump ore)", om["d_disc"])]
        if "d_prod" in om and abs(om.get("d_prod", 0.0)) + abs(om.get("d_rest", 0.0)) > 0 and \
                abs(om["d_prod"] + om["d_rest"] - om["d_oper"]) < 1e-9:
            rows += [("Operation: production (TRZ residence), scaled", om["d_prod"]),
                     ("Operation: coke layers, fuels, oxygen, scaled", om["d_rest"])]
        else:
            rows.append(("Operation: coke layers, production (residence), oxygen", om["d_oper"]))
        rows.append(("Blast moisture (weather record, RH estimate)", om.get("d_moist", 0.0)))
        if abs(om.get("d_nut", 0.0)) > 0:
            rows.append(("Nut coke against the calibration (GAS-23)", om["d_nut"]))
        if abs(om.get("d_R", 0.0)) > 0:
            rows.append(("Gas quantity (capacity ratio R)", om["d_R"]))
        parts = pd.DataFrame(rows, columns=["Part of Δω against the calibration", "Δω"])
        parts["Coke, kg/tHM dry"] = parts["Δω"] * om["dK_domega"]
        st.dataframe(parts, hide_index=True, width="stretch", column_config={
            "Δω": st.column_config.NumberColumn(format="%+.4f"),
            "Coke, kg/tHM dry": st.column_config.NumberColumn(format="%+.1f")})
        st.caption(f"Capacity ratio R {om['R']:.3f}; blend reduction rate {om['k_blend']:.3f} (sinter at the test size = 1); "
                   f"gas in rings with surplus gas {100 * om['gas_in_excess']:.1f} %; wall ring gas {100 * om['wall_gas_share']:.1f} %. "
                   f"dK/dω {om['dK_domega']:.0f} kg/tHM per unit ω including the coke-layer loop. Not in σ: the "
                   f"calibration's own error (± {om['cal_se_K']:.1f} kg/tHM, common to every plan) and the day-to-day "
                   f"spread (± {om['day_sd_K']:.1f}). Model {om['version']}.")
        net = om.get("network") or {}
        if net.get("backend") == "network":
            dist = max((net.get("distance") or {}).values(), default=float("nan"))
            st.caption(f"STRATA-Flow network: {net.get('solves', 0)} solves so far for this calibration "
                       f"({net.get('seconds', 0):.0f} s; Jacobian {net.get('jacobian', '')}). This plan's ω comes from "
                       "solves at its blend, blend + production and full-operation states"
                       + (" (exact)." if dist <= 1e-9 else f" (largest distance from a solve {dist:.2f} scaled units: "
                          "under 1 pt of a share or 5 kg of coke)."))
        fixed_k = _fixed_omega_coke(eng, r)
        if fixed_k is not None:
            st.caption(f"Same blend under Fixed-ω (Rist baseline): coke {fixed_k:.1f} kg/tHM dry; STRATA-ω gives "
                       f"{r['out']['K']:.1f} ({r['out']['K'] - fixed_k:+.1f}). The difference is the blend's and the "
                       "operation's effect on gas utilisation that the fixed model does not see.")
        if om.get("investigate"):
            st.error("**Investigate tier** — " + "; ".join(om["investigate"]) + ". Beyond common practice: the "
                     "coke prediction rests on the assumptions below, not on BF-02 evidence. Trial before adoption.")
        if om["outside"]:
            st.warning("Outside what BF-02 has run: " + "; ".join(om["outside"]) + ".")
        render_critical_assumptions(eng, r, om)


def _fixed_omega_coke(eng: Any, r: Mapping[str, Any]) -> float | None:
    """Dry coke of the plan's blend with ω held at the calibration (Fixed-ω), for the side-by-side comparison."""
    try:
        from dataclasses import replace

        import numpy as np

        from utils.burden_engine.engine import Settings

        s = Settings(**{k: v for k, v in r["settings"].items() if k in Settings.__dataclass_fields__})
        o = eng.evaluate(np.asarray(r["z"], float), replace(s, omega_mode="fixed"), float(r["P_th"]))
        return float(o["K"])
    except Exception:  # a comparison must never break the results page
        log.exception("Fixed-omega comparison failed")
        return None


def render_critical_assumptions(eng: Any, r: Mapping[str, Any], om: Mapping[str, Any]) -> None:
    """Coke effect of each critical assumption taking another credible value. On a Low / Investigate / extrapolated
    result the largest ones are floated on the face of the panel; otherwise they sit in an expander."""
    try:
        from utils.burden_engine.engine import Settings

        s = Settings(**{k: v for k, v in r["settings"].items() if k in Settings.__dataclass_fields__})
        rows = eng.critical_assumptions(r, s)
    except Exception:  # the sensitivity table must never break the results page
        log.exception("critical assumptions failed")
        return
    if not rows:
        return
    T = pd.DataFrame([dict(Assumption=x["assumption"], Value=x["value"], Alternative=x["alternative"],
                           **{"Coke effect, kg/tHM": (f"± {x['d_coke_kg']:.1f}" if x.get("symmetric") else
                                                      f"{x['d_coke_lo_kg']:+.1f} / {x['d_coke_hi_kg']:+.1f}"
                                                      if "d_coke_lo_kg" in x else f"{x['d_coke_kg']:+.1f}")},
                           Basis=x["basis"]) for x in rows])
    flag = om.get("certainty") in ("Low", "Investigate") or bool(om.get("outside"))
    if flag:
        top = rows[:N_FLOAT]
        st.warning("**Critical assumptions behind this extrapolated result** (coke effect if the assumption takes "
                   "the alternative value): " + "; ".join(
                       f"{x['assumption']} {x['value']} → {x['alternative']}: "
                       + (f"± {x['d_coke_kg']:.1f}" if x.get("symmetric") else f"{x['d_coke_kg']:+.1f}")
                       + " kg/tHM" for x in top)
                   + ". Reducibility of pellet and lump ores is a literature value (no plant RI test).")
    with st.expander("Critical assumptions (sensitivity of this plan's coke)", expanded=flag):
        st.dataframe(T, hide_index=True, width="stretch")
        st.caption("Each row changes one assumption, re-anchors the model on the calibration and re-evaluates this "
                   "plan at the same coke rate; effect on coke through dK/dω. RI = ISO 7215 reducibility index.")


def _generate(system_prompt: str, user_prompt: str) -> str:
    from agents.llm.llm_client import get_llm_client

    return str(get_llm_client().generate(system_prompt, user_prompt) or "").strip()


def render_auto(auto: Mapping[str, Any] | None, names: Mapping[str, str], *, mode_label: str,
                calibration: Mapping[str, Any] | None = None, key: str = "bfo_auto") -> None:
    """Candidates, the rule's pick and (on a button) the language-model review with its checks."""
    st.markdown("### 🤖 Auto mode: candidates, rule and review")
    if not auto:
        st.info("Auto mode needs blend-dependent shaft efficiency and a feasible plan.")
        return
    rows = auto["rows"]
    T = pd.DataFrame([dict(ID=r["id"], Candidate=r["label"], Limits=("met" if r.get("feasible") else "broken")
                           if r.get("available") else "n/a", **({} if not r.get("available") else {
                               "Cost, Rs/tHM": r["cost"], "σ cost": r["sigma_cost"], "Cost + 1σ": r["cost_plus_sigma"],
                               "Coke dry": r["coke_dry"], "σ coke": r["sigma_coke"], "ω": r["omega"],
                               "Certainty": r["certainty"], "Saving vs today": r.get("saving_vs_today"),
                               "σ saving": r.get("sigma_saving")}))
                      for r in rows])
    num = st.column_config.NumberColumn
    st.dataframe(T, hide_index=True, width="stretch", column_config={
        "Cost, Rs/tHM": num(format="%,.0f"), "σ cost": num(format="%,.0f"), "Cost + 1σ": num(format="%,.0f"),
        "Coke dry": num(format="%.1f"), "σ coke": num(format="%.1f"), "ω": num(format="%.3f"),
        "Saving vs today": num(format="%+,.0f"), "σ saving": num(format="%,.0f")})
    with st.expander("Blends of the candidates"):
        B = pd.DataFrame({r["id"]: r["shares"] for r in rows if r.get("available")}).fillna(0.0)
        B.index = [names.get(i, i) for i in B.index]
        st.dataframe(B, width="stretch", column_config={c: num(format="%.1f") for c in B.columns})
    pid = auto.get("pick")
    st.success(f"**Rule pick: {pid or 'none'}.** {auto.get('reason', '')}")
    ctx = review_context(rows, pid, auto.get("reason", ""), names=names, mode_label=mode_label, calibration=calibration)
    digest = hashlib.sha1(ctx.encode("utf-8")).hexdigest()[:12]
    store = st.session_state.setdefault(key, {})
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.caption("The review is generated text from the table above and the plant policy. It cannot change the pick: a "
               "different choice is shown as a disagreement. Every number it writes is checked against the table.")
    if c2.button("Review the pick", type="primary", key=f"{key}_go", width="stretch"):
        with st.spinner("Reviewing the candidates…"):
            try:
                store[digest] = _generate(REVIEW_SYSTEM, ctx)
            except Exception as exc:  # noqa: BLE001 - never break the results page
                log.exception("auto review failed")
                st.error("Could not generate the review: " + (str(exc).splitlines() or [""])[0][:200])
    text = store.get(digest)
    if text:
        ids = [r["id"] for r in rows if r.get("available")]
        rev, problems = parse_review(text, ids)
        if rev is None:
            st.error("The review could not be read: " + "; ".join(problems))
            st.code(text)
        else:
            bad = ungrounded_numbers(rev, ctx)
            agree = rev.get("choice") == pid
            with st.container(border=True):
                st.markdown(f"**Review: {rev.get('choice')}** "
                            + (":green-badge[agrees with the rule]" if agree else ":red-badge[DISAGREES with the rule]"))
                st.markdown(str(rev.get("summary", "")))
                for title, k in (("Reasons", "reasons"), ("Risks", "risks"), ("Monitoring", "monitoring")):
                    if rev.get(k):
                        st.markdown(f"*{title}*\n" + "\n".join(f"- {x}" for x in rev[k]))
            if problems:
                st.warning("Review format problems: " + "; ".join(problems))
            if bad:
                st.warning("Numbers in the review not found in the table (check before acting): " + ", ".join(bad))
            if not agree:
                st.info("The rule's pick stands. The review's disagreement is a prompt to check the candidates above, "
                        "not a replacement for the rule.")
    with st.expander("What the review is given"):
        st.code(REVIEW_SYSTEM + "\n\n" + ctx, language="markdown")
