"""Shaft efficiency on the Burden & Fuel Optimiser page: the settings (model, certainty filter, risk aversion, auto
mode), the shaft-efficiency panel of a plan, and the auto-mode candidates with the language-model review.

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

FILTERS = {"any": "Any certainty", "medium": "Medium or better", "high": "High only", "auto": "Auto: rule + review"}
RISK = {0.0: "None (expected cost)", 1.0: "Moderate (+1 σ)", 1.28: "Cautious (+1.28 σ)", 1.64: "Strict (+1.64 σ)"}
BADGE = {"High": ":green-badge[High certainty]", "Medium": ":orange-badge[Medium certainty]",
         "Low": ":red-badge[Low certainty]", "Held": ":gray-badge[ω held]"}


def render_omega_settings(key: str = "bfo_om") -> dict[str, Any]:
    """Model choice, certainty filter and risk aversion. Returns Settings fields plus ``auto``."""
    c1, c2, c3 = st.columns([2, 2, 2])
    blend = c1.toggle("Blend-dependent shaft efficiency", value=True, key=f"{key}_mode",
                      help="On: ω responds to the blend's reducibility and size, where the burden puts ore and coke, "
                           "residence time and gas quantity (BF-02 gas-flow model, scaled by the plant data; GAS-13..18). "
                           "Off: ω held at the calibration for every blend (ASM-01), as before.")
    if not blend:
        c2.caption("ω held at the calibration: a blend changes coke through its chemistry only.")
        return dict(omega_mode="fixed", certainty="any", risk_k=0.0, auto=False)
    filt = c2.selectbox("Certainty filter", list(FILTERS), format_func=FILTERS.get, index=0, key=f"{key}_filter",
                        help="High: σ(coke) ≤ 2 kg/tHM and blend and production inside what BF-02 ran. Medium: σ ≤ 5 and "
                             "within the margin. Auto: the plan under three settings, a fixed rule picks, a language "
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
                    help="1 sigma of the plan's coke relative to the calibration state: data posterior, operating and "
                         "gas-capacity terms, surrogate. "
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
        parts = pd.DataFrame([
            ("Blend: reducibility, size, burden volume (data-scaled)", om["d_blend"]),
            ("Blend: class correction from the plant data", om["d_disc"]),
            ("Operation: coke layers, production (residence), oxygen", om["d_oper"]),
            ("Gas quantity (capacity ratio R)", om["d_R"]),
        ], columns=["Part of Δω against the calibration", "Δω"])
        parts["Coke, kg/tHM dry"] = parts["Δω"] * om["dK_domega"]
        st.dataframe(parts, hide_index=True, width="stretch", column_config={
            "Δω": st.column_config.NumberColumn(format="%+.4f"),
            "Coke, kg/tHM dry": st.column_config.NumberColumn(format="%+.1f")})
        st.caption(f"Capacity ratio R {om['R']:.3f}; blend reduction rate {om['k_blend']:.3f} (sinter at the test size = 1); "
                   f"gas in rings with surplus gas {100 * om['gas_in_excess']:.1f} %; wall ring gas {100 * om['wall_gas_share']:.1f} %. "
                   f"dK/dω {om['dK_domega']:.0f} kg/tHM per unit ω including the gas-quantity loop. Not in σ: the "
                   f"calibration's own error (± {om['cal_se_K']:.1f} kg/tHM, common to every plan) and the day-to-day "
                   f"spread (± {om['day_sd_K']:.1f}). Model {om['version']}.")
        if om["outside"]:
            st.warning("Outside what BF-02 has run: " + "; ".join(om["outside"]) + ".")


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
