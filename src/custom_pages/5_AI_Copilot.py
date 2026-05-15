"""AI Copilot page.

Three tabs:
  1. Anomalies       — live InfluxDB snapshot + channeling propensity + LLM anomaly summary
  2. Unit Cost & Burden Distribution — static analysis findings from BURDEN_UNITCOST.md

Analysis findings are maintained in ``src/assets/data/copilot_analysis/``.
To update findings after a new regression run, edit the .md files there — no Python changes needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.anomaly_propensity import Channeling
from utils.copilot.data import fetch_recent_online
from agents.llm.llm_client import OPENAI_MODEL, call_llm
from utils.copilot.prompts import (
    ANOMALY_SYSTEM,
    BURDEN_SYSTEM,
    build_anomaly_prompt,
    build_burden_prompt,
    load_burden_findings,
    load_sensor_desc,
)

# ── Page config & header ──────────────────────────────────────────────────────

st.title("🤖 AI Copilot")

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Configuration")
run_llm = st.sidebar.toggle("Run LLM Analysis", key="run_llm_button")
st.sidebar.caption(f"Model: `{OPENAI_MODEL}`")

# ── Live data (shared across tabs) ────────────────────────────────────────────

combined_df = fetch_recent_online()
if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
    st.dataframe(combined_df.sort_index(), width='stretch')
else:
    st.info("No live data returned. Will retry on next refresh.")

# ── Tabs ──────────────────────────────────────────────────────────────────────

anomaly_tab, burden_tab = st.tabs(["🔍 Anomalies", "📦 Unit Cost & Burden Dist"])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Anomalies
# ═══════════════════════════════════════════════════════════════════════════════

with anomaly_tab:
    st.subheader("Anomalies")

    # Operator notes (form avoids rerun on every keystroke)
    with st.form("operator_notes_form", clear_on_submit=False):
        notes = st.text_area("Operator notes (optional)", key="operator_notes")
        if st.form_submit_button("Save note"):
            st.success("Note captured in session.")

    st.divider()

    # ── Channeling propensity ─────────────────────────────────────────────────
    detector = Channeling()

    with st.expander("Channeling propensity", expanded=True, key="copilot_channeling"):
        c1, c2, c3 = st.columns([1, 1, 2])
        enable = c1.toggle("Enable", value=False, key="channeling_propensity_enable")
        live = c2.toggle(
            "Auto-refresh (10 min)", value=True, key="channeling_propensity_autorefresh"
        )
        lookback = c3.selectbox(
            "Lookback",
            options=[
                "last 1 hour",
                "last 6 hours",
                "last 12 hours",
                "last 1 day",
                "last 3 days",
                "last 1 week",
                "last 2 weeks",
                "last 1 month",
                "last 2 months",
                "last 3 months",
            ],
            index=2,
            key="channeling_propensity_lookback",
        )

        @st.cache_data(ttl=600, show_spinner=False)
        def _fetch_cached(tr: str, window_by: str) -> pd.DataFrame:
            return fetch_recent_online(tr=tr, window_by=window_by)

        run_every = 600 if (enable and live) else None

        @st.fragment(run_every=run_every)
        def render_channeling() -> None:
            with st.spinner("Computing propensities…"):
                df_in = _fetch_cached(tr=lookback, window_by="1 minute")
                scores = detector.score_timeseries(df_in)

            if scores.empty:
                st.info("No series to plot for this propensity.")
                return

            scores = scores[scores["channeling_score"].between(0, 1.3)].dropna(
                subset=["channeling_score"]
            )
            last = float(scores["channeling_score"].iloc[-1])
            prev = (
                float(scores["channeling_score"].iloc[-2])
                if len(scores) > 1
                else np.nan
            )
            delta = (last - prev) if np.isfinite(prev) else None

            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Current channeling score",
                f"{last:.2f}",
                delta=f"{delta:+.2f}" if delta is not None else None,
            )
            k2.metric(
                "Components used (of 7)", int(scores["n_components_used"].iloc[-1])
            )
            k3.caption(f"Last updated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")

            _sc = scores[["channeling_score"]].copy()
            if _sc.index.tz is None:
                _sc.index = _sc.index.tz_localize("UTC")
            st.line_chart(_sc.tz_convert("Asia/Kolkata"), height=220)

            # Circumferential quadrant profile
            q_labels = ["Q1", "Q2", "Q3", "Q4"]
            q_keys = ["q1_score", "q2_score", "q3_score", "q4_score"]
            latest = scores.iloc[-1]
            q_vals = [
                float(latest.get(k, 0)) if np.isfinite(float(latest.get(k, 0))) else 0.0
                for k in q_keys
            ]

            circ_col, info_col = st.columns([3, 2])
            with circ_col:
                vmin, vmax = min(q_vals), max(q_vals)

                def _yrb(frac: float) -> str:
                    f = max(0.0, min(1.0, frac))
                    if f <= 0.5:
                        return f"rgb(255,{int(255*(1-2*f))},0)"
                    return f"rgb({int(255*(2-2*f))},0,0)"

                fig_q = go.Figure()
                R_IN, R_OUT, SEGS = 3, 8, 60
                for qi in range(4):
                    th0 = qi * 90
                    thetas = np.linspace(
                        np.deg2rad(th0), np.deg2rad(th0 + 90), SEGS + 1
                    )
                    xs = np.concatenate(
                        [
                            R_OUT * np.cos(thetas),
                            R_IN * np.cos(thetas[::-1]),
                            [R_OUT * np.cos(thetas[0])],
                        ]
                    )
                    ys = np.concatenate(
                        [
                            R_OUT * np.sin(thetas),
                            R_IN * np.sin(thetas[::-1]),
                            [R_OUT * np.sin(thetas[0])],
                        ]
                    )
                    frac = (q_vals[qi] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    ang_mid = np.deg2rad(th0 + 45)
                    lbl_r = (R_IN + R_OUT) / 2
                    fig_q.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            fill="toself",
                            fillcolor=_yrb(frac),
                            line=dict(color="white", width=2),
                            showlegend=False,
                            hovertemplate=f"<b>{q_labels[qi]}</b><br>Score: {q_vals[qi]:.3f}<extra></extra>",
                        )
                    )
                    fig_q.add_annotation(
                        x=lbl_r * np.cos(ang_mid),
                        y=lbl_r * np.sin(ang_mid),
                        text=f"<b>{q_labels[qi]}</b><br>{q_vals[qi]:.3f}",
                        showarrow=False,
                        font=dict(size=12, color="green"),
                        align="center",
                    )
                fig_q.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            colorscale=[[0, "yellow"], [0.5, "red"], [1, "black"]],
                            cmin=vmin,
                            cmax=vmax,
                            color=[vmin],
                            showscale=True,
                            colorbar=dict(title="Score", x=1.0, thickness=14, len=0.7),
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig_q.update_layout(
                    title="Circumferential Anomaly Profile",
                    height=300,
                    margin=dict(t=40, b=10, l=10, r=60),
                    xaxis=dict(visible=False, scaleanchor="y", range=[-10.5, 10.5]),
                    yaxis=dict(visible=False, range=[-10.5, 10.5]),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_q, width='stretch')

            with info_col:
                st.caption("**Quadrant scores** (latest window)")
                for lbl, val in zip(q_labels, q_vals):
                    st.text(f"  {lbl}: {val:.3f}")
                dominant = q_labels[int(np.argmax(q_vals))]
                st.markdown(
                    f"**Dominant quadrant: {dominant}**"
                    if sum(q_vals) > 1e-6
                    else "**All quadrants balanced.**"
                )

            with st.expander(f"Component breakdown (latest points)", expanded=False, key="copilot_breakdown"):
                display_cols = [
                    c
                    for c in [
                        "channeling_score",
                        "uptake_score",
                        "skin_score",
                        "hbp_score",
                        "topdp_score",
                        "bottomdp_score",
                        "eta_co_score",
                        "heatload_score",
                        "stave_score",
                        "q1_score",
                        "q2_score",
                        "q3_score",
                        "q4_score",
                        "n_components_used",
                    ]
                    if c in scores.columns
                ]
                n = st.number_input(
                    "Points to show", value=12, min_value=1, max_value=1000, step=1
                )
                st.dataframe(scores[display_cols].tail(n), width='stretch')

        if enable:
            render_channeling()
        else:
            st.caption("Turn on **Enable** to start computing channeling propensity.")

    # ── LLM anomaly summary ───────────────────────────────────────────────────
    if st.button("Check Anomalies"):
        with st.spinner("Fetching data from InfluxDB…"):
            df_recent = fetch_recent_online(tr="last 8 hours", window_by="15 minutes")
            df_past = fetch_recent_online(tr="last 1 day", window_by="15 minutes")
            if not df_past.empty:
                cutoff = df_past.index.max() - pd.Timedelta(hours=16)
                df_past = df_past[df_past.index <= cutoff]

        if df_recent.empty:
            st.warning("No recent data fetched from InfluxDB.")
        elif not run_llm:
            st.info(
                "Enable **Run LLM Analysis** in the sidebar to generate the anomaly summary."
            )
        else:
            sensor_desc = load_sensor_desc()
            prompt = build_anomaly_prompt(df_recent, df_past, sensor_desc, notes)
            with st.spinner("Summarising anomalies…"):
                st.markdown(call_llm(ANOMALY_SYSTEM, prompt))

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Unit Cost & Burden Distribution
# ═══════════════════════════════════════════════════════════════════════════════

with burden_tab:
    st.subheader("Unit Cost & Burden Distribution Analysis")
    st.caption(
        "Findings are loaded from `src/assets/data/copilot_analysis/BURDEN_UNITCOST.md`. "
        "Edit that file to update after a new regression run — no code change needed."
    )

    if st.button("Generate Review"):
        findings = load_burden_findings()
        prompt = build_burden_prompt(findings)

        if not run_llm:
            # Show the raw findings without LLM narration
            st.info("LLM narration is off. Showing raw analysis findings.")
            st.markdown(findings)
        else:
            with st.spinner("Generating review…"):
                st.markdown(call_llm(BURDEN_SYSTEM, prompt))

# ── Footer ────────────────────────────────────────────────────────────────────

with st.expander("⚙️ Setup notes", key="copilot_setup_notes"):
    st.markdown(f"""
- **OpenAI Responses API** with `code_interpreter` enabled. Set `OPENAI_API_KEY` and (optionally) `OPENAI_MODEL` (current: `{OPENAI_MODEL}`).
- **Analysis files**: edit `src/assets/data/copilot_analysis/BURDEN_UNITCOST.md` to update burden findings after a new regression run.
- **Sensor descriptions**: edit `src/assets/data/copilot_analysis/ANOMALY_SENSOR_DESC.md` if sensor layout changes.
- **InfluxDB**: set `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_TOKEN`. Bucket is `bf2_evonith_raw`.
    """)

# ── Operator feedback ─────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Operator feedback")

for key, default in [("op_fb_vote", None), ("op_fb_text", ""), ("op_feedback", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

fb_up, fb_down = st.columns(2)
with fb_up:
    if st.button("👍 Useful", key="op_fb_up"):
        st.session_state["op_fb_vote"] = "up"
        st.session_state["op_fb_text"] = ""
with fb_down:
    if st.button("👎 Not useful", key="op_fb_down"):
        st.session_state["op_fb_vote"] = "down"

if st.session_state.get("op_fb_vote") == "down":
    st.text_area(
        "What was not useful?",
        key="op_fb_text",
        placeholder="How could this be improved?",
    )
    if st.button("Submit feedback", key="op_fb_submit"):
        st.session_state["op_feedback"].append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "vote": "down",
                "text": st.session_state.get("op_fb_text", ""),
            }
        )
        st.success("Thanks — feedback captured.")
        st.session_state["op_fb_vote"] = None
        st.session_state["op_fb_text"] = ""
elif st.session_state.get("op_fb_vote") == "up":
    st.info("Thanks for confirming it was useful.")
