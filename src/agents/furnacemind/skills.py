"""SkillEngine — pre-computes analysis and builds operator prompts.

All calibration numbers live in ``storage/furnacemind/skill_params.yml``.
To recalibrate after a new regression run: edit that YAML file only.
No Python changes required.

Usage::

    engine = SkillEngine()
    prompt = engine.optimise_prompt()              # stores fig in session_state
    prompt = engine.shift_to_best_prompt("2026-04-03", "B")
    prompt = engine.heatload_prompt()              # LLM fetches live data
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from agents.furnacemind.prompts import HEATLOAD_PLOT_CODE, HEATLOAD_REPORT_TEMPLATE
from agents.furnace_tools import fetch_ml_data, load_static_shift_data

_PARAMS_PATH = (
    Path(__file__).resolve().parents[2] / "storage" / "furnacemind" / "skill_params.yml"
)


def _load_params() -> dict:
    with _PARAMS_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Load once at module import — YAML is ~2 KB, negligible cost.
_PARAMS: dict = _load_params()


# ── Low-level helpers (also useful for testing in isolation) ─────────────────


def col_mean(df: pd.DataFrame, col: str) -> float | None:
    """Mean of df[col] (NaN-dropped), or None if column absent or all-NaN."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return round(float(s.mean()), 3) if len(s) else None


def is_adverse(cfg: dict[str, Any], val: float) -> bool:
    """Return True if *val* falls in the adverse region defined by cfg['adverse']."""
    a = cfg.get("adverse")
    if not a:
        return False
    t = a["type"]
    if t == "lt":
        return val < a["threshold"]
    if t == "range":
        return a["lo"] <= val <= a["hi"]
    return False


# ── SkillEngine ──────────────────────────────────────────────────────────────


class SkillEngine:
    """Pre-computes skill analysis; returns numbered prompts ready for the LLM.

    The LLM receives only *numbers*, never code.  Charts are built here and
    stored in ``st.session_state["fm_fig"]`` before the LLM is called.
    """

    def __init__(self) -> None:
        self._t1: dict[str, dict] = _PARAMS["tier1"]
        self._t2: dict[str, dict] = _PARAMS["tier2"]
        self._t3_cols: list[str] = _PARAMS["tier3_guardrail_cols"]
        self._uc_col: str = _PARAMS["unit_cost_col"]
        self._bmark_pct: float = _PARAMS["benchmark_percentile"]

    # ── Public skill entry points ────────────────────────────────────────────

    def optimise_prompt(self) -> str:
        """Run 30-day ML analysis; return a prompt containing only pre-computed numbers."""
        data, _ = self._compute_optimise_data()
        if "error" in data:
            return self._optimise_fallback(data["error"])
        self._build_gap_figure(
            data["tier1"], title="Unit Cost Optimisation — Tier 1 Gap"
        )
        return self._format_optimise_prompt(data)

    def shift_to_best_prompt(self, shift_date: str, label: str) -> str:
        """Compare a specific shift against best-shift targets."""
        load_static_shift_data(shift_date=shift_date, shift_label=label)
        shift_df: pd.DataFrame | None = st.session_state.get("fm_df")

        if shift_df is None or shift_df.empty:
            return (
                f"SKILL: Shift to Best — {shift_date} Shift {label}\n"
                f"Static data load failed. Call load_static_shift_data(shift_date='{shift_date}', "
                f"shift_label='{label}') then compare Tier 1/2 parameters against "
                "SKILLS_BESTSHIFT.md bands and write a concise report."
            )

        # Use last 4 rows of the 8h shift window as the lagged source for angle params.
        df_lagged = shift_df.iloc[-4:] if len(shift_df) >= 4 else shift_df
        tier1_rows = self._compute_tier1_rows(shift_df, df_lagged=df_lagged)
        self._build_gap_figure(
            tier1_rows, title=f"Shift to Best — {shift_date} Shift {label}"
        )

        uc = col_mean(shift_df, self._uc_col)
        uc_benchmark = self._fetch_uc_benchmark()
        # Restore shift df — benchmark fetch overwrites session_state["fm_df"]
        st.session_state["fm_df"] = shift_df

        t1_lines = self._format_tier1_lines(tier1_rows)
        t2_lines = self._tier2_lines_from_df(shift_df)

        return (
            f"SKILL: Shift to Best — {shift_date} Shift {label}\n"
            "Pre-computed results below. The comparison chart is already in the Artifacts panel. "
            "Write the operator report ONLY — no code, no tool calls needed.\n\n"
            f"UNIT COST (shift avg): {uc} Lakhs/tHM | Best-shift benchmark (30d p20): {uc_benchmark} Lakhs/tHM\n\n"
            "TIER 1 (ranked by impact score):\n" + "\n".join(t1_lines) + "\n\n"
            "TIER 2:\n"
            + ("\n".join(t2_lines) or "  (no Tier 2 data available)")
            + "\n\n"
            "Write this report:\n"
            f"**Shift**: {shift_date} Shift {label}\n"
            f"**Unit cost**: [value] Lakhs/tHM vs benchmark {uc_benchmark} Lakhs/tHM\n"
            "**Guardrails**: [state TOPBAR, DP, CHEM_PCT_SI status if available]\n"
            "**Top 3 Tier 1 actions** (adverse=YES first, then highest impact):\n"
            "  1. [param]: [current] → [target] | ACTION: [specific setpoint] | REASON: [from SKILLS_BESTSHIFT] | MAGNITUDE: [effect]\n"
            "  2. ...\n"
            "  3. ...\n"
            "**Tier 2 flags**: [list ADVERSE/OUTSIDE_BAND items]\n"
            "*Lag note: burden angle effects appear ~4h after change.*\n"
        )

    def heatload_prompt(self) -> str:
        """Return the heatload skill prompt (LLM fetches live data, then reports)."""
        return (
            "SKILL: Check Heatloads\n"
            "Do not output any code or planning. Call the two tools below, then write the report.\n\n"
            "STEP 1 — Call: fetch_online_data(lookback_hours=8, "
            'measurement_groups=["heatload_delta_t","temperature_profile","process_params"])\n\n'
            "STEP 2 — Call: execute_python_plot with this exact code (copy verbatim):\n"
            f"{HEATLOAD_PLOT_CODE}\n\n"
            "STEP 3 — After both tool calls, write this report (fill in actual numbers from the data):\n"
            f"{HEATLOAD_REPORT_TEMPLATE}\n"
        )

    def shift_report_prompt(
        self, shift_date: str, label: str, start_utc: str, end_utc: str
    ) -> str:
        """Return the live shift handover report prompt.

        The LLM will fetch online + offline data for the given shift window,
        then render the report following SKILLS_SHIFTREPORT.md.
        """
        return (
            f"SKILL: Shift Handover Report — {shift_date} Shift {label}\n"
            "Do NOT call execute_python_plot — this skill produces text tables only, no charts. "
            "Do NOT output any code, planning text, or commentary. "
            "Execute the steps below in order, then write the report.\n\n"
            "--- COLUMN NAME REFERENCE ---\n"
            "Online columns use canonical InfluxDB field names from setting_ds_dv.yml.\n"
            "  Hot blast volume     -> 'hot_blast_vol_nm3h'\n"
            "  Hot blast temp       -> 'hot_blast_temp'\n"
            "  Hot blast pressure   -> 'hot_blast_press'\n"
            "  Permeability         -> 'body_perm'\n"
            "  ETA CO               -> 'body_etaco'\n"
            "  RAFT                 -> 'body_raft'\n"
            "  Uptake Temp Q1-Q4    -> 'top_temp_1/2/3/4'\n"
            "  Runner Temp (HM Temp)-> 'runner_temp_pci_taphole'\n"
            "  Skip car trips (sum) -> 'skip_car_trips_hour'\n"
            "  Bosh DeltaT Q1-Q4    -> 'delta_t_avg_row6_10_q1/q2/q3/q4'\n"
            "  Hearth 4.3m A        -> 'temp_4373_a'\n"
            "  Hearth 5.4m C        -> 'temp_5411_c'\n"
            "  Hearth 5.7m C        -> 'temp_5757_c'\n"
            "  Hearth 6.1m B        -> 'temp_6103_b'\n"
            "  Lower Stack Q1-Q4    -> 'temp_18660_a/b/c/d'\n"
            "  Belly Q1-Q4          -> 'temp_15162_a/b/c/d'\n"
            "  Bosh Q1-Q4           -> 'temp_12975_a/b/c/d'\n"
            "Offline columns follow the format: 'Offline[<Report>] - <field>'\n"
            "  HM Si/S/Fe           → 'Offline[HM_Slag] - chem_pct_si/s/fe'\n"
            "  Slag basicity inputs → 'Offline[HM_Slag] - slag_pct_cao', 'Offline[HM_Slag] - slag_pct_sio2'\n"
            "  Coke                 → SUM 'Offline[Charge] - coke_1_mt' through 'coke_2_mt'\n"
            "  Nut coke             → SUM 'Offline[Charge] - nut_coke_1_mt' through 'nut_coke_2_mt'\n"
            "  Sinter               → SUM 'Offline[Charge] - sinter_1_mt' through 'sinter_4_mt'\n"
            "  Ore                  → SUM 'Offline[Charge] - ore_1_mt' through 'ore_12_mt'\n"
            "  Pellet               → SUM 'Offline[Charge] - pellet_1_mt' through 'pellet_2_mt'\n"
            "  Flux                 → SUM 'Offline[Charge] - flux_1_mt' through 'flux_3_mt'\n"
            "  PCI                  → 'Offline[Charge] - pci_mt'  (SUM)\n"
            "NOTE: production_per_hour, coke_rate, nut_coke_rate, and coal_rate_actual_value are online fields.\n"
            "  Derive fuel_rate = coke_rate + nut_coke_rate + coal_rate_actual_value\n"
            "  If theoretical_production unavailable, mark kg/tHM fields as '-'.\n"
            "No. of Taps = count of non-null rows in HM_SLAG data for this shift window.\n\n"
            "--- STEPS ---\n"
            f"STEP 1 — Call fetch_online_data with:\n"
            f"  start_time_utc='{start_utc}'\n"
            f"  end_time_utc='{end_utc}'\n"
            "  window='15 minutes'\n"
            "  measurement_groups=['process_params','temperature_profile','delta_t','miscellaneous']\n\n"
            f"STEP 2 — Call fetch_offline_data with:\n"
            f"  report_type='HM_SLAG'\n"
            f"  start_time_utc='{start_utc}'\n"
            f"  end_time_utc='{end_utc}'\n\n"
            f"STEP 3 — Call fetch_offline_data with:\n"
            f"  report_type='CHARGE'\n"
            f"  start_time_utc='{start_utc}'\n"
            f"  end_time_utc='{end_utc}'\n\n"
            "STEP 4 — Using all fetched data, write the SHIFT HANDOVER REPORT "
            "following the template in SKILLS_SHIFTREPORT.md exactly.\n\n"
            f"Shift window: {shift_date} Shift {label}  |  UTC: {start_utc} → {end_utc}\n"
        )

    # ── Data computation ─────────────────────────────────────────────────────

    def _compute_optimise_data(self) -> tuple[dict, pd.DataFrame | None]:
        start_date = (date.today() - timedelta(days=30)).isoformat()
        fetch_ml_data(start_time=start_date)
        df: pd.DataFrame | None = st.session_state.get("fm_df")
        if df is None or df.empty:
            return {"error": "ML data load failed"}, None

        df_sorted = df.sort_index().dropna(how="all")
        df_curr = df_sorted.tail(8)
        # 4h-lagged window for burden angle params: rows 8–12 hours ago
        df_lagged = df_sorted.iloc[-12:-4] if len(df_sorted) >= 12 else df_curr

        tier1_rows = self._compute_tier1_rows(df_curr, df_lagged=df_lagged)
        if not tier1_rows:
            return {"error": "No Tier 1 columns found in ML dataset"}, df

        tier2 = self._compute_tier2_dict(df_curr)
        tier3 = {c: col_mean(df_curr, c) for c in self._t3_cols}

        uc_series = (
            df_sorted[self._uc_col].dropna()
            if self._uc_col in df_sorted.columns
            else pd.Series(dtype=float)
        )
        uc_benchmark = (
            round(float(uc_series.quantile(self._bmark_pct)), 3)
            if len(uc_series)
            else None
        )

        return {
            "tier1": tier1_rows,
            "tier2": tier2,
            "tier3": tier3,
            "unit_cost": col_mean(df_curr, self._uc_col),
            "uc_benchmark": uc_benchmark,
        }, df

    def _compute_tier1_rows(
        self,
        df_curr: pd.DataFrame,
        df_lagged: pd.DataFrame | None = None,
    ) -> list[dict]:
        """Build Tier 1 rows sorted descending by impact score."""
        if df_lagged is None:
            df_lagged = df_curr
        rows = []
        for col, cfg in self._t1.items():
            src = df_lagged if cfg["lag_hours"] > 0 else df_curr
            v = col_mean(src, col)
            if v is None:
                continue
            mid = cfg["best_mid"]
            coeff = cfg["abs_coeff"]
            rows.append(
                {
                    "param": col,
                    "current": v,
                    "target": mid,
                    "score": round(abs(mid - v) * coeff, 3),
                    "adverse": is_adverse(cfg, v),
                    "unit": cfg.get("unit", ""),
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    def _compute_tier2_dict(self, df: pd.DataFrame) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for col, cfg in self._t2.items():
            v = col_mean(df, col)
            if v is None:
                continue
            lo, hi, adv = (
                cfg.get("band_lo"),
                cfg.get("band_hi"),
                cfg.get("adverse_threshold"),
            )
            flag = "OK"
            if adv is not None and v > adv:
                flag = "ADVERSE"
            elif lo is not None and hi is not None and not (lo <= v <= hi):
                flag = "OUTSIDE_BAND"
            result[col] = {
                "value": v,
                "band": f"{lo}–{hi}" if lo is not None else "—",
                "flag": flag,
            }
        return result

    def _fetch_uc_benchmark(self) -> float | None:
        """Fetch 30-day ML history solely to compute the unit-cost benchmark percentile."""
        fetch_ml_data(start_time=(date.today() - timedelta(days=30)).isoformat())
        df: pd.DataFrame | None = st.session_state.get("fm_df")
        if df is None or self._uc_col not in df.columns:
            return None
        s = df[self._uc_col].dropna()
        return round(float(s.quantile(self._bmark_pct)), 3) if len(s) else None

    # ── Figure builders ──────────────────────────────────────────────────────

    def _build_gap_figure(self, tier1_rows: list[dict], *, title: str) -> None:
        """Build a current-vs-target bar chart and store it in session_state."""
        params = [r["param"] for r in tier1_rows]
        curr_vals = [r["current"] for r in tier1_rows]
        tgt_vals = [r["target"] for r in tier1_rows]
        colours = ["#e74c3c" if r["adverse"] else "#3498db" for r in tier1_rows]

        fig = go.Figure()
        fig.add_bar(
            name="Current (8h avg)", x=params, y=curr_vals, marker_color=colours
        )
        fig.add_bar(
            name="Best-shift target",
            x=params,
            y=tgt_vals,
            marker_color="rgba(46,204,113,0.55)",
        )
        fig.update_layout(
            title=title + "  (red = adverse)",
            barmode="group",
            xaxis_tickangle=-20,
            legend=dict(orientation="h", y=1.12),
            height=400,
            margin=dict(t=80),
        )
        st.session_state["fm_fig"] = fig

    # ── Prompt formatters ────────────────────────────────────────────────────

    @staticmethod
    def _format_tier1_lines(rows: list[dict]) -> list[str]:
        return [
            f"  {r['param']}: current={r['current']}, target={r['target']}, "
            f"impact={r['score']}, adverse={'YES' if r['adverse'] else 'no'}"
            for r in rows
        ]

    def _tier2_lines_from_df(self, df: pd.DataFrame) -> list[str]:
        return [
            f"  {col}: {v['value']} (band {v['band']}) [{v['flag']}]"
            for col, v in self._compute_tier2_dict(df).items()
        ]

    def _format_optimise_prompt(self, data: dict) -> str:
        t1 = "\n".join(self._format_tier1_lines(data["tier1"]))
        t2 = (
            "\n".join(
                f"  {col}: {v['value']} (band {v['band']}) [{v['flag']}]"
                for col, v in data["tier2"].items()
            )
            or "  (no data)"
        )
        t3 = (
            "\n".join(
                f"  {col}: {val}"
                for col, val in data["tier3"].items()
                if val is not None
            )
            or "  (no data)"
        )
        uc = data["unit_cost"]
        ucb = data.get("uc_benchmark", "N/A")

        return (
            "SKILL: Optimise Unit Cost\n"
            "Pre-computed results below. The gap chart is already in the Artifacts panel. "
            "Write the operator report ONLY — no code, no tool calls.\n\n"
            "DATA: Last 30 days ML static data. Current state = last 8h average. "
            "Burden angles use 4h-lagged window.\n\n"
            f"UNIT COST (last 8h avg): {uc} Lakhs/tHM | Benchmark (best 20% of 30-day history): {ucb} Lakhs/tHM\n\n"
            f"TIER 1 PARAMETERS (ranked by impact score = gap × |coefficient|):\n{t1}\n\n"
            f"TIER 2 QUALITY (last 8h vs best-shift band):\n{t2}\n\n"
            f"TIER 3 GUARDRAILS (last 8h):\n{t3}\n\n"
            "Write this report (use the exact numbers above):\n"
            f"**Unit cost**: {uc} Lakhs/tHM (benchmark {ucb} Lakhs/tHM)\n"
            "**Guardrails**: [CLEAR or WARNING — name the param]\n"
            "**Top 3 actions** (adverse=YES first, then by impact_score):\n"
            "  1. [param]: [current] → [target] | ACTION: [specific setpoint] | REASON: [from SKILLS_BESTSHIFT.md] | MAGNITUDE: [expected effect]\n"
            "  2. ...\n"
            "  3. ...\n"
            "**Tier 2 flags**: [list ADVERSE/OUTSIDE_BAND items with values]\n"
            "*Burden angle changes show cost impact after ~4h. Do not push enrichment if guardrails are adverse.*\n"
        )

    @staticmethod
    def _optimise_fallback(error: str) -> str:
        start = (date.today() - timedelta(days=30)).isoformat()
        return (
            f"SKILL: Optimise Unit Cost — pre-computation failed: {error}. "
            f"Call fetch_ml_data(start_time='{start}') "
            "then analyse Tier 1 parameters vs SKILLS_BESTSHIFT.md bands."
        )
