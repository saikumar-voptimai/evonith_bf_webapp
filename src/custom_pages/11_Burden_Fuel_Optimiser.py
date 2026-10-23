"""Burden & Fuel Optimiser — BF2 (epic #260).

The Blend Mix Optimiser's question with fuel in it: which ferrous blend and flux minimise ore + flux + fuel cost per
tonne of hot metal, and how much hot metal the day's limits allow. Coke is not a hand coefficient: for every candidate
blend it is the coke the two-zone (Rist) heat and mass balance needs for the target HMT and Si (the Coke Energy
model, calibrated on recent stable shifts). The blast limit comes from BF-02's burden-resistance rules; every rule,
band and default limit is in the fact register.

    Model check                          burnt coke against the energy and oxygen balances on recent shifts
    Step 1 · cheapest production level   ore + flux + fuel per tHM against output under the day's limits
    Step 2 · fixed production            cheapest blend at a target, next to the ore + flux-only answer
    Maximum production                   blend and controls at the production ceiling

Thin UI: the engine is utils.burden_engine, the adapters data.burden_engine, the parts ui.burden_optimiser.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.session import is_logged_in

# ── Auth gate ─────────────────────────────────────────────────────────
if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

# ── Lazy imports (only after auth passes) ─────────────────────────────
import datetime as dt
import json

from data.burden_engine.inputs import build_engine_inputs
import ui.burden_optimiser.loaders as loaders
from ui.burden_optimiser.sections import (plan_sentence, render_controls, render_facts, render_frontier, render_limits,
                                          render_model_check, render_plan_summary, render_plan_tabs, render_status,
                                          render_thermal, render_yard_editor)
from ui.burden_optimiser.commentary import render_furnace_commentary
from ui.burden_optimiser.omega_panel import render_auto, render_omega_result, render_omega_settings
from utils.burden_engine import BandError, Engine, Infeasible, load_register
from utils.burden_engine.commentary import build_context, load_notes
from utils.burden_engine.scenarios import fuel_defaults, limit_labels, limits_from, settings_from_regime
from utils.coke_energy.weather import blast_moisture_g

MODES = {
    "cheapest": "Step 1 · Cheapest production level",
    "fixed": "Step 2 · Fixed production",
    "max": "Maximum production",
}

_cfg_path = Path(__file__).resolve().parents[1] / "config" / "burden_optimiser.yml"
cfg = loaders.page_config(_cfg_path.stat().st_mtime_ns if _cfg_path.exists() else 0)
reg = load_register()
nonce = int(st.session_state.get("bfo_nonce", 0))

st.title("Burden & Fuel Optimiser — BF2")
st.caption(
    "Chooses the ferrous blend and flux that minimise ore + flux + fuel cost per tonne of hot metal, with the coke "
    "rate from the two-zone energy model at your HMT and Si target, and the output the day's limits allow. PCI and "
    "nut coke are fixed inputs. Coke rates are dry, shares wet % of the ferrous charge, costs Rs/tHM."
)


def _refresh() -> None:
    st.session_state["bfo_nonce"] = nonce + 1
    loaders.clear()
    st.rerun()


# ── Yard and calibration ──────────────────────────────────────────────
yard, yard_source, yard_msg = loaders.yard(loaders.as_json(cfg), nonce)
with st.spinner("Calibrating on recent stable shifts (shared with the Coke Energy Model page)…"):
    info, _ce = loaders.calibration(cfg, nonce)
render_status(yard_source, yard_msg, info, reg, _refresh)
regime = info.regime

# ── Model check: the model against the furnace, before any plan ──────
model_shown = render_model_check(getattr(getattr(_ce, "result", None), "shifts", None), cfg.get("model_check", {}), reg)

st.divider()
st.subheader("Optimise")
mode = st.radio("What to optimise", list(MODES), format_func=MODES.get, horizontal=True, key="bfo_mode")
process = mode in ("cheapest", "max")
om_set = render_omega_settings("bfo_om")
prod_cfg = dict(cfg.get("production", {}))
if mode == "fixed":
    today_t_day = (regime or {}).get("prod_th")
    default_target = round(today_t_day * 24, -1) if today_t_day else float(prod_cfg.get("default_target_t_day", 2270.0))
    target = st.columns(4)[0].number_input("Hot metal target, t/d", min_value=1500.0, max_value=3000.0, step=10.0,
                                           value=float(default_target), format="%.0f", key="bfo_target",
                                           help="Default: today's production (median of the regime window)")
else:
    target = float(prod_cfg.get("default_target_t_day", 2270.0))

# ── Inputs ────────────────────────────────────────────────────────────
with st.expander("Yard, prices and stock", expanded=False):
    materials, fluxes, fuel = render_yard_editor(yard, "bfo")

humidity_default = (regime or {}).get("humidity_g") or blast_moisture_g(dt.date.today())   # DATA-11
with st.expander("Thermal target and fixed fuels", expanded=False):
    thermal_defaults = {**cfg.get("thermal", {}), **fuel_defaults(regime, cfg.get("thermal", {}), info.cal)}
    thermal = render_thermal(thermal_defaults, reg, humidity_default, "bfo_th")

charging = dict(cfg.get("charging", {}))
saved = yard.get("charging_saved") or {}
L0 = limits_from(reg, yard.get("slag"), charging)
with st.expander("Day limits", expanded=False):
    if saved and abs(saved.get("charges_per_h", charging.get("charges_per_h", 0)) - charging.get("charges_per_h", 0)) > 1e-6:
        st.caption(f"The Blend Mix Optimiser's saved charging is {saved['charges_per_h']:.2f} charges/h × "
                   f"{saved.get('charge_t', 0):.2f} t; this page starts from the plant figure "
                   f"{charging.get('charges_per_h')} × {charging.get('charge_t')} t.")
    L = render_limits(L0, regime, "bfo_lim", show_process=process)
    if process:
        search_o2 = st.checkbox("Search oxygen enrichment up to its limit", value=True, key="bfo_o2",
                                help="Oxygen has no price here: it is used only where it buys output.")
    else:
        search_o2 = False

s0 = settings_from_regime(regime, cfg.get("thermal", {}), info.cal)
s0_d = dict(s0.__dict__, **thermal)
s0_d.update(omega_mode=om_set["omega_mode"], certainty=om_set["certainty"], risk_k=om_set["risk_k"])
if mode == "fixed":
    with st.expander("Controls held for the plan (start: today's medians)", expanded=False):
        hy_eng = Engine(cal=info.cal)          # only for the enrichment-tag conversion
        s0_d.update(render_controls(s0_d, hy_eng.hy.enrich_tag, hy_eng.hy.o2_per_air_from_tag, regime, "bfo_ctl"))

try:
    inputs = build_engine_inputs(materials, fluxes, fuel,
                                 stock_horizon_days=float(cfg.get("yard", {}).get("stock_horizon_days", 1.0)))
except ValueError as exc:
    st.error(f"Cannot build the yard for the optimiser: {exc}")
    st.stop()
names = {m["id"]: m.get("name", m["id"]) for m in inputs["materials"]} | {f["id"]: f.get("name", f["id"])
                                                                           for f in inputs["fluxes"]}
prices = {m["id"]: m["price"] for m in inputs["materials"]} | {f["id"]: f["price"] for f in inputs["fluxes"]}
zero_stock = [names.get(k, k) for k, v in inputs["not_carried"].items() if v == "zero stock"]
if zero_stock:
    st.caption("Selected but out of stock, so not offered: " + ", ".join(zero_stock) + ".")
# Today's charged blend (last 24 h): the reference Step 2 is compared against. Live yard only.
today_blend = loaders.charged_blend(loaders.as_json(yard["materials"]), nonce) if yard_source == "live" else {}
today_shares = dict(today_blend.get("shares") or {})

# ── Run ───────────────────────────────────────────────────────────────
run = st.button("Optimise", type="primary", key="bfo_run")
args = (mode, loaders.as_json(inputs), loaders.as_json(info.cal), s0_d, L.__dict__ | {"process": process}, float(target),
        bool(search_o2), int(prod_cfg.get("o2_levels", 9)), float(prod_cfg.get("frontier_lo_t_h", 84.0)),
        float(prod_cfg.get("frontier_step_t_h", 1.0)), loaders.as_json(today_shares if mode == "fixed" else {}),
        bool(om_set["auto"]))
if run:
    st.session_state["bfo_args"] = args
if st.session_state.get("bfo_args") is None:
    st.info("Set the inputs and press Optimise. Step 1 and maximum production search output and oxygen and take "
            "about a minute; Step 2 takes a second.")
    st.stop()
if st.session_state["bfo_args"] != args:
    st.caption("Inputs changed since the last run: press Optimise to update the results below.")

args = st.session_state["bfo_args"]
try:
    _dur = {"fixed": "about 30 s", "max": "about 2 min", "cheapest": "about 8 min"}.get(args[0], "")
    with st.spinner("Optimising (two-zone balance inside a sequential LP)…" if args[3].get("omega_mode") != "blend" else
                    "Optimising: two-zone balance inside a sequential LP, with the STRATA-Flow network solved at every "
                    "iteration and for every reported plan (about 1 s per solve; " + (f"{_dur} for this mode; " if _dur else "")
                    + "the first run of a new calibration also computes the network Jacobian, about 20 s)…"):
        res = loaders.run_mode(*args)
except BandError as exc:
    st.error(f"{exc}. The engine does not report outside the plausibility bands.")
    st.stop()
except Infeasible as exc:
    st.error(f"The LP has no solution for these share bounds and stock caps ({exc}). Widen a share bound or add stock.")
    st.stop()

eng = loaders.engine(json.loads(args[1]), json.loads(args[2]))
st.divider()
flux_ids = [v.id for v in eng.flux]
today_t_day = (regime or {}).get("prod_th") * 24 if (regime or {}).get("prod_th") else None
today, r0, window = None, None, ""
if res["mode"] == "fixed":
    r, r0, today = res["with_fuel"], res["ore_only"], res.get("today")
    window = (f"from {today_blend['window'][0]} to {today_blend['window'][1]} IST" if today is not None
              else "over the last 24 h")
    render_plan_summary(r, f"Cheapest blend at {r['P_tday']:,.0f} t/d", flux_ids=flux_ids, today=today,
                        lead=plan_sentence(r, names, flux_ids, today), other=r0, other_label="ore + flux only")
    if today_blend.get("unmapped"):
        st.caption("Today's blend leaves out materials without lab chemistry in the window: "
                   + ", ".join(f"{k} {v:.1f} %" for k, v in today_blend["unmapped"].items()) + ".")
    elif yard_source == "live" and today is None:
        st.caption("Today's charged blend is not available"
                   + (f" ({today_blend['error']})" if today_blend.get("error") else "") + ", so there is no comparison.")
    render_omega_result(eng, r, today)
    render_plan_tabs(eng, r, "fixed", other=r0, value_of_limits=res.get("value_of_limits"), today=today,
                     names=names, prices=prices, today_window=window)
    final, plan_label = r, f"fixed production {r['P_tday']:,.0f} t/d"

elif res["mode"] == "cheapest":
    if res["best"] is None:
        st.error("No production level meets every limit at these settings.")
        st.stop()
    st.subheader("Cost against production under the day's limits")
    render_frontier(res)
    which = st.radio("Plan shown below", ["Cheapest output", "Highest output"], horizontal=True, key="bfo_which")
    final = res["best"] if which == "Cheapest output" or res["top"] is None else res["top"]
    render_plan_summary(final, f"{which}: {final['P_tday']:,.0f} t/d", flux_ids=flux_ids, today_t_day=today_t_day)
    render_omega_result(eng, final)
    render_plan_tabs(eng, final, "cheapest", names=names, prices=prices)
    plan_label = f"{which.lower()} on the Step 1 frontier, {final['P_tday']:,.0f} t/d"
else:
    final = res["result"]
    if final is None:
        st.error("No production level meets every limit at these settings.")
        st.stop()
    render_plan_summary(final, f"Maximum production: {final['P_tday']:,.0f} t/d", flux_ids=flux_ids,
                        today_t_day=today_t_day,
                        lead=("plain", "Limited by: " + (limit_labels(final.get("binding", []), names) or "–")
                              + ". Controls at their limits; oxygen searched within its flow and enrichment limits, "
                              "least oxygen among equal outputs."))
    render_omega_result(eng, final)
    render_plan_tabs(eng, final, "max", names=names, prices=prices)
    plan_label = f"maximum production {final['P_tday']:,.0f} t/d"

# ── Auto mode ─────────────────────────────────────────────────────────
_cal_txt = dict(source=info.source, window=(f"{info.window[0]} to {info.window[1]}" if info.source == "live"
                                            else " to ".join(info.cal.get("period", ["", ""]))), omega=info.cal.get("omega"))
if len(args) > 11 and args[11]:
    st.divider()
    render_auto(res.get("auto"), names, mode_label=f"{MODES[res['mode']]}, {plan_label}", calibration=_cal_txt)

# ── Furnace commentary ────────────────────────────────────────────────
st.divider()
_cal = info.cal
context = build_context(
    notes=load_notes(), reg=reg, mode=MODES[res["mode"]], plan=final, plan_label=plan_label, names=names,
    flux_ids=flux_ids, settings=final["settings"], limits=final["limits"], yard_source=yard_source,
    calibration=dict(source=info.source, window=(f"{info.window[0]} to {info.window[1]}" if info.source == "live"
                                                 else " to ".join(_cal.get("period", ["", ""]))),
                     omega=_cal.get("omega"), Q_LZ_GJh=_cal.get("Q_LZ_GJh"), Q_UZ_GJh=_cal.get("Q_UZ_GJh"),
                     message=info.message if info.source != "live" else ""),
    regime=regime, target_t_day=float(args[5]) if res["mode"] == "fixed" else None, model_check=model_shown,
    today=today, today_window=window, ore_only=r0, value_of_limits=res.get("value_of_limits"),
    frontier=res if res["mode"] == "cheapest" else None,
    materials=materials + [m for m in yard["materials"] if not m.get("carried", True)], fluxes=fluxes, fuel=fuel)
render_furnace_commentary(context, key="bfo")

with st.expander("Facts this result rests on"):
    render_facts(reg, final)
with st.expander("Assumptions and what the engine does not see"):
    st.markdown(
        "- **Shaft efficiency** is either Fixed-ω (held at the calibration, the Rist baseline; ASM-01) or STRATA-ω "
        "(default; ASM-18, GAS-13..25, STRATA 1.4): the STRATA-Flow gas-flow network's ring counter-current description "
        "of the thermal reserve zone (size of each ferrous, coke layers, fuels, oxygen, nut coke), solved for every plan "
        "- one network solve per optimiser iteration and three per reported plan, about 1 s each, no surrogate. "
        "Plant-data scales: pellet 1.39 ± 0.46, lump ore 0.97 ± 0.46, other operation 0.83 ± 0.29; blast-moisture and "
        "nut-coke terms; no production channel (production acts through the balance; the channel was not replicated "
        "on 2023-24 data, GAS-16). Tested out of sample on 175 days of 2026 (burnt-coke skill +0.14 day, +0.22 7-day "
        "against Fixed-ω; GAS-20) and with frozen parameters on 413 days of 2023-24 (RMSE 7.40 against 7.95 kg/tHM; "
        "GAS-25). Each plan carries σ(coke), a certainty class (High / Medium / "
        "Low / Investigate), its critical assumptions and the network solves behind it; the calibration's own error "
        "(about ±2 kg/tHM) and the day-to-day spread (about ±7 kg/tHM) are reported apart.\n"
        "- **Reducibility and RDI** are the Evonith laboratory tests as given (GAS-21, DATA-13: pellet 82, Lloyds CLO 84, "
        "Geomin 79, NMDC ROM 61, Donimalai 78 % RI; Gomti not tested). The ore RDIs act in the network as shaft fines. "
        "The link from RI to reduction rate is calibrated on the plant data: the measured RI differences show no effect "
        "on the plant's shaft efficiency, particle size does (RI exponent 0, size exponent 1.0; both are critical "
        "assumptions of every plan).\n"
        "- **Oversize**: the Lloyds fines oversize (ore_6) reduces at the Lloyds CLO size (no size credit, GAS-24); the "
        "Donimalai oversize return (ore_11) is Donimalai.\n"
        "- **Investigate tier**: pellet above 25 % of ferrous or σ(coke) above 8 kg/tHM is beyond common practice; such "
        "plans are excluded by default and are trial proposals only.\n"
        "- **Blast moisture** is the weather record's relative humidity, smoothed over 31 days and mapped to 7-21 g/Nm³ "
        "(DATA-11; no step changes), the monthly table outside the record. **PCI and nut coke** default to the regime "
        "window's medians (ASM-08); both act through the balance, nut coke also through STRATA-ω (GAS-23).\n"
        "- **Burden-resistance rules** (FLOW-02..07) were fitted on pellet up to 16.6 % and Gomti up to 11.6 % of "
        "ferrous; beyond that the page warns that it is extrapolating.\n"
        "- **Process limits** default to the Jun-Oct 2026 operating envelope, not equipment ratings. Oxygen and stove "
        "gas have no price.\n"
        "- **Not limited:** phosphorus, alkalis, zinc, sulphur; sinter-plant cost of a basicity change.\n"
        "- Steady day average: no transition between blends; dust and heat losses (GJ/h) stay at the calibration."
    )
