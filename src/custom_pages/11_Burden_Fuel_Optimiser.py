"""Burden & Fuel Optimiser — BF2 (epic #260).

The Blend Mix Optimiser's question with fuel in it: which ferrous blend and flux minimise ore + flux + fuel cost per
tonne of hot metal, and how much hot metal the day's limits allow. Coke is not a hand coefficient: for every candidate
blend it is the coke the two-zone (Rist) heat and mass balance needs for the target HMT and Si (the Coke Energy
model, calibrated on recent stable shifts). The blast limit comes from BF-02's burden-resistance rules; every rule,
band and default limit is in the fact register.

    Model check                          burnt coke against the energy and oxygen balances on recent shifts
    Cheapest production level   ore + flux + fuel per tHM against output under the day's limits
    Fixed production            cheapest blend at a target, next to the ore + flux-only answer
    Maximum production                   blend and controls at the production ceiling

Thin UI: the engine is utils.burden_engine, the adapters data.burden_engine, the parts ui.burden_optimiser.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.session import is_admin, is_logged_in, is_supervisor

# ── Auth gate ─────────────────────────────────────────────────────────
if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

# ── Lazy imports (only after auth passes) ─────────────────────────────
import datetime as dt
import json

from data.burden_engine.inputs import build_engine_inputs, describe_slag_basis
import ui.burden_optimiser.loaders as loaders
from services import operator_inputs
from ui.burden_optimiser.sections import (plan_sentence, render_controls, render_dust_editor, render_facts, render_frontier,
                                          render_fuel_ash_inputs, render_limits, render_model_check, render_plan_summary,
                                          render_plan_tabs, render_status, render_thermal, render_yard_editor)
from ui.burden_optimiser.commentary import render_furnace_commentary
from ui.burden_optimiser.omega_panel import render_auto, render_omega_result, render_omega_settings
from utils.burden_engine import BandError, Engine, Infeasible, load_register
from utils.burden_engine.commentary import build_context, load_assumptions, load_notes
from utils.burden_engine.scenarios import fuel_defaults, limit_labels, limits_from, settings_from_regime
from utils.coke_energy.weather import plan_blast_moisture

MODES = {
    "cheapest": "Cheapest production level",
    "fixed": "Fixed production",
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
    "nut coke are fixed inputs. Coke rates are as charged (wet; dry alongside), shares wet % of the ferrous charge, "
    "costs Rs/tHM."
)


def _refresh() -> None:
    st.session_state["bfo_nonce"] = nonce + 1
    loaders.clear()
    st.rerun()


# ── Yard and calibration ──────────────────────────────────────────────
yard, yard_source, yard_msg = loaders.yard(loaders.as_json(cfg), nonce, operator_inputs.token())
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
with st.expander("Advanced settings", expanded=False):
    om_set = render_omega_settings("bfo_om")
prod_cfg = dict(cfg.get("production", {}))
if mode == "fixed":
    # the configured target is the default (user decision 11 Nov 2026); today's production is shown as a hint
    today_t_day = (regime or {}).get("prod_th")
    default_target = float(prod_cfg.get("default_target_t_day", 2270.0))
    today_hint = f" Today's production: about {round(today_t_day * 24, -1):,.0f} t/d (regime median)." if today_t_day else ""
    target = st.columns(4)[0].number_input("Hot metal target, t/d", min_value=1500.0, max_value=3000.0, step=10.0,
                                           value=float(default_target), format="%.0f", key="bfo_target",
                                           help=f"Default: the configured target, {default_target:,.0f} t/d.{today_hint}")
else:
    target = float(prod_cfg.get("default_target_t_day", 2270.0))

# ── Inputs ────────────────────────────────────────────────────────────
with st.expander("Yard, prices and stock", expanded=False):
    materials, fluxes, fuel = render_yard_editor(yard, "bfo")

# Blast moisture for the plan: Open-Meteo daily value for today at Wardha, fetched live (DATA-11); the calibration
# window's own Open-Meteo mean sits in the calibration (cal.case.humidity_g).
_hum_g, _hum_src, _hum_day = plan_blast_moisture()
humidity_default = dict(value=_hum_g, source=_hum_src, day=_hum_day.date(),
                        window=((info.cal or {}).get("case") or {}).get("humidity_g"))
with st.expander("Thermal target and fixed fuels", expanded=False):
    thermal_defaults = {**cfg.get("thermal", {}),
                        **fuel_defaults(regime, cfg.get("thermal", {}), info.cal, live_pci=loaders.live_pci())}
    thermal = render_thermal(thermal_defaults, reg, humidity_default, "bfo_th")

# the slag side is the Blend Mix Optimiser's (held Si / S / C, dust per stream, minor oxides, FeO rule); the dust's kg
# per charge and analysis are editable here, applied to this run, and saved as operator inputs on request
with st.expander("Top-gas dust and fuel ash", expanded=False):
    slag_basis, dust_df = render_dust_editor(yard.get("slag_basis"), "bfo")
    _can_save = is_admin() or is_supervisor()
    _c1, _c2 = st.columns([1, 3])
    if _c1.button("Save as operator inputs", key="bfo_dust_save", disabled=(not _can_save) or dust_df.empty,
                  help="Saves the dust rows as a new version of the operator inputs (shared with the Blend Mix Optimiser, "
                       "the API and the worker). Edits above apply to this run without saving."
                       + ("" if _can_save else " Needs the supervisor or admin role.")):
        from ui.bmo.operator_inputs import save_dust_preferences

        try:
            written = save_dust_preferences(operator_inputs.YAML_PATH, dust_df)
        except Exception as exc:  # noqa: BLE001 - the page keeps the edits for this run
            st.error(f"Dust inputs not saved: {exc}")
        else:
            st.success(f"Dust inputs saved ({written}); every optimiser reads them from the next run.")
            _refresh()
    _c2.caption("The DPR's measured streams stay off: the plant burden sheet's row is held constant "
                "(setting_bmo.yml → data_sources.dust_quantity_source). Held hot-metal chemistry and the minor-oxide rules "
                "are the Blend Mix Optimiser's.")
    render_fuel_ash_inputs((info.cal or {}).get("case"), slag_basis)
    st.caption(describe_slag_basis(slag_basis))

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
                                 stock_horizon_days=float(cfg.get("yard", {}).get("stock_horizon_days", 1.0)),
                                 slag_basis=slag_basis)
except ValueError as exc:
    st.error(f"Cannot build the yard for the optimiser: {exc}")
    st.stop()
names = {m["id"]: m.get("name", m["id"]) for m in inputs["materials"]} | {f["id"]: f.get("name", f["id"])
                                                                           for f in inputs["fluxes"]}
prices = {m["id"]: m["price"] for m in inputs["materials"]} | {f["id"]: f["price"] for f in inputs["fluxes"]}
zero_stock = [names.get(k, k) for k, v in inputs["not_carried"].items() if v == "zero stock"]
if zero_stock:
    st.caption("Selected but out of stock, so not offered: " + ", ".join(zero_stock) + ".")
# Today's charged blend (last 24 h): the reference a fixed-production plan is compared against. Live yard only.
today_blend = loaders.charged_blend(loaders.as_json(yard["materials"]), nonce) if yard_source == "live" else {}
today_shares = dict(today_blend.get("shares") or {})

# ── Run ───────────────────────────────────────────────────────────────
_b, _t = st.columns([1, 5], vertical_alignment="center")
run = _b.button("Optimise", type="primary", key="bfo_run")
_t.caption(loaders.expected_text(mode, om_set["omega_mode"]))
args = (mode, loaders.as_json(inputs), loaders.as_json(info.cal), s0_d, L.__dict__ | {"process": process}, float(target),
        bool(search_o2), int(prod_cfg.get("o2_levels", 9)), float(prod_cfg.get("frontier_lo_t_h", 84.0)),
        float(prod_cfg.get("frontier_step_t_h", 1.0)), loaders.as_json(today_shares if mode == "fixed" else {}),
        bool(om_set["auto"]))
if run:
    st.session_state["bfo_args"] = args
if st.session_state.get("bfo_args") is None:
    st.stop()
if st.session_state["bfo_args"] != args:
    st.caption("Inputs changed since the last run: press Optimise to update the results below.")

args = st.session_state["bfo_args"]
try:
    res = loaders.run_with_progress(*args)
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
    plan_label = f"{which.lower()} on the production frontier, {final['P_tday']:,.0f} t/d"
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
    frontier=res if res["mode"] == "cheapest" else None, slag_basis=inputs.get("slag_basis"),
    materials=materials + [m for m in yard["materials"] if not m.get("carried", True)], fluxes=fluxes, fuel=fuel)
render_furnace_commentary(context, key="bfo")

with st.expander("Facts this result rests on"):
    render_facts(reg, final)
with st.expander("Assumptions and what the engine does not see"):
    st.markdown(load_assumptions())
