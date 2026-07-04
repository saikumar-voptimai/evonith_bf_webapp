"""Material Balance Visualiser — BF2.

Single-date element balance showing how much of each element (Fe, C, Si,
Ca, Mg, Al, Mn, S, P, O, N, H) enters via raw materials + blast + steam
and leaves via hot metal + slag + top gas + dust catcher.

Data source: static furnace ML dataset.

Layout:
    Top row : date picker | refresh | dust catcher input
    Summary : prominent overall-closure status card + totals
    Left 70%: 3 tabs — Sankey | Per-element bars | Closure table
    Right 30%: lightweight furnace cross-section diagram
    Bottom  : Lag settings expander, DPR mapping expander, Assumptions expander
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from config.frontend_settings import is_backend_api_enabled
from services.api_errors import FrontendApiError
from services.material_balance_api import run_material_balance
from utils.session import is_logged_in

# ── Auth gate ─────────────────────────────────────────────────────────
if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()


def _api_token() -> str | None:
    token = str(st.session_state.get("auth_access_token") or "").strip()
    return token or None


def _render_api_mode() -> None:
    st.title("Material Balance - BF2")
    st.caption("Material Balance mode: Backend API")
    yesterday = date.today() - timedelta(days=1)
    selected_day = st.date_input("Date (IST)", value=yesterday, max_value=yesterday)
    dust_catcher_t = st.number_input(
        "Dust Catcher (t/day)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=1.0,
    )
    export = st.checkbox("Create CSV artifact", value=False)
    if st.button("Run Material Balance", type="primary"):
        try:
            result = run_material_balance(
                {
                    "source": "static_dataset",
                    "date": selected_day.isoformat(),
                    "options": {"dust_catcher_t": float(dust_catcher_t)},
                    "export": export,
                },
                token=_api_token(),
            )
        except FrontendApiError as exc:
            request_id = f" Request ID: {exc.request_id}." if exc.request_id else ""
            st.error(f"Backend Material Balance failed: {exc.message}.{request_id}")
            return
        st.metric("Overall closure", result.get("kpis", {}).get("overall_closure_pct"))
        table = (result.get("tables") or {}).get("closure") or {}
        st.dataframe(table.get("rows") or [], width="stretch")
        for artifact in result.get("artifacts") or []:
            st.link_button("Download CSV", artifact.get("download_url") or "")


if is_backend_api_enabled("material_balance"):
    _render_api_mode()
    st.stop()

# ── Lazy imports (only after auth passes) ─────────────────────────────
from utils.material_balance.compute import run_full_balance
from utils.material_balance.data_sources import (
    clear_day_caches,
    get_csv_date_range,
)
from utils.material_balance.dpr_mapping import (
    ASH_ANALYSIS_FIELDS,
    ASH_MATERIAL_CONFIG_KEYS,
    CANONICAL_MASS_FIELDS,
    discover_dpr_fields,
    load_dpr_mapping,
    load_full_config,
    mapping_is_complete,
    save_ash_analyses,
    save_dpr_mapping,
)
from plotters.material_balance_plots import (
    build_furnace_diagram,
    build_per_element_bars,
    style_closure_table,
)

# ── Page chrome ───────────────────────────────────────────────────────
st.title("⚖️ Material Balance — BF2")

# ── static dataset date availability hint ─────────────────────────────
csv_min, csv_max = get_csv_date_range()
if csv_min and csv_max:
    st.caption(
        f"Data source: static ML dataset  ·  "
        f"Available range: **{csv_min}** → **{csv_max}**"
    )
else:
    st.warning(
        "Static ML dataset is not available."
    )

# ── Top control row ───────────────────────────────────────────────────
yesterday = date.today() - timedelta(days=1)
ctrl1, ctrl2, ctrl3 = st.columns([0.30, 0.18, 0.52])

with ctrl1:
    selected_day = st.date_input(
        "Date (IST)",
        value=yesterday,
        max_value=yesterday,
        min_value=csv_min or date(2024, 1, 1),
        help="Pick a completed IST calendar day.",
    )

with ctrl2:
    st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    if st.button("Refresh data", width="stretch"):
        clear_day_caches(selected_day)
        st.rerun()

with ctrl3:
    dust_catcher_t = st.number_input(
        "Dust Catcher (t/day)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=1.0,
        help=(
            "Manually enter the daily dry-dust tonnage captured by the "
            "cyclonic dust catcher. Elemental composition applied from "
            "`material_balance.yml → dust_catcher_composition_pct`."
        ),
    )

# ── Lag settings ──────────────────────────────────────────────────────
with st.expander("⏱ Lag settings (advanced)", expanded=False, key="mb_lag_settings"):
    st.markdown(
        """
Set **lag hours** to account for the carry-forward behaviour of furnace inputs:

| Input | Typical lag | Rationale |
|---|---|---|
| RM composition (burden) | 0 – 96 h | Burden takes 4–8 h to descend; composition changes propagate slowly |
| Blast (wind, O₂, steam) | 0 – 4 h | Blast changes affect top-gas / HM within the same shift or next |

A positive lag shifts the **input window** backward by that many hours while the output
window (HM chemistry, slag chemistry) remains on the selected date.
"""
    )
    lag_col1, lag_col2 = st.columns(2)
    with lag_col1:
        rm_lag_hours = st.number_input(
            "RM composition lag (hours)",
            min_value=0,
            max_value=240,
            value=0,
            step=24,
            help="0 = same day as selected; 96 = 4 days prior.",
        )
    with lag_col2:
        blast_lag_hours = st.number_input(
            "Blast / process-param lag (hours)",
            min_value=0,
            max_value=48,
            value=0,
            step=1,
            help="0 = same day; 4 = blast values from 4 hours earlier.",
        )

# ── Run the balance ───────────────────────────────────────────────────
with st.spinner("Computing element balance …"):
    result = run_full_balance(
        selected_day,
        rm_lag_hours=int(rm_lag_hours),
        blast_lag_hours=int(blast_lag_hours),
        dust_catcher_t=float(dust_catcher_t),
    )

# ── KPI tile ──────────────────────────────────────────────────────────
ct = result.closure_table
total_in = ct["In_t"].sum()
total_out = ct["Out_t"].sum()
overall_pct = (total_out / total_in * 100.0) if total_in > 0 else 0.0
hm_t = result.gas_phase.get("hm_mass_t", 0)

if 95 <= overall_pct <= 105:
    status_label = "Good"
    accent = "#166534"
    bg = "#f0fdf4"
elif 85 <= overall_pct <= 115:
    status_label = "Warning"
    accent = "#b45309"
    bg = "#fffbeb"
else:
    status_label = "Critical"
    accent = "#b91c1c"
    bg = "#fef2f2"

st.markdown(
    f"""
<div style="
    border-left:6px solid {accent};
    background:{bg};
    border-radius:10px;
    padding:0.75rem 1rem 0.85rem;
    margin:0.2rem 0 0.7rem 0;">
  <div style="font-size:0.82rem;color:#64748b;font-weight:700;letter-spacing:0.02em;">
    Overall closure | {status_label}
  </div>
  <div style="font-size:2rem;font-weight:800;line-height:1.1;color:{accent};">
    {overall_pct:.1f} %
  </div>
  <div style="font-size:0.92rem;color:#334155;margin-top:0.22rem;">
    In {total_in:,.0f} t | Out {total_out:,.0f} t
    {f"| HM {hm_t:,.0f} t" if hm_t > 0 else ""}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Warnings bar ──────────────────────────────────────────────────────
for w in result.warnings:
    st.warning(w, icon="⚠️")

# ── Main layout: furnace diagram (left 45%) | closure table (right 55%) ─────
cfg = load_full_config()

fig_col, table_col = st.columns([0.45, 0.55])

with fig_col:
    st.markdown("**Furnace mass flows**")
    burden_t = sum(result.material_masses.values())
    wind_nm3h = result.gas_phase.get("wind_nm3h", 0)
    steam_kgh = result.gas_phase.get("steam_kgh", 0)
    hot_blast_t = wind_nm3h * 24 * 1.293 / 1000
    pci_steam_t = result.material_masses.get("PCI", 0) + steam_kgh * 24 / 1000
    top_gas_t = sum(
        stream_dict.get("Top Gas", 0.0)
        for stream_dict in result.outputs.values()
    )
    in_labels = {
        "Burden": burden_t,
        "Hot Blast": hot_blast_t,
        "PCI": pci_steam_t,
    }
    out_labels = {
        "Top Gas": top_gas_t,
        "Hot Metal": result.gas_phase.get("hm_mass_t", 0),
        "Slag": result.gas_phase.get("slag_mass_t", 0),
        "Dust Catcher": result.gas_phase.get("dust_catcher_t", 0),
    }
    diagram = build_furnace_diagram(in_labels, out_labels)
    st.plotly_chart(diagram, width='stretch', key="furnace_diagram")

with table_col:
    st.markdown("**Element closure**")
    thresholds = cfg.get("closure_thresholds", {})
    good = tuple(thresholds.get("good", [95, 105]))
    warning_thr = tuple(thresholds.get("warning", [85, 115]))
    styler = style_closure_table(
        result.closure_table, good=good, warning=warning_thr
    )
    st.dataframe(styler, width='stretch', hide_index=True)

# Full-width per-element breakdown (optional deep-dive)
with st.expander("📊 Per-element input/output breakdown", expanded=False, key="mb_element_breakdown"):
    fig_bars = build_per_element_bars(
        result.closure_table, result.inputs, result.outputs
    )
    st.plotly_chart(fig_bars, width='stretch', key="bars_fig")

# ── Ash Analysis editor expander ───────────────────────────────────────
with st.expander("🔬 Ash Analysis — Coke / Nut Coke / PCI", expanded=False, key="mb_ash_analysis"):
    st.caption(
        "Enter the **ash chemical analysis** (wt % of ash) from lab reports. "
        "These values are used directly in the element balance to compute "
        "Si, Ca, Mg, Al, Fe, Na, K, Ti, S and P entering the furnace via fuel ash. "
        "Na₂O and K₂O: enter individually if the split is known; "
        "otherwise put the total alkali in K₂O and leave Na₂O = 0."
    )

    _ASH_LABELS = {
        "coke":    "EML Coke",
        "nutcoke": "Nut Coke",
        "pci":     "PCI Coal",
    }
    ash_tab_coke, ash_tab_nut, ash_tab_pci = st.tabs(
        [_ASH_LABELS[k] for k in ("coke", "nutcoke", "pci")]
    )

    for _ash_tab, _mat_key in [
        (ash_tab_coke, "coke"),
        (ash_tab_nut,  "nutcoke"),
        (ash_tab_pci,  "pci"),
    ]:
        with _ash_tab:
            _yml_key = ASH_MATERIAL_CONFIG_KEYS[_mat_key]
            _current = cfg.get(_yml_key, {}) or {}
            _updated: dict = {}

            _ncols = 5
            _field_cols = st.columns(_ncols)
            for _i, (_fkey, _flabel, _fkind) in enumerate(ASH_ANALYSIS_FIELDS):
                _unit = "wt% ash"
                _help = (
                    f"{'Oxide' if _fkind == 'oxide' else 'Element'} wt% of ash. "
                    "Leave 0 if not measured."
                )
                with _field_cols[_i % _ncols]:
                    _updated[_fkey] = st.number_input(
                        f"% {_flabel}",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(_current.get(_fkey, 0.0) or 0.0),
                        step=0.001,
                        format="%.3f",
                        key=f"ash_{_mat_key}_{_fkey}",
                        help=_help,
                    )

            _total = sum(_updated.values())
            _other = max(0.0, 100.0 - _total)
            _colour = "green" if _total <= 100.0 else "red"
            st.markdown(
                f"Sum of entered components: "
                f"<span style='color:{_colour}'><b>{_total:.2f} %</b></span> "
                f"\u00b7 Unaccounted / other: **{_other:.2f} %**",
                unsafe_allow_html=True,
            )

            if st.button(
                f"Save {_ASH_LABELS[_mat_key]} ash analysis",
                key=f"save_ash_{_mat_key}",
            ):
                save_ash_analyses({_mat_key: _updated})
                st.success(
                    f"{_ASH_LABELS[_mat_key]} ash analysis saved. "
                    "Rerunning balance…"
                )
                clear_day_caches(selected_day)
                st.rerun()

# ── DPR field mapping expander ─────────────────────────────────────────
current_mapping = load_dpr_mapping()
mapping_ok = mapping_is_complete(current_mapping)
with st.expander(
    "🗂 DPR field mapping" + ("" if mapping_ok else " (optional)"),
    expanded=False,
    key="mb_dpr_mapping",
):
    st.caption(
        "The DPR mapping is optional when using the static dataset. "
        "It allows you to override material masses from the live InfluxDB "
        "daily production report if configured."
    )
    dpr_fields = discover_dpr_fields(selected_day)
    if not dpr_fields:
        st.info(
            "No DPR data found for the selected day in InfluxDB. "
            "The page uses static ML dataset masses instead."
        )
    else:
        options = ["(unmapped)"] + dpr_fields
        new_mapping: dict[str, str | None] = {}
        cols_a, cols_b, cols_c = st.columns(3)
        for i, field_name in enumerate(CANONICAL_MASS_FIELDS):
            target_col = [cols_a, cols_b, cols_c][i % 3]
            with target_col:
                current = current_mapping.get(field_name) or "(unmapped)"
                idx = options.index(current) if current in options else 0
                choice = st.selectbox(
                    field_name.replace("_", " ").title(),
                    options,
                    index=idx,
                    key=f"dpr_map_{field_name}",
                )
                new_mapping[field_name] = None if choice == "(unmapped)" else choice

        if st.button("Save DPR mapping"):
            save_dpr_mapping(new_mapping)
            st.success("Mapping saved. Rerunning balance…")
            clear_day_caches(selected_day)
            st.rerun()

# ── Assumptions & limitations expander ────────────────────────────────
with st.expander("ℹ️ Assumptions & limitations", key="mb_assumptions"):
    dust_comp = cfg.get("dust_catcher_composition_pct", {})
    dust_comp_str = (
        ", ".join(f"{k}: {v}%" for k, v in dust_comp.items() if k != "other")
        if dust_comp
        else "not configured"
    )
    st.markdown(
        f"""
**Data source**: Static furnace ML dataset.
{result.n_rm_rows} hourly rows found for the selected day.

**RM lag**: {result.rm_lag_hours} h &nbsp;|&nbsp; **Blast lag**: {result.blast_lag_hours} h

**Material masses**: {"DPR override applied" if result.used_dpr else "Summed from hourly *_CALC_MT columns in the static dataset"}.

**Ash composition**: Coke and PCI ash distributed into oxides using constants in
`material_balance.yml` (coke_ash_assumption_pct / pci_ash_assumption_pct).

**Dust catcher composition** (wt%): {dust_comp_str}.
Override in `material_balance.yml → dust_catcher_composition_pct`.

**Slag mass**: {"DPR value used" if result.used_dpr else f"Fallback 0.30 × HM = {result.gas_phase.get('slag_mass_t', 0):,.0f} t"}.

**Top-gas volume**: {result.gas_phase.get('top_gas_nm3_per_day', 0):,.0f} Nm³/day
(wind = {result.gas_phase.get('wind_nm3h', 0):,.0f} Nm³/h).
From `bosh_vol_from_formula`; capped at 10× daily wind if sanity check fails.

**LOI**: Loss-on-ignition on ore, pellet, flux dropped in v1 — future update
will split into CO₂ + H₂O via per-material yml config.

**Unaccounted stream**: Sludge and granulation losses not yet modelled.
"""
    )
