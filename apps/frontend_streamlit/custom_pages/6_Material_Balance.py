"""Material Balance Visualiser - BF2."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st

from apps.frontend_streamlit.plotters.material_balance_plots import build_furnace_diagram, build_per_element_bars, style_closure_table
from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.services.material_balance_gateway import adapt_result_for_plotters, get_material_balance_gateway
from apps.frontend_streamlit.utils.session import is_logged_in

if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

STATE_CONFIG = "material_balance.config"
STATE_LAST_REQUEST = "material_balance.last_submitted_request"
STATE_LAST_RESULT = "material_balance.last_result"
STATE_RESULT_STALE = "material_balance.result_stale"
STATE_DPR = "material_balance.dpr_mapping_draft"
STATE_ASH = "material_balance.ash_draft"
STATE_ARTIFACT_BYTES = "material_balance.artifact_bytes"


def _api_token() -> str | None:
    token = str(st.session_state.get("auth_access_token") or "").strip()
    return token or None


def _request_id(exc: Exception) -> str:
    request_id = getattr(exc, "request_id", None)
    return f" Request ID: {request_id}." if request_id else ""


def _parse_date(value: object, fallback: date) -> date:
    if isinstance(value, date):
        return value
    if value:
        return date.fromisoformat(str(value))
    return fallback


def _today_ist() -> date:
    return datetime.now(ZoneInfo("Asia/Kolkata")).date()


def _permissions() -> set[str]:
    return {str(item) for item in st.session_state.get("permissions") or []}


def _can_write_config(config: dict) -> bool:
    caps = config.get("capabilities") or {}
    return bool(caps.get("runtime_configuration_writable")) and "material_balance:config:write" in _permissions()


def _fingerprint(payload: dict) -> tuple:
    options = payload.get("options") or {}
    return (
        payload.get("day"),
        payload.get("expected_dataset_version"),
        payload.get("expected_config_version"),
        int(options.get("rm_lag_hours") or 0),
        int(options.get("blast_lag_hours") or 0),
        float(options.get("dust_catcher_t") or 0.0),
        options.get("algorithm_version"),
        payload.get("export_format"),
    )


st.title("Material Balance - BF2")

try:
    gateway = get_material_balance_gateway(access_token=_api_token())
except FrontendApiError as exc:
    st.error(f"Material Balance API authentication failed: {exc.message}.{_request_id(exc)}")
    st.stop()

try:
    config = gateway.get_config()
    st.session_state[STATE_CONFIG] = config
except FrontendApiError as exc:
    config = st.session_state.get(STATE_CONFIG)
    if not config:
        st.error(f"Material Balance configuration could not be loaded: {exc.message}.{_request_id(exc)}")
        st.stop()
    st.warning(f"Using the last loaded configuration. Refresh failed: {exc.message}.{_request_id(exc)}")

range_info = (config.get("dataset") or {}).get("available_date_range") or {}
max_completed = _today_ist() - timedelta(days=1)
min_day = _parse_date(range_info.get("minimum"), date(2024, 1, 1))
max_day = min(_parse_date(range_info.get("maximum"), max_completed), max_completed)
if max_day < min_day:
    max_day = max_completed

dataset = config.get("dataset") or {}
st.caption(f"Data source: static ML dataset | Dataset version: {dataset.get('version') or 'unavailable'} | Config version: {config.get('effective_config_version')}")
if dataset.get("status") != "ready":
    st.warning("Static ML dataset is not ready for Material Balance runs.")

limits = config.get("limits") or {}
defaults = config.get("defaults") or {}

with st.form("material_balance_run_form"):
    col1, col2, col3, col4 = st.columns([0.25, 0.22, 0.22, 0.31])
    with col1:
        selected_day = st.date_input("Date (IST)", value=max_day, min_value=min_day, max_value=max_day, help="Pick a completed IST calendar day.")
    with col2:
        rm_lag_hours = st.number_input("RM lag (h)", min_value=int(limits.get("rm_lag_hours_min", 0)), max_value=int(limits.get("rm_lag_hours_max", 240)), value=int(defaults.get("rm_lag_hours", 0)), step=24)
    with col3:
        blast_lag_hours = st.number_input("Blast lag (h)", min_value=int(limits.get("blast_lag_hours_min", 0)), max_value=int(limits.get("blast_lag_hours_max", 48)), value=int(defaults.get("blast_lag_hours", 0)), step=1)
    with col4:
        dust_catcher_t = st.number_input("Dust Catcher (t/day)", min_value=float(limits.get("dust_catcher_t_min", 0.0)), max_value=float(limits.get("dust_catcher_t_max", 500.0)), value=float(defaults.get("dust_catcher_t", 0.0)), step=1.0)
    export_choice = st.selectbox("Export", ["None", "Closure CSV", "Full JSON"], index=0)
    submitted = st.form_submit_button("Run Material Balance", type="primary")

run_request = {
    "source": "static_dataset",
    "day": selected_day.isoformat(),
    "expected_dataset_version": dataset.get("version"),
    "expected_config_version": config.get("effective_config_version"),
    "options": {
        "rm_lag_hours": int(rm_lag_hours),
        "blast_lag_hours": int(blast_lag_hours),
        "dust_catcher_t": float(dust_catcher_t),
        "algorithm_version": defaults.get("algorithm_version", "legacy_v1"),
    },
    "export_format": {"None": None, "Closure CSV": "closure_csv", "Full JSON": "full_json"}[export_choice],
}
if st.session_state.get(STATE_LAST_REQUEST) and _fingerprint(run_request) != _fingerprint(st.session_state[STATE_LAST_REQUEST]):
    st.session_state[STATE_RESULT_STALE] = True

refresh_col, _ = st.columns([0.22, 0.78])
with refresh_col:
    if st.button("Refresh data", width="stretch"):
        try:
            gateway.refresh_cache({"day": selected_day.isoformat(), "scopes": ["calculation_snapshot", "dpr"]})
            st.session_state[STATE_RESULT_STALE] = True
            st.success("Material Balance caches refreshed.")
        except FrontendApiError as exc:
            st.error(f"Refresh failed: {exc.message}.{_request_id(exc)}")

if submitted:
    try:
        with st.spinner("Computing element balance..."):
            result = gateway.run(run_request)
        st.session_state[STATE_LAST_REQUEST] = run_request
        st.session_state[STATE_LAST_RESULT] = result
        st.session_state[STATE_RESULT_STALE] = False
    except FrontendApiError as exc:
        st.session_state[STATE_RESULT_STALE] = True
        st.error(f"Material Balance run failed: {exc.message}.{_request_id(exc)}")

result = st.session_state.get(STATE_LAST_RESULT)
if result and st.session_state.get(STATE_RESULT_STALE):
    st.info("The displayed result was calculated with earlier controls or configuration. Run again to update it.")

if result:
    summary = result.get("summary") or {}
    status = str(summary.get("closure_status") or "unavailable").title()
    palette = {"Good": ("#166534", "#f0fdf4"), "Warning": ("#b45309", "#fffbeb"), "Critical": ("#b91c1c", "#fef2f2")}
    accent, bg = palette.get(status, ("#475569", "#f8fafc"))
    st.markdown(
        f"""
<div style="border-left:6px solid {accent};background:{bg};border-radius:8px;padding:0.75rem 1rem 0.85rem;margin:0.2rem 0 0.7rem 0;">
  <div style="font-size:0.82rem;color:#64748b;font-weight:700;">Overall closure | {status}</div>
  <div style="font-size:2rem;font-weight:800;line-height:1.1;color:{accent};">{summary.get('overall_closure_pct') if summary.get('overall_closure_pct') is not None else 'N/A'} %</div>
  <div style="font-size:0.92rem;color:#334155;margin-top:0.22rem;">In {summary.get('total_input_element_t', 0):,.0f} t | Out {summary.get('total_output_element_t', 0):,.0f} t</div>
</div>
""",
        unsafe_allow_html=True,
    )

    for warning in result.get("warnings") or []:
        st.warning(warning.get("message") if isinstance(warning, dict) else str(warning))

    resolved = result.get("resolved_windows") or {}
    if resolved:
        with st.expander("Resolved windows", expanded=False):
            st.json(resolved)

    adapted = adapt_result_for_plotters(result)
    fig_col, table_col = st.columns([0.45, 0.55])
    with fig_col:
        st.markdown("**Furnace mass flows**")
        flows = result.get("diagram_flows") or {}
        in_labels = {item["label"]: item.get("mass_t") or 0.0 for item in flows.get("inputs") or []}
        out_labels = {item["label"]: item.get("mass_t") or 0.0 for item in flows.get("outputs") or []}
        st.plotly_chart(build_furnace_diagram(in_labels, out_labels), width="stretch", key="furnace_diagram")
    with table_col:
        st.markdown("**Element closure**")
        thresholds = result.get("closure_thresholds") or config.get("closure_thresholds") or {}
        good = thresholds.get("good") or {"minimum": 95, "maximum": 105}
        warning_thr = thresholds.get("warning") or {"minimum": 85, "maximum": 115}
        styler = style_closure_table(adapted["closure_table"], good=(good["minimum"], good["maximum"]), warning=(warning_thr["minimum"], warning_thr["maximum"]))
        st.dataframe(styler, width="stretch", hide_index=True)

    with st.expander("Per-element input/output breakdown", expanded=False, key="mb_element_breakdown"):
        st.plotly_chart(build_per_element_bars(adapted["closure_table"], adapted["inputs"], adapted["outputs"]), width="stretch", key="bars_fig")

    for artifact in result.get("artifacts") or []:
        artifact_id = artifact.get("artifact_id")
        if artifact_id and st.button(f"Prepare {artifact.get('filename', 'artifact')}", key=f"prep_{artifact_id}"):
            try:
                st.session_state.setdefault(STATE_ARTIFACT_BYTES, {})[artifact_id] = gateway.download_artifact(artifact_id)
            except FrontendApiError as exc:
                st.error(f"Export download failed: {exc.message}.{_request_id(exc)}")
        data = st.session_state.get(STATE_ARTIFACT_BYTES, {}).get(artifact_id)
        if data:
            st.download_button("Download export", data=data, file_name=artifact.get("filename") or "material_balance_export", mime=artifact.get("content_type") or "application/octet-stream", key=f"download_{artifact_id}")
else:
    st.info("Choose a completed day and run Material Balance to see the closure, flows and assumptions.")
with st.expander("Ash Analysis - Coke / Nut Coke / PCI", expanded=False, key="mb_ash_analysis"):
    try:
        ash_data = gateway.get_ash_analyses()
        st.session_state[STATE_ASH] = ash_data
    except FrontendApiError as exc:
        ash_data = st.session_state.get(STATE_ASH)
        st.error(f"Ash analysis could not be loaded: {exc.message}.{_request_id(exc)}")
    if ash_data:
        can_save = _can_write_config(config)
        edited_materials = []
        tabs = st.tabs([m.get("label", m.get("material_id", "Material")) for m in ash_data.get("materials") or []])
        for tab, material in zip(tabs, ash_data.get("materials") or []):
            with tab:
                species_values = []
                ash_total = 0.0
                cols = st.columns(5)
                for idx, species in enumerate(material.get("species") or []):
                    basis = species.get("basis", "ash")
                    label = f"% {species.get('label') or species.get('species_id')}" + (" (net fuel)" if basis == "net_fuel" else "")
                    with cols[idx % 5]:
                        value = st.number_input(label, min_value=0.0, max_value=100.0, value=float(species.get("value") or 0.0), step=0.001, format="%.3f", key=f"ash_{material['material_id']}_{species['species_id']}")
                    species_values.append({**species, "value": float(value)})
                    if basis == "ash":
                        ash_total += float(value)
                st.caption(f"Ash-basis total: {ash_total:.2f}%")
                edited_materials.append({"material_id": material["material_id"], "label": material.get("label"), "species": species_values})
        if can_save:
            if st.button("Save ash analyses"):
                try:
                    updated = gateway.update_ash_analyses({"expected_config_version": ash_data.get("config_version"), "materials": edited_materials})
                    st.session_state[STATE_ASH] = updated
                    st.session_state[STATE_RESULT_STALE] = True
                    st.success("Ash analyses saved.")
                except FrontendApiError as exc:
                    st.error(f"Ash save failed: {exc.message}.{_request_id(exc)}")
        else:
            st.info("Ash analysis save requires material_balance:config:write permission.")

with st.expander("DPR field mapping", expanded=False, key="mb_dpr_mapping"):
    st.caption("DPR mapping uses the backend-approved offline report/database source. It is optional; unmapped materials fall back to the static dataset.")
    load_clicked = st.button("Load / Refresh DPR Mapping")
    if load_clicked or STATE_DPR not in st.session_state:
        try:
            st.session_state[STATE_DPR] = gateway.get_dpr_mapping(sample_day=selected_day.isoformat())
        except FrontendApiError as exc:
            st.error(f"DPR mapping could not be loaded: {exc.message}.{_request_id(exc)}")
    dpr_data = st.session_state.get(STATE_DPR)
    if dpr_data:
        st.write(f"Mapping status: **{str(dpr_data.get('status', 'none')).title()}**")
        options = [None] + [field["source_field_id"] for field in dpr_data.get("approved_source_fields") or []]
        labels = {None: "(unmapped)", **{field["source_field_id"]: f"{field['label']} | {field['unit']} | {field['aggregation_policy']}" for field in dpr_data.get("approved_source_fields") or []}}
        current = {item["canonical_field_id"]: item.get("source_field_id") for item in dpr_data.get("mapping") or []}
        new_mapping = []
        cols = st.columns(3)
        for idx, field in enumerate(dpr_data.get("canonical_fields") or []):
            canonical = field["canonical_field_id"]
            with cols[idx % 3]:
                choice = st.selectbox(field.get("label") or canonical, options, index=options.index(current.get(canonical)) if current.get(canonical) in options else 0, format_func=lambda value: labels.get(value, str(value)), key=f"dpr_{canonical}")
            new_mapping.append({"canonical_field_id": canonical, "source_field_id": choice})
        if _can_write_config(config):
            if st.button("Save DPR mapping"):
                try:
                    updated = gateway.update_dpr_mapping({"expected_config_version": dpr_data.get("config_version"), "mapping": new_mapping})
                    st.session_state[STATE_DPR] = updated
                    st.session_state[STATE_RESULT_STALE] = True
                    st.success("DPR mapping saved.")
                except FrontendApiError as exc:
                    st.error(f"DPR mapping save failed: {exc.message}.{_request_id(exc)}")
        else:
            st.info("DPR mapping save requires material_balance:config:write permission.")

with st.expander("Assumptions & limitations", key="mb_assumptions"):
    if result:
        versions = result.get("versions") or {}
        st.markdown(
            f"""
**Algorithm**: {result.get('algorithm_version')}

**Window policy**: {result.get('window_policy_version')}

**Dataset version**: {versions.get('dataset_version')}

**Config version**: {versions.get('config_version')}
"""
        )
        for assumption in result.get("assumptions") or []:
            st.markdown(f"**{assumption.get('label')}**: {assumption.get('text')}")
    else:
        st.info("Assumptions are returned with each backend calculation result.")