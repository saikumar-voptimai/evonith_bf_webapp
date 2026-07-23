from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.services.vsense_gateway import get_vsense_gateway


CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "css" / "recommendation_style.css"
SESSION_PREFIX = "vsense."


def _api_token() -> str | None:
    token = str(st.session_state.get("auth_access_token") or "").strip()
    return token or None


def _session(name: str, default: Any = None) -> Any:
    key = f"{SESSION_PREFIX}{name}"
    if key not in st.session_state and default is not None:
        st.session_state[key] = default
    return st.session_state.get(key)


def _set_session(name: str, value: Any) -> None:
    st.session_state[f"{SESSION_PREFIX}{name}"] = value


def _new_key(scope: str) -> str:
    return f"{scope}-{uuid.uuid4().hex}"


def _request_id_text(payload: dict[str, Any] | None) -> str:
    request_id = (payload or {}).get("request_id")
    return f" Request ID: {request_id}." if request_id else ""


def _handle_error(title: str, exc: FrontendApiError) -> None:
    suffix = f" Request ID: {exc.request_id}." if exc.request_id else ""
    code = f" ({exc.error_code})" if exc.error_code else ""
    st.error(f"{title}{code}: {exc.message}.{suffix}")


def _load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _parameter_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in catalog.get("parameters") or []}


def _optimization_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in catalog.get("optimization_types") or []}


def _current_gateway():
    return get_vsense_gateway(access_token=_api_token())


def _load_catalog(force: bool = False) -> dict[str, Any] | None:
    cached = _session("catalog")
    if cached and not force:
        return cached
    try:
        catalog = _current_gateway().get_catalog()
    except FrontendApiError as exc:
        _handle_error("Unable to load V-Sense catalog", exc)
        if cached:
            st.warning("Showing the previous catalog snapshot.")
            return cached
        return None
    _set_session("catalog", catalog)
    _set_session("catalog_version", catalog.get("catalog_version"))
    return catalog


def _load_profile(optimization_type_id: str, *, force: bool = False) -> dict[str, Any] | None:
    profile = _session("control_profile")
    if profile and profile.get("optimization_type_id") == optimization_type_id and not force:
        return profile
    try:
        profile = _current_gateway().get_control_profile(optimization_type_id)
    except FrontendApiError as exc:
        _handle_error("Unable to load control profile", exc)
        return None
    _set_session("control_profile", profile)
    _set_session("control_draft", [dict(item) for item in profile.get("parameters") or []])
    _set_session("control_draft_dirty", False)
    return profile


def _load_context(optimization_type_id: str) -> dict[str, Any] | None:
    try:
        context = _current_gateway().create_context(
            {"optimization_type_id": optimization_type_id, "data_mode": "live", "as_of": None},
            idempotency_key=_new_key("context"),
        )
    except FrontendApiError as exc:
        _handle_error("Unable to load current context", exc)
        return None
    _set_session("context", context)
    _set_session("context_fingerprint", context.get("context_id"))
    _set_session("input_overrides", {})
    _load_profile(optimization_type_id, force=True)
    return context


def _context_matches(context: dict[str, Any] | None, optimization_type_id: str) -> bool:
    return bool(context and context.get("optimization_type_id") == optimization_type_id)


def _format_value(value: Any, precision: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    return f"{number:.{max(0, int(precision))}f}"


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _context_value_map(context: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {item["parameter_id"]: item for item in (context or {}).get("controls") or []}


def _render_context(context: dict[str, Any] | None) -> None:
    st.subheader("Current Context")
    if not context:
        st.info("Load the current context before running the optimiser.")
        return
    dataset = context.get("dataset") or {}
    cols = st.columns(4)
    cols[0].metric("As of", str(context.get("as_of") or "--")[:19])
    cols[1].metric("Dataset", str(dataset.get("version") or "--")[:18])
    cols[2].metric("Freshness", f"{dataset.get('staleness_seconds', '--')} s")
    cols[3].metric("Profile", f"v{(context.get('control_profile') or {}).get('version', '--')}")
    warnings = context.get("warnings") or []
    for warning in warnings:
        st.warning(str(warning))


def _render_controls(
    catalog: dict[str, Any],
    optimization: dict[str, Any],
    context: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    st.subheader("Control Parameters")
    params = _parameter_map(catalog)
    values = _context_value_map(context)
    draft = _session("control_draft") or [dict(item) for item in (profile or {}).get("parameters") or []]
    draft_by_id = {item["parameter_id"]: dict(item) for item in draft}
    updated: list[dict[str, Any]] = []
    cols = st.columns(3)
    for index, parameter_id in enumerate(optimization.get("control_parameter_ids") or []):
        definition = params[parameter_id]
        current = values.get(parameter_id) or {}
        item = draft_by_id.get(parameter_id) or {
            "parameter_id": parameter_id,
            "mode": "optimize",
            "lower_bound": definition.get("approved_min"),
            "upper_bound": definition.get("approved_max"),
            "fixed_value": None,
        }
        with cols[index % 3]:
            st.markdown(f"<div class='param-title'>{definition['label']}</div>", unsafe_allow_html=True)
            st.caption(
                "Current "
                f"{_format_value(current.get('value'), definition.get('precision', 2))} "
                f"{definition.get('unit') or ''} | source {current.get('source', '--')}"
            )
            mode = st.radio(
                "Mode",
                ["optimize", "fixed"],
                horizontal=True,
                index=0 if item.get("mode") != "fixed" else 1,
                key=f"vsense.control.{parameter_id}.mode",
            )
            lower_default = _finite_or_none(item.get("lower_bound"))
            upper_default = _finite_or_none(item.get("upper_bound"))
            approved_min = _finite_or_none(definition.get("approved_min"))
            approved_max = _finite_or_none(definition.get("approved_max"))
            lower_value = lower_default if lower_default is not None else approved_min or 0.0
            upper_value = upper_default if upper_default is not None else approved_max or lower_value
            lower = st.number_input(
                "Min",
                value=float(lower_value),
                key=f"vsense.control.{parameter_id}.min",
            )
            upper = st.number_input(
                "Max",
                value=float(upper_value),
                key=f"vsense.control.{parameter_id}.max",
            )
            fixed_default = _finite_or_none(item.get("fixed_value"))
            if fixed_default is None:
                fixed_default = _finite_or_none(current.get("value")) or float(lower)
            fixed = st.number_input(
                "Fixed value",
                value=float(fixed_default),
                disabled=mode != "fixed",
                key=f"vsense.control.{parameter_id}.fixed",
            )
            st.caption(
                "Observed "
                f"{_format_value(current.get('observed_min'), definition.get('precision', 2))} to "
                f"{_format_value(current.get('observed_max'), definition.get('precision', 2))}; "
                "approved "
                f"{_format_value(definition.get('approved_min'), definition.get('precision', 2))} to "
                f"{_format_value(definition.get('approved_max'), definition.get('precision', 2))}"
            )
            updated.append(
                {
                    "parameter_id": parameter_id,
                    "mode": mode,
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "fixed_value": float(fixed) if mode == "fixed" else None,
                }
            )
    if updated != draft:
        _set_session("control_draft_dirty", True)
    _set_session("control_draft", updated)
    dirty = bool(_session("control_draft_dirty"))
    if dirty:
        st.caption("Unsaved control-profile draft")
    save_disabled = not profile or not context or not dirty
    if st.button("Submit CP & Save bounds", disabled=save_disabled, type="secondary"):
        try:
            saved = _current_gateway().update_control_profile(
                optimization["id"],
                {
                    "profile_id": (profile or {}).get("profile_id", "plant-default"),
                    "expected_version": int((profile or {}).get("version", 0)),
                    "parameters": updated,
                },
                idempotency_key=_new_key("profile"),
            )
        except FrontendApiError as exc:
            if exc.status_code == 409:
                st.warning("Control profile changed on the backend. Reloading the latest profile.")
                _load_profile(optimization["id"], force=True)
            _handle_error("Unable to save control profile", exc)
        else:
            st.success("Bounds saved successfully.")
            _set_session("control_profile", saved)
            _set_session("control_draft", [dict(item) for item in saved.get("parameters") or []])
            _set_session("control_draft_dirty", False)
    return updated


def _render_inputs(catalog: dict[str, Any], context: dict[str, Any] | None) -> list[dict[str, Any]]:
    params = _parameter_map(catalog)
    overrides = dict(_session("input_overrides", {}) or {})
    if not context:
        return []
    with st.expander("Input Parameters - Click to expand and override"):
        if st.button("Clear input overrides"):
            overrides = {}
            _set_session("input_overrides", overrides)
        for group in context.get("input_groups") or []:
            st.write(f"### {group.get('label')}")
            cols = st.columns(3)
            for index, value_row in enumerate(group.get("values") or []):
                parameter_id = value_row["parameter_id"]
                definition = params.get(parameter_id, {"label": parameter_id, "precision": 2})
                default = _finite_or_none(value_row.get("value")) or 0.0
                current_override = overrides.get(parameter_id)
                with cols[index % 3]:
                    value = st.number_input(
                        definition["label"],
                        value=float(current_override if current_override is not None else default),
                        key=f"vsense.input.{parameter_id}",
                    )
                    st.caption(f"{value_row.get('source', '--')} | {definition.get('unit') or ''}")
                    if _finite_or_none(value) != _finite_or_none(default):
                        overrides[parameter_id] = float(value)
                    else:
                        overrides.pop(parameter_id, None)
        if st.button("Submit Input Params"):
            _set_session("input_overrides", overrides)
            st.success("Input parameters recorded.")
    return [
        {"parameter_id": key, "value": float(value)}
        for key, value in sorted(overrides.items())
        if _finite_or_none(value) is not None
    ]


def _render_run_panel(
    catalog: dict[str, Any],
    optimization: dict[str, Any],
    context: dict[str, Any] | None,
    control_plan: list[dict[str, Any]],
    input_overrides: list[dict[str, Any]],
) -> None:
    st.subheader("Optimiser")
    limits = catalog.get("limits") or {}
    capabilities = catalog.get("capabilities") or {}
    cols = st.columns(2)
    with cols[0]:
        lambda_reg = st.slider(
            "Regularisation Parameter (Lambda)",
            min_value=float(limits.get("lambda_min", 0.0)),
            max_value=float(limits.get("lambda_max", 0.5)),
            value=0.05,
            step=0.005,
        )
    with cols[1]:
        request_review = st.checkbox(
            "Run LLM Analysis",
            value=False,
            disabled=not bool(capabilities.get("llm_review_available")),
        )
    valid_context = _context_matches(context, optimization["id"])
    if st.button("Run Optimiser", type="primary", disabled=not valid_context):
        try:
            accepted = _current_gateway().create_run(
                {
                    "context_id": context["context_id"],
                    "optimization_type_id": optimization["id"],
                    "control_plan": control_plan,
                    "input_overrides": input_overrides,
                    "options": {
                        "lambda_reg": float(lambda_reg),
                        "iteration_budget_id": "standard",
                        "request_llm_review": bool(request_review),
                        "advanced_diagnostics": False,
                    },
                },
                idempotency_key=_new_key("run"),
            )
        except FrontendApiError as exc:
            if exc.error_code == "VSENSE_CONTEXT_EXPIRED":
                _set_session("context", None)
            _handle_error("Unable to start V-Sense run", exc)
            return
        _set_session("active_run_id", accepted["run_id"])
        _set_session("last_event_sequence", 0)
        st.success("Optimiser run requested.")
    elif not valid_context:
        st.info("Load a matching current context before running the optimiser.")


def _poll_run() -> dict[str, Any] | None:
    run_id = _session("active_run_id")
    if not run_id:
        return _session("last_completed_result")
    try:
        status = _current_gateway().get_run(str(run_id))
        events = _current_gateway().get_run_events(
            str(run_id),
            after=int(_session("last_event_sequence", 0) or 0),
        )
    except FrontendApiError as exc:
        _handle_error("Unable to refresh V-Sense run", exc)
        previous = _session("last_completed_result")
        if previous:
            st.warning("Showing the previous successful result; current run status is stale.")
        return previous
    for event in events.get("events") or []:
        _set_session("last_event_sequence", event.get("sequence", 0))
        st.caption(f"{event.get('stage')}: {event.get('message')}")
    _set_session("run_status", status)
    if status.get("cancellable") and st.button("Cancel Run"):
        try:
            status = _current_gateway().cancel_run(str(run_id))
        except FrontendApiError as exc:
            _handle_error("Unable to cancel V-Sense run", exc)
    if status.get("status") == "completed" and status.get("result"):
        _set_session("last_completed_result", status)
        _set_session("active_run_id", None)
    elif status.get("status") in {"failed", "cancelled", "expired"}:
        _set_session("active_run_id", None)
    return status


def _render_results(status: dict[str, Any] | None, catalog: dict[str, Any]) -> None:
    if not status:
        return
    progress = status.get("progress")
    if status.get("status") not in {"completed", "failed", "cancelled"}:
        st.progress(float(progress or 0.0) / 100.0, text=status.get("message") or "Running")
    result = status.get("result")
    if not result:
        if status.get("error_code"):
            st.error(f"{status.get('error_code')}: {status.get('error_message') or 'Run failed.'}")
        return
    st.subheader("Optimisation Results")
    st.info("V-Sense recommendations are advisory and require operator review before any action.")
    target = result.get("target") or {}
    st.metric(
        target.get("label") or target.get("parameter_id") or "Target",
        _format_value(target.get("recommended"), 3),
        _format_value(target.get("delta"), 3),
    )
    st.write("#### Control Changes")
    st.table(
        [
            {
                "Parameter": item.get("label"),
                "Mode": item.get("mode"),
                "Baseline": _format_value(item.get("baseline"), 3),
                "Recommended": _format_value(item.get("recommended"), 3),
                "Delta": _format_value(item.get("delta"), 3),
                "At bound": item.get("at_bound"),
            }
            for item in result.get("controls") or []
        ]
    )
    impacts = result.get("impacts") or []
    if impacts:
        st.write("#### Impact Metrics")
        st.table(
            [
                {
                    "Metric": item.get("label"),
                    "Baseline": _format_value(item.get("baseline"), 3),
                    "Recommended": _format_value(item.get("recommended"), 3),
                    "Delta": _format_value(item.get("delta"), 3),
                }
                for item in impacts
            ]
        )
    deps = result.get("dependent_parameters") or []
    if deps:
        with st.expander("Dependent Parameters (Auto-calculated) - optional", expanded=False):
            st.table(
                [
                    {
                        "Parameter": item.get("label"),
                        "Baseline": _format_value(item.get("baseline"), 4),
                        "Recommended": _format_value(item.get("recommended"), 4),
                        "Delta": _format_value(item.get("delta"), 4),
                    }
                    for item in deps
                ]
            )
    feasibility = result.get("feasibility") or {}
    if feasibility.get("violations"):
        st.warning(f"Feasibility violations: {len(feasibility['violations'])}")
    review = result.get("review")
    if review:
        st.subheader("Recommendations")
        if review.get("markdown"):
            st.markdown(review["markdown"])
        for warning in review.get("warnings") or []:
            st.warning(str(warning))
    with st.expander("Diagnostics", expanded=False):
        st.json({"versions": result.get("versions"), "diagnostics": result.get("diagnostics")})


def main() -> None:
    st.markdown(
        """
        <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; ">
            V-Sense - Blast Parameter Optimisation
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    _load_css()

    catalog = _load_catalog()
    if not catalog:
        return
    optimizations = _optimization_map(catalog)
    labels = {item["label"]: item["id"] for item in optimizations.values()}
    selected_label = st.selectbox("Select Optimisation Type", list(labels.keys()))
    optimization_type_id = labels[selected_label]
    previous_type = _session("selected_optimization_type")
    if previous_type and previous_type != optimization_type_id:
        _set_session("context", None)
        _set_session("control_profile", None)
        _set_session("control_draft", None)
        _set_session("input_overrides", {})
    _set_session("selected_optimization_type", optimization_type_id)
    optimization = optimizations[optimization_type_id]

    top_cols = st.columns([1, 1, 2])
    if top_cols[0].button("Load / Refresh Current Context"):
        _load_context(optimization_type_id)
    if top_cols[1].button("Refresh Catalog"):
        catalog = _load_catalog(force=True) or catalog
    context = _session("context")
    if not _context_matches(context, optimization_type_id):
        context = None
    profile = _load_profile(optimization_type_id)

    _render_context(context)
    control_plan = _render_controls(catalog, optimization, context, profile)
    input_overrides = _render_inputs(catalog, context)
    _render_run_panel(catalog, optimization, context, control_plan, input_overrides)
    _render_results(_poll_run(), catalog)


main()
