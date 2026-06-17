"""Streamlit UI for Daemon Jobs control-plane definitions."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

import pandas as pd
import streamlit as st

from data.daemon_jobs import (
    DaemonJobConcurrencyPolicy,
    DaemonJobCriticality,
    DaemonJobKind,
    DaemonJobRestartPolicy,
    DaemonJobScheduleType,
    DaemonJobService,
    DaemonJobView,
    SystemdPreview,
    suggest_systemd_unit_name,
)
from data.daemon_jobs.validators import DEFAULT_JOB_PAYLOAD, JSON_FIELDS

JOB_KIND_OPTIONS = [item.value for item in DaemonJobKind]
SCHEDULE_OPTIONS = [item.value for item in DaemonJobScheduleType]
RESTART_OPTIONS = [item.value for item in DaemonJobRestartPolicy]
CONCURRENCY_OPTIONS = [item.value for item in DaemonJobConcurrencyPolicy]
CRITICALITY_OPTIONS = [item.value for item in DaemonJobCriticality]


@st.cache_resource(show_spinner=False)
def get_daemon_job_service() -> DaemonJobService:
    """Return a cached daemon job service instance for the page session."""
    return DaemonJobService()


def render_daemon_jobs_page() -> None:
    """Render the Daemon Jobs dashboard page."""
    actor = _require_daemon_job_access()
    service = get_daemon_job_service()

    st.title("Daemon Jobs")
    st.caption(
        "Control-plane definitions for LLM, LangGraph, and FurnaceMind agent jobs."
    )

    tabs = st.tabs(
        [
            "Overview",
            "Create / Edit Job",
            "Systemd Preview",
            "Audit Log",
            "Help / Runbook",
        ]
    )

    with tabs[0]:
        _render_overview_tab(service=service, actor=actor)
    with tabs[1]:
        _render_create_edit_tab(service=service, actor=actor)
    with tabs[2]:
        _render_preview_tab(service=service, actor=actor)
    with tabs[3]:
        _render_audit_log_tab(service=service)
    with tabs[4]:
        _render_help_tab()


def _require_daemon_job_access() -> str | None:
    """Guard page access to admins/supervisors where session helpers exist."""
    try:
        from utils.session import is_admin, is_logged_in, is_supervisor

        if not is_logged_in():
            st.warning("Please login to access this page.")
            st.stop()
        if not (is_admin() or is_supervisor()):
            st.error("Only admin and supervisor users can access Daemon Jobs.")
            st.stop()
    except ImportError:
        st.warning(
            "Role helpers are unavailable. Allowing the current logged-in session."
        )

    actor = str(st.session_state.get("auth_user", "")).strip()
    return actor or None


def _render_overview_tab(*, service: DaemonJobService, actor: str | None) -> None:
    """Render daemon job metrics, table, and lifecycle actions."""
    jobs = service.list_jobs()
    total_jobs = len(jobs)
    enabled_jobs = sum(1 for job in jobs if job.enabled)
    disabled_jobs = total_jobs - enabled_jobs
    critical_jobs = sum(1 for job in jobs if job.criticality == "critical")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total jobs", total_jobs)
    metric_cols[1].metric("Enabled", enabled_jobs)
    metric_cols[2].metric("Disabled", disabled_jobs)
    metric_cols[3].metric("Critical", critical_jobs)

    st.subheader("Job definitions")
    if jobs:
        st.dataframe(
            pd.DataFrame([_job_table_row(job) for job in jobs]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No daemon jobs have been created yet.")

    selected_job = _select_job(jobs, key="overview_job_select")
    if selected_job is None:
        return

    st.divider()
    st.markdown(f"**Selected:** {selected_job.name}")
    action_cols = st.columns([1, 1, 1, 1, 1])

    if action_cols[0].button("Edit", use_container_width=True):
        st.session_state["daemon_jobs_edit_id"] = selected_job.id
        st.success("Selected job loaded in the Create / Edit Job tab.")

    enable_label = "Disable" if selected_job.enabled else "Enable"
    if action_cols[1].button(enable_label, use_container_width=True):
        service.set_enabled(selected_job.id, not selected_job.enabled, actor=actor)
        st.success(f"Job {enable_label.lower()}d.")
        st.rerun()

    if action_cols[2].button("Clone", use_container_width=True):
        cloned = service.clone_job(selected_job.id, actor=actor)
        st.session_state["daemon_jobs_edit_id"] = cloned.id
        st.success(f"Cloned as {cloned.name}.")
        st.rerun()

    if action_cols[3].button("Preview", use_container_width=True):
        preview = service.preview_systemd(selected_job.id, actor=actor)
        _store_preview(preview)
        st.session_state["daemon_jobs_preview_job_id"] = selected_job.id
        st.success("Preview generated. Open the Systemd Preview tab to review it.")

    confirm_delete = st.checkbox(
        "Confirm delete",
        key=f"daemon_jobs_confirm_delete_{selected_job.id}",
    )
    if action_cols[4].button(
        "Delete",
        use_container_width=True,
        disabled=not confirm_delete,
    ):
        service.delete_job(selected_job.id, actor=actor)
        st.success("Job soft-deleted.")
        st.rerun()


def _render_create_edit_tab(*, service: DaemonJobService, actor: str | None) -> None:
    """Render create/edit form for daemon job definitions."""
    edit_id = st.session_state.get("daemon_jobs_edit_id")
    edit_job = service.get_job(edit_id) if edit_id else None
    if edit_id and edit_job is None:
        st.session_state.pop("daemon_jobs_edit_id", None)
        st.warning("The selected job is no longer available.")

    header_cols = st.columns([2, 1, 1])
    header_cols[0].subheader("Edit Job" if edit_job else "Create Job")
    preset_name = header_cols[1].selectbox(
        "Template",
        options=["Blank", *PRESETS.keys()],
        key="daemon_jobs_preset_name",
    )
    if header_cols[2].button("Load Template", use_container_width=True):
        st.session_state["daemon_jobs_form_seed"] = _preset_payload(preset_name)
        st.session_state.pop("daemon_jobs_edit_id", None)
        st.rerun()

    if edit_job and st.button("Start New Job"):
        st.session_state.pop("daemon_jobs_edit_id", None)
        st.session_state["daemon_jobs_form_seed"] = _default_payload()
        st.rerun()

    seed_payload = _payload_from_job(edit_job) if edit_job else _current_seed_payload()
    submitted_payload, validate_clicked, save_clicked = _render_job_form(
        payload=seed_payload,
        is_edit=edit_job is not None,
    )

    if not (validate_clicked or save_clicked):
        return

    existing_job_id = edit_job.id if edit_job else None
    validation = service.validate_payload(
        submitted_payload, existing_job_id=existing_job_id
    )
    _render_validation_result(validation)

    if not save_clicked:
        return
    if not validation.is_valid:
        st.error("Fix validation errors before saving.")
        return

    try:
        if edit_job:
            saved = service.update_job(edit_job.id, submitted_payload, actor=actor)
            st.success(f"Updated {saved.name}.")
        else:
            saved = service.create_job(submitted_payload, actor=actor)
            st.success(f"Created {saved.name}.")
        st.session_state["daemon_jobs_edit_id"] = saved.id
        st.session_state["daemon_jobs_form_seed"] = _payload_from_job(saved)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))


def _render_preview_tab(*, service: DaemonJobService, actor: str | None) -> None:
    """Render systemd service/timer previews for a selected job."""
    jobs = service.list_jobs()
    selected_job = _select_job(
        jobs,
        key="preview_job_select",
        preferred_id=st.session_state.get("daemon_jobs_preview_job_id"),
    )

    st.warning(
        "Step 1 only previews systemd units. It does not install anything on the Pi."
    )
    if selected_job is None:
        return

    if st.button("Generate Preview", type="primary"):
        preview = service.preview_systemd(selected_job.id, actor=actor)
        _store_preview(preview)
        st.session_state["daemon_jobs_preview_job_id"] = selected_job.id
        st.rerun()

    preview_payload = st.session_state.get("daemon_jobs_preview")
    if not preview_payload or preview_payload.get("job_id") != selected_job.id:
        st.info("Generate a preview to render service, timer, and command text.")
        return

    for warning in preview_payload.get("warnings", []):
        st.warning(warning)

    st.subheader(".service")
    st.code(preview_payload["service_unit"], language="ini")

    st.subheader(".timer")
    if preview_payload["timer_unit"]:
        st.code(preview_payload["timer_unit"], language="ini")
    else:
        st.info("No .timer unit is rendered for this schedule type.")

    st.subheader("Install Commands")
    st.code(preview_payload["install_commands"], language="bash")

    st.subheader("Uninstall Commands")
    st.code(preview_payload["uninstall_commands"], language="bash")


def _render_audit_log_tab(*, service: DaemonJobService) -> None:
    """Render audit events for active and soft-deleted jobs."""
    jobs = service.list_jobs(include_deleted=True)
    selected_job = _select_job(jobs, key="audit_job_select")
    if selected_job is None:
        return

    events = service.get_audit_events(selected_job.id)
    if not events:
        st.info("No audit events found for this job.")
        return

    rows = [
        {
            "created_at": _format_dt(event.created_at),
            "event_type": event.event_type,
            "actor": event.actor or "",
            "message": event.message,
        }
        for event in events
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    latest_snapshot = events[0].snapshot_json
    with st.expander("Latest Snapshot JSON"):
        if latest_snapshot:
            st.json(json.loads(latest_snapshot))
        else:
            st.info("No snapshot was stored for the latest event.")


def _render_help_tab() -> None:
    """Render the Daemon Jobs runbook."""
    st.subheader("Step 1 Scope")
    st.markdown(dedent("""
        This dashboard stores daemon job definitions and previews systemd unit text.
        It does not install files, change cron, run processes, or control systemd on
        the host or Raspberry Pi.

        Step 2 should add the Pi-side installer, runner, process controls, and any
        migration from cron expressions to systemd timers or runner scheduling.
        """).strip())

    st.subheader("Scheduling Modes")
    st.markdown(dedent("""
        - `systemd_timer`: stores a systemd `OnCalendar` value and renders a `.timer`.
        - `cron_expression`: stores a five-field cron expression for compatibility.
        - `manual_only`: stores a service definition with no schedule.
        """).strip())

    st.subheader("Unit Names")
    st.markdown(dedent("""
        `systemd_unit_name` is the safe unit stem. Use lowercase letters, digits,
        dashes, and underscores only. Do not include `.service` or `.timer`; the
        preview renderer adds those suffixes.
        """).strip())

    st.subheader("Recommended Production Settings")
    st.markdown(dedent("""
        - Keep jobs disabled until the Step 2 installer has reviewed and deployed them.
        - Use `forbid` concurrency for reports and operational watch jobs.
        - Use `on-failure` restart only for future manual long-running jobs.
        - Keep timeout and max runtime bounded to prevent stuck jobs.
        - Prefer systemd timers over cron expressions for Pi deployment.
        """).strip())

    st.subheader("Safety Rules")
    st.markdown(dedent("""
        - Jobs should be read-only by default.
        - Critical jobs require non-empty reporting rules.
        - Daemon jobs must not perform direct process-control writes.
        - Path fields must be absolute Pi-side paths without shell metacharacters.
        - ExecStart is rendered as fixed arguments: Python module plus `--job-id`.
        """).strip())


def _render_job_form(
    *,
    payload: dict[str, Any],
    is_edit: bool,
) -> tuple[dict[str, Any], bool, bool]:
    """Render the Streamlit form and return submitted payload/buttons."""
    suggested_unit = suggest_systemd_unit_name(str(payload.get("name") or ""))
    default_unit = str(payload.get("systemd_unit_name") or suggested_unit)

    with st.form("daemon_jobs_create_edit_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            name = st.text_input("Name", value=str(payload.get("name") or ""))
        with col2:
            enabled = st.toggle("Enabled", value=bool(payload.get("enabled", False)))
        with col3:
            criticality = st.selectbox(
                "Criticality",
                options=CRITICALITY_OPTIONS,
                index=_option_index(CRITICALITY_OPTIONS, payload.get("criticality")),
            )

        description = st.text_area(
            "Description",
            value=str(payload.get("description") or ""),
            height=90,
        )

        col4, col5, col6 = st.columns(3)
        with col4:
            job_kind = st.selectbox(
                "Job kind",
                options=JOB_KIND_OPTIONS,
                index=_option_index(JOB_KIND_OPTIONS, payload.get("job_kind")),
            )
        with col5:
            schedule_type = st.selectbox(
                "Schedule type",
                options=SCHEDULE_OPTIONS,
                index=_option_index(SCHEDULE_OPTIONS, payload.get("schedule_type")),
            )
        with col6:
            timezone = st.text_input(
                "Timezone",
                value=str(payload.get("timezone") or "Asia/Kolkata"),
            )

        col7, col8, col9 = st.columns(3)
        with col7:
            cron_expression = st.text_input(
                "Cron expression",
                value=str(payload.get("cron_expression") or ""),
                disabled=schedule_type != "cron_expression",
            )
        with col8:
            on_calendar = st.text_input(
                "OnCalendar",
                value=str(payload.get("on_calendar") or ""),
                disabled=schedule_type != "systemd_timer",
            )
        with col9:
            systemd_unit_name = st.text_input("systemd_unit_name", value=default_unit)

        if not is_edit and name:
            st.caption(f"Suggested unit name: {suggest_systemd_unit_name(name)}")

        col10, col11 = st.columns(2)
        with col10:
            working_directory = st.text_input(
                "Working directory",
                value=str(
                    payload.get("working_directory")
                    or DEFAULT_JOB_PAYLOAD["working_directory"]
                ),
            )
            python_executable = st.text_input(
                "Python executable",
                value=str(
                    payload.get("python_executable")
                    or DEFAULT_JOB_PAYLOAD["python_executable"]
                ),
            )
            module_path = st.text_input(
                "Module path",
                value=str(
                    payload.get("module_path") or DEFAULT_JOB_PAYLOAD["module_path"]
                ),
            )
            env_file = st.text_input(
                "Environment file",
                value=str(payload.get("env_file") or DEFAULT_JOB_PAYLOAD["env_file"]),
            )
            persist_jobs_md_path = st.text_input(
                "persist_jobs.md path",
                value=str(
                    payload.get("persist_jobs_md_path")
                    or DEFAULT_JOB_PAYLOAD["persist_jobs_md_path"]
                ),
            )
        with col11:
            user_name = st.text_input(
                "User", value=str(payload.get("user_name") or "pi")
            )
            group_name = st.text_input(
                "Group", value=str(payload.get("group_name") or "pi")
            )
            restart_policy = st.selectbox(
                "Restart policy",
                options=RESTART_OPTIONS,
                index=_option_index(RESTART_OPTIONS, payload.get("restart_policy")),
            )
            concurrency_policy = st.selectbox(
                "Concurrency policy",
                options=CONCURRENCY_OPTIONS,
                index=_option_index(
                    CONCURRENCY_OPTIONS, payload.get("concurrency_policy")
                ),
            )
            number_cols = st.columns(3)
            restart_sec = number_cols[0].number_input(
                "Restart sec",
                min_value=1,
                max_value=300,
                value=int(
                    payload.get("restart_sec") or DEFAULT_JOB_PAYLOAD["restart_sec"]
                ),
                step=1,
            )
            timeout_sec = number_cols[1].number_input(
                "Timeout sec",
                min_value=60,
                max_value=21600,
                value=int(
                    payload.get("timeout_sec") or DEFAULT_JOB_PAYLOAD["timeout_sec"]
                ),
                step=60,
            )
            max_runtime_sec = number_cols[2].number_input(
                "Max runtime sec",
                min_value=60,
                max_value=21600,
                value=int(
                    payload.get("max_runtime_sec")
                    or DEFAULT_JOB_PAYLOAD["max_runtime_sec"]
                ),
                step=60,
            )

        st.markdown("**JSON configuration**")
        json_cols = st.columns(2)
        json_values: dict[str, str] = {}
        for idx, field in enumerate(JSON_FIELDS):
            with json_cols[idx % 2]:
                json_values[field] = st.text_area(
                    field,
                    value=str(payload.get(field) or DEFAULT_JOB_PAYLOAD[field]),
                    height=120,
                )

        notes = st.text_area("Notes", value=str(payload.get("notes") or ""), height=90)

        button_cols = st.columns([1, 1, 4])
        validate_clicked = button_cols[0].form_submit_button("Validate")
        save_clicked = button_cols[1].form_submit_button("Save", type="primary")

    submitted = {
        "name": name,
        "description": description,
        "enabled": enabled,
        "job_kind": job_kind,
        "schedule_type": schedule_type,
        "cron_expression": (
            cron_expression if schedule_type == "cron_expression" else None
        ),
        "on_calendar": on_calendar if schedule_type == "systemd_timer" else None,
        "timezone": timezone,
        "systemd_unit_name": systemd_unit_name or suggest_systemd_unit_name(name),
        "working_directory": working_directory,
        "python_executable": python_executable,
        "module_path": module_path,
        "job_args_json": json_values["job_args_json"],
        "env_file": env_file,
        "user_name": user_name,
        "group_name": group_name,
        "restart_policy": restart_policy,
        "restart_sec": restart_sec,
        "timeout_sec": timeout_sec,
        "max_runtime_sec": max_runtime_sec,
        "concurrency_policy": concurrency_policy,
        "criticality": criticality,
        "tools_allowed_json": json_values["tools_allowed_json"],
        "tools_blocked_json": json_values["tools_blocked_json"],
        "memory_short_json": json_values["memory_short_json"],
        "memory_long_json": json_values["memory_long_json"],
        "reporting_rules_json": json_values["reporting_rules_json"],
        "criticality_rules_json": json_values["criticality_rules_json"],
        "persist_jobs_md_path": persist_jobs_md_path,
        "notes": notes,
    }
    return submitted, validate_clicked, save_clicked


def _render_validation_result(validation: Any) -> None:
    """Render validation result messages."""
    if validation.is_valid:
        st.success("Validation passed.")
    else:
        for error in validation.errors:
            st.error(error)
    for warning in validation.warnings:
        st.warning(warning)


def _select_job(
    jobs: list[DaemonJobView],
    *,
    key: str,
    preferred_id: str | None = None,
) -> DaemonJobView | None:
    """Render a selectbox for daemon jobs and return the selected job."""
    if not jobs:
        st.info("No jobs available.")
        return None

    id_to_job = {job.id: job for job in jobs}
    options = [job.id for job in jobs]
    default_index = 0
    if preferred_id in id_to_job:
        default_index = options.index(preferred_id)

    selected_id = st.selectbox(
        "Select job",
        options=options,
        index=default_index,
        key=key,
        format_func=lambda job_id: _job_option_label(id_to_job[job_id]),
    )
    return id_to_job[selected_id]


def _job_option_label(job: DaemonJobView) -> str:
    """Return display text for job selectors."""
    suffix = " deleted" if job.deleted else ""
    return f"{job.name} ({job.systemd_unit_name}){suffix}"


def _job_table_row(job: DaemonJobView) -> dict[str, Any]:
    """Return table row for the overview grid."""
    schedule = job.on_calendar or job.cron_expression or "manual"
    return {
        "name": job.name,
        "enabled": job.enabled,
        "job_kind": job.job_kind,
        "schedule_type": job.schedule_type,
        "schedule": schedule,
        "criticality": job.criticality,
        "systemd_unit_name": job.systemd_unit_name,
        "updated_at": _format_dt(job.updated_at),
    }


def _format_dt(value: Any) -> str:
    """Format datetimes for compact Streamlit tables."""
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _store_preview(preview: SystemdPreview) -> None:
    """Store preview output in session state."""
    st.session_state["daemon_jobs_preview"] = preview.model_dump()


def _payload_from_job(job: DaemonJobView | None) -> dict[str, Any]:
    """Return editable payload from a job view."""
    if job is None:
        return _default_payload()
    return {
        field: getattr(job, field)
        for field in DEFAULT_JOB_PAYLOAD
        if hasattr(job, field)
    }


def _default_payload() -> dict[str, Any]:
    """Return a fresh default daemon job payload."""
    return dict(DEFAULT_JOB_PAYLOAD)


def _current_seed_payload() -> dict[str, Any]:
    """Return current create-form seed payload."""
    seed = st.session_state.get("daemon_jobs_form_seed")
    if isinstance(seed, dict):
        return {**_default_payload(), **seed}
    return _default_payload()


def _preset_payload(preset_name: str) -> dict[str, Any]:
    """Return a payload seed for a named preset."""
    if preset_name == "Blank":
        return _default_payload()
    return {**_default_payload(), **PRESETS[preset_name]}


def _option_index(options: list[str], value: Any) -> int:
    """Return safe selectbox index for an option value."""
    try:
        return options.index(str(value))
    except ValueError:
        return 0


def _json_text(value: Any) -> str:
    """Render JSON defaults with stable indentation."""
    return json.dumps(value, indent=2, sort_keys=True)


PRESETS: dict[str, dict[str, Any]] = {
    "Shift Report": {
        "name": "FurnaceMind Shift Report",
        "description": "Generate shift handover intelligence from FurnaceMind context.",
        "job_kind": "furnacemind_shift_report",
        "schedule_type": "systemd_timer",
        "on_calendar": "Mon..Sun 07:00",
        "systemd_unit_name": "evonith-furnacemind-shift-report",
        "criticality": "high",
        "job_args_json": _json_text({"report_type": "shift", "window_hours": 8}),
        "tools_allowed_json": _json_text(["shift_report", "furnacemind_memory"]),
        "reporting_rules_json": _json_text(
            {"notify": ["shift_supervisor"], "persist": True}
        ),
    },
    "Heatload Watch": {
        "name": "Heatload Watch",
        "description": "Watch heatload signals and prepare summary context.",
        "job_kind": "heatload_watch",
        "schedule_type": "systemd_timer",
        "on_calendar": "*:0/15",
        "systemd_unit_name": "evonith-heatload-watch",
        "criticality": "normal",
        "job_args_json": _json_text({"watch": "heatload", "window_minutes": 60}),
        "tools_allowed_json": _json_text(["heatload_data", "furnacemind_memory"]),
    },
    "Channeling Watch": {
        "name": "Channeling Watch",
        "description": "Watch channeling indicators and preserve investigation context.",
        "job_kind": "channeling_watch",
        "schedule_type": "systemd_timer",
        "on_calendar": "*:0/15",
        "systemd_unit_name": "evonith-channeling-watch",
        "criticality": "high",
        "job_args_json": _json_text({"watch": "channeling", "window_minutes": 60}),
        "tools_allowed_json": _json_text(
            ["temperature_contours", "furnacemind_memory"]
        ),
        "reporting_rules_json": _json_text({"notify": ["operations"], "persist": True}),
    },
    "Daily Material Balance": {
        "name": "Daily Material Balance",
        "description": "Prepare daily material balance report context.",
        "job_kind": "material_balance_report",
        "schedule_type": "systemd_timer",
        "on_calendar": "daily",
        "systemd_unit_name": "evonith-daily-material-balance",
        "criticality": "normal",
        "job_args_json": _json_text({"report_type": "daily_material_balance"}),
        "tools_allowed_json": _json_text(["material_balance"]),
    },
    "Custom LangGraph Agent": {
        "name": "Custom LangGraph Agent",
        "description": "Custom agent job definition for future Step 2 runner execution.",
        "job_kind": "custom_langgraph_agent",
        "schedule_type": "manual_only",
        "on_calendar": None,
        "systemd_unit_name": "evonith-custom-langgraph-agent",
        "criticality": "normal",
        "job_args_json": _json_text({"graph": "custom"}),
    },
}
