"""Scheduled-job configuration and control page.

This page persists configuration and operational requests only. Report
execution and email delivery are worker concerns.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, time
from typing import Any

import streamlit as st
from jsonschema import ValidationError

from data.db import ScheduledJobService
from utils.scheduled_jobs import (
    FREQUENCIES,
    LOCAL_TIMEZONE,
    LOCAL_TIMEZONE_NAME,
    WEEKDAYS,
    archive_job,
    build_email_configuration,
    build_job_document,
    build_retry_configuration,
    build_schedule,
    clone_job_document,
    generate_job_id,
    pause_job,
    request_run_now,
    resume_job,
    revise_job,
    validate_job_document,
    validate_required_fields,
)
from utils.shift_windows import SHIFT_WINDOWS

log = logging.getLogger(__name__)
ENFORCE_UNIQUE_JOB_NAME = True
EDITOR_RESET_KEY = "scheduled_jobs_editor_reset_prefix"
EXISTING_SELECTOR_GENERATION_KEY = "scheduled_existing_selector_generation"
SERVICE_CACHE_VERSION = 2


@st.cache_resource(show_spinner=False)
def _get_cached_scheduled_job_service(
    cache_version: int,
) -> ScheduledJobService:
    """Return one relational service per Streamlit process."""
    del cache_version
    return ScheduledJobService()


def get_scheduled_job_service() -> ScheduledJobService:
    """Return a current service, replacing instances cached before code reloads."""
    service = _get_cached_scheduled_job_service(SERVICE_CACHE_VERSION)
    if service.__class__ is not ScheduledJobService:
        _get_cached_scheduled_job_service.clear()
        service = _get_cached_scheduled_job_service(SERVICE_CACHE_VERSION)
    return service


def _index(options: list[str] | tuple[str, ...], value: str, fallback: int = 0) -> int:
    try:
        return list(options).index(value)
    except ValueError:
        return fallback


def _json_text(job: dict[str, Any]) -> str:
    """Validate once more before making a job document downloadable."""
    validate_job_document(job)
    return json.dumps(job, indent=2, ensure_ascii=False)


def _render_schedule_inputs(
    prefix: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    frequency = st.selectbox(
        "Frequency",
        FREQUENCIES,
        index=_index(FREQUENCIES, str(defaults.get("frequency", "Every Day")), 1),
        key=f"{prefix}_frequency",
    )
    values: dict[str, Any] = {"frequency": frequency}
    if frequency == "Every Hour":
        values["execution_minute"] = int(
            st.number_input(
                "Execution minute",
                min_value=0,
                max_value=59,
                value=int(defaults.get("execution_minute", 0)),
                step=1,
                key=f"{prefix}_execution_minute",
            )
        )
    elif frequency == "Every Day":
        values["execution_time"] = st.time_input(
            "Execution time",
            value=_parse_time(defaults.get("execution_time"), time(6, 0)),
            key=f"{prefix}_daily_time",
        )
    elif frequency == "Every Shift":
        shift_labels = [
            f"{window['label']} ({int(window['start_hour']):02d}:00-"
            f"{int(window['end_hour']):02d}:00)"
            for window in SHIFT_WINDOWS
        ]
        st.info(
            "Uses configured production shifts: "
            + (", ".join(shift_labels) if shift_labels else "none configured")
        )
    elif frequency == "Every Week":
        left, right = st.columns(2)
        with left:
            values["weekday"] = st.selectbox(
                "Weekday",
                WEEKDAYS,
                index=_index(WEEKDAYS, str(defaults.get("weekday", "Monday"))),
                key=f"{prefix}_weekday",
            )
        with right:
            values["execution_time"] = st.time_input(
                "Execution time",
                value=_parse_time(defaults.get("execution_time"), time(6, 0)),
                key=f"{prefix}_weekly_time",
            )
    else:
        values["cron_expression"] = st.text_input(
            "Cron expression",
            value=str(defaults.get("cron", "0 6 * * 1-5")),
            placeholder="minute hour day-of-month month day-of-week",
            help="Standard five-field numeric cron, for example: 0 6 * * 1-5",
            key=f"{prefix}_cron",
        )
    return values


def _parse_time(value: Any, fallback: time) -> time:
    if isinstance(value, time):
        return value
    try:
        return datetime.strptime(str(value), "%H:%M").time()
    except (TypeError, ValueError):
        return fallback


def _render_email_inputs(
    prefix: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    enabled = st.checkbox(
        "Enable email delivery",
        value=bool(defaults.get("enabled", False)),
        key=f"{prefix}_email_enabled",
    )
    values: dict[str, Any] = {"enabled": enabled}
    if enabled:
        values["recipients"] = st.text_area(
            "Recipients",
            value=", ".join(defaults.get("recipients", [])),
            help="Separate multiple addresses with commas, semicolons, or new lines.",
            key=f"{prefix}_email_recipients",
        )
        values["subject"] = st.text_input(
            "Email subject",
            value=str(defaults.get("subject", "Scheduled report")),
            key=f"{prefix}_email_subject",
        )
        st.caption("Attachments")
        attachment_defaults = set(defaults.get("attachment_formats", []))
        attachment_columns = st.columns(3)
        attachments = []
        for column, extension in zip(
            attachment_columns, ("png", "csv", "json"), strict=True
        ):
            with column:
                if st.checkbox(
                    extension.upper(),
                    value=extension in attachment_defaults,
                    key=f"{prefix}_attachment_{extension}",
                ):
                    attachments.append(extension)
        values["attachment_formats"] = attachments
    else:
        values.update(recipients="", subject="", attachment_formats=[])
    return values


def _render_retry_inputs(
    prefix: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    columns = st.columns(3)
    with columns[0]:
        maximum_attempts = int(
            st.number_input(
                "Maximum attempts",
                min_value=1,
                value=int(defaults.get("maximum_attempts", 3)),
                step=1,
                key=f"{prefix}_maximum_attempts",
            )
        )
    with columns[1]:
        retry_interval = int(
            st.number_input(
                "Retry interval (seconds)",
                min_value=1,
                value=int(defaults.get("retry_interval_seconds", 60)),
                step=1,
                key=f"{prefix}_retry_interval",
            )
        )
    with columns[2]:
        timeout = int(
            st.number_input(
                "Timeout (seconds)",
                min_value=1,
                value=int(defaults.get("timeout_seconds", 300)),
                step=1,
                key=f"{prefix}_timeout",
            )
        )
    notify = st.checkbox(
        "Notify on failure",
        value=bool(defaults.get("notify_on_failure", True)),
        key=f"{prefix}_notify_failure",
    )
    return {
        "maximum_attempts": maximum_attempts,
        "retry_interval_seconds": retry_interval,
        "timeout_seconds": timeout,
        "notify_on_failure": notify,
    }


def _render_editor(
    prefix: str,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = defaults or {}
    job_name = st.text_input(
        "Job Name",
        value=str(defaults.get("job_name", "")),
        key=f"{prefix}_job_name",
    )
    instructions = st.text_area(
        "Job Instructions",
        value=str(defaults.get("instructions", "")),
        key=f"{prefix}_instructions",
    )

    st.markdown("#### Schedule")
    schedule_values = _render_schedule_inputs(
        prefix,
        dict(defaults.get("schedule", {})),
    )

    st.markdown("#### Email delivery")
    email_values = _render_email_inputs(prefix, dict(defaults.get("email", {})))

    st.markdown("#### Retry and timeout")
    retry_values = _render_retry_inputs(prefix, dict(defaults.get("retry", {})))

    return {
        "job_name": job_name,
        "instructions": instructions,
        "schedule_values": schedule_values,
        "email_values": email_values,
        "retry_values": retry_values,
    }


def _configuration_from_editor(values: dict[str, Any]) -> dict[str, Any]:
    errors = validate_required_fields(
        job_name=values["job_name"],
        instructions=values["instructions"],
        frequency=values["schedule_values"]["frequency"],
    )
    if errors:
        raise ValueError(errors[0])

    schedule = build_schedule(**values["schedule_values"])
    email = build_email_configuration(**values["email_values"])
    retry = build_retry_configuration(**values["retry_values"])
    return {
        "job_name": str(values["job_name"]).strip(),
        "instructions": str(values["instructions"]).strip(),
        "timezone": LOCAL_TIMEZONE_NAME,
        "schedule": schedule,
        "email": email,
        "retry": retry,
    }


def _new_identity(prefix: str) -> tuple[str, datetime]:
    id_key = f"{prefix}_document_id"
    time_key = f"{prefix}_created_at"
    if id_key not in st.session_state:
        st.session_state[id_key] = generate_job_id()
        st.session_state[time_key] = datetime.now(LOCAL_TIMEZONE).isoformat()
    return (
        str(st.session_state[id_key]),
        datetime.fromisoformat(str(st.session_state[time_key])),
    )


def _queue_editor_reset(prefix: str) -> None:
    """Clear this editor's widget state before widgets render on the next run."""
    st.session_state[EDITOR_RESET_KEY] = prefix


def _editor_prefix(base: str) -> str:
    generation = int(st.session_state.get(f"{base}_generation", 0))
    return f"{base}_{generation}"


def _rotate_editor(base: str) -> None:
    """Give the next editor fresh widget keys and retire its current state."""
    generation_key = f"{base}_generation"
    generation = int(st.session_state.get(generation_key, 0))
    _queue_editor_reset(f"{base}_{generation}")
    st.session_state[generation_key] = generation + 1


def _existing_selector_key() -> str:
    generation = int(st.session_state.get(EXISTING_SELECTOR_GENERATION_KEY, 0))
    return f"scheduled_existing_selector_{generation}"


def _rotate_existing_selector() -> None:
    generation = int(st.session_state.get(EXISTING_SELECTOR_GENERATION_KEY, 0))
    st.session_state[EXISTING_SELECTOR_GENERATION_KEY] = generation + 1


def _apply_queued_editor_reset() -> None:
    prefix = st.session_state.pop(EDITOR_RESET_KEY, None)
    if not prefix:
        return
    key_prefix = f"{prefix}_"
    for key in list(st.session_state):
        if str(key).startswith(key_prefix):
            del st.session_state[key]


def _set_creation_intent(key: str, status: str) -> None:
    """Set preview/persistence status before Streamlit's button rerun renders."""
    st.session_state[key] = status


def _candidate_from_editor(
    values: dict[str, Any],
    *,
    status: str,
    actor: str,
    identity_prefix: str,
) -> dict[str, Any]:
    configuration = _configuration_from_editor(values)
    configuration.pop("timezone", None)
    job_id, created_at = _new_identity(identity_prefix)
    return build_job_document(
        **configuration,
        status=status,
        created_by=actor,
        job_id=job_id,
        now=created_at,
    )


def _render_generated_json(job: dict[str, Any] | None, key: str) -> None:
    st.markdown("#### Generated JSON")
    if job is None:
        st.caption("Complete the required fields to preview and download the JSON.")
        return
    st.json(job)
    try:
        document = _json_text(job)
    except (ValueError, ValidationError) as exc:
        st.error(f"Generated JSON is not safe to download: {exc}")
        return
    st.download_button(
        "Download JSON",
        data=document,
        file_name=f"{job['job_id']}_rev_{job['revision']}.json",
        mime="application/json",
        key=key,
    )


def _duplicate_name_message(name: str) -> str:
    return f"A scheduled job named '{name}' already exists. Job Name must be unique."


def _save_new_job(
    service: ScheduledJobService,
    job: dict[str, Any],
    *,
    editor_base: str,
) -> None:
    try:
        if ENFORCE_UNIQUE_JOB_NAME and service.job_name_exists(job["job_name"]):
            st.error(_duplicate_name_message(job["job_name"]))
            return
        service.create_job(job)
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
        return
    except Exception:
        log.exception("Unable to create scheduled job %s", job.get("job_id"))
        st.error("Unable to save the scheduled job. Check the database connection.")
        return

    _rotate_editor(editor_base)
    st.session_state.pop("scheduled_clone_source", None)
    st.session_state["scheduled_jobs_flash"] = (
        f"Job {job['job_id']} created successfully | Revision {job['revision']}"
    )
    st.rerun()


def _render_create_tab(
    service: ScheduledJobService,
    actor: str,
) -> None:
    st.subheader("Create a scheduled job")
    st.caption("Create a schedule configuration. No report is executed here.")
    editor_base = "scheduled_create"
    prefix = _editor_prefix(editor_base)
    values = _render_editor(prefix)
    intent_key = f"{prefix}_intent"
    requested_status = st.session_state.get(intent_key)

    candidate = None
    try:
        candidate = _candidate_from_editor(
            values,
            status=str(requested_status or "draft"),
            actor=actor,
            identity_prefix=prefix,
        )
    except (ValueError, ValidationError):
        pass
    _render_generated_json(candidate, f"{prefix}_download")

    save_column, activate_column, _ = st.columns([1, 1, 3])
    save_column.button(
        "Save Draft",
        key=f"{prefix}_save_draft",
        on_click=_set_creation_intent,
        args=(intent_key, "draft"),
    )
    activate_column.button(
        "Create & Activate",
        type="primary",
        key=f"{prefix}_create_activate",
        on_click=_set_creation_intent,
        args=(intent_key, "active"),
    )
    if requested_status not in {"draft", "active"}:
        return

    st.session_state.pop(intent_key, None)
    try:
        job = candidate or _candidate_from_editor(
            values,
            status=str(requested_status),
            actor=actor,
            identity_prefix=prefix,
        )
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
        return
    _save_new_job(service, job, editor_base=editor_base)


def _table_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Job Name": job.get("job_name", ""),
            "Frequency": job.get("schedule", {}).get("frequency", ""),
            "Job Status": str(job.get("status", "")).title(),
            "Last Run": job.get("last_run") or "—",
            "Next Run": job.get("next_run") or "—",
            "Revision": job.get("revision", ""),
        }
        for job in jobs
    ]


def _show_history(title: str, entries: list[dict[str, Any]]) -> None:
    st.markdown(f"##### {title}")
    if entries:
        st.json(entries)
    else:
        st.caption("No history yet")


def _update_job(
    service: ScheduledJobService,
    job: dict[str, Any],
    message: str,
) -> None:
    try:
        service.update_job(job["job_id"], job)
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
        return
    except Exception:
        log.exception("Unable to update scheduled job %s", job.get("job_id"))
        st.error("Unable to update the scheduled job. Check the database connection.")
        return
    st.session_state["scheduled_jobs_flash"] = message
    st.rerun()


@st.dialog("Archive scheduled job")
def _confirm_archive(job_id: str, job_name: str, actor: str) -> None:
    st.warning(
        f"Archive '{job_name}'? Its configuration and histories will be preserved, "
        "but future scheduled execution will be disabled."
    )
    confirm, cancel = st.columns(2)
    if confirm.button("Confirm Archive", type="primary", key=f"confirm_{job_id}"):
        try:
            service = get_scheduled_job_service()
            current = service.get_job(job_id)
            if current is None:
                st.error("Scheduled job was not found.")
                return
            archived = archive_job(current, actor=actor)
            service.update_job(job_id, archived)
        except (ValueError, ValidationError) as exc:
            st.error(str(exc))
            return
        except Exception:
            log.exception("Unable to archive scheduled job %s", job_id)
            st.error(
                "Unable to archive the scheduled job. Check the database connection."
            )
            return
        st.session_state["scheduled_jobs_flash"] = f"Job {job_id} archived."
        st.rerun()
    if cancel.button("Cancel", key=f"cancel_{job_id}"):
        st.rerun()


@st.dialog("Delete scheduled job")
def _confirm_delete(job_id: str, job_name: str) -> None:
    st.error(
        f"Permanently delete '{job_name}'? Its configuration and all histories "
        "will be removed. This action cannot be undone."
    )
    confirm, cancel = st.columns(2)
    if confirm.button(
        "Confirm Delete",
        type="primary",
        key=f"confirm_delete_{job_id}",
    ):
        try:
            service = get_scheduled_job_service()
            service.delete_job(job_id)
        except KeyError:
            st.error("Scheduled job was not found. It may already be deleted.")
            return
        except Exception:
            log.exception("Unable to delete scheduled job %s", job_id)
            st.error(
                "Unable to delete the scheduled job. Check the database connection."
            )
            return
        if st.session_state.get("scheduled_clone_source") == job_id:
            st.session_state.pop("scheduled_clone_source", None)
        _rotate_existing_selector()
        st.session_state["scheduled_jobs_flash"] = f"Job {job_id} deleted."
        st.rerun()
    if cancel.button("Cancel", key=f"cancel_delete_{job_id}"):
        st.rerun()


def _render_actions(
    service: ScheduledJobService,
    job: dict[str, Any],
    actor: str,
) -> None:
    st.markdown("#### Job actions")
    (
        status_column,
        run_column,
        clone_column,
        archive_column,
        delete_column,
    ) = st.columns(5)
    if job["status"] == "archived":
        st.info("This job is archived. Configuration and history remain read-only.")
        if delete_column.button("Delete", key=f"delete_{job['job_id']}"):
            _confirm_delete(job["job_id"], job["job_name"])
        return

    if job["status"] == "active":
        if status_column.button("Pause", key=f"pause_{job['job_id']}"):
            updated = pause_job(job, actor=actor)
            _update_job(service, updated, f"Job {job['job_id']} paused.")
    else:
        label = "Activate" if job["status"] == "draft" else "Resume"
        if status_column.button(label, key=f"resume_{job['job_id']}"):
            updated = resume_job(job, actor=actor)
            _update_job(service, updated, f"Job {job['job_id']} is active.")

    if run_column.button("Run Now", key=f"run_{job['job_id']}"):
        updated = request_run_now(job, actor=actor)
        _update_job(
            service,
            updated,
            f"Manual execution request queued for job {job['job_id']}.",
        )
    if clone_column.button("Clone", key=f"clone_{job['job_id']}"):
        st.session_state["scheduled_clone_source"] = job["job_id"]
        _rotate_editor(f"scheduled_clone_{job['job_id']}")
        st.rerun()
    if archive_column.button("Archive", key=f"archive_{job['job_id']}"):
        _confirm_archive(job["job_id"], job["job_name"], actor)
    if delete_column.button("Delete", key=f"delete_{job['job_id']}"):
        _confirm_delete(job["job_id"], job["job_name"])


def _render_edit_panel(
    service: ScheduledJobService,
    job: dict[str, Any],
    actor: str,
) -> None:
    if job["status"] == "archived":
        return
    with st.expander("Edit configuration", expanded=False):
        values = _render_editor(f"scheduled_edit_{job['job_id']}", job)
        if not st.button("Save Configuration Changes", key=f"save_{job['job_id']}"):
            return
        try:
            updates = _configuration_from_editor(values)
            if ENFORCE_UNIQUE_JOB_NAME and service.job_name_exists(
                updates["job_name"], exclude_job_id=job["job_id"]
            ):
                st.error(_duplicate_name_message(updates["job_name"]))
                return
            revised = revise_job(job, updates, saved_by=actor)
            service.update_job(job["job_id"], revised)
        except (ValueError, ValidationError) as exc:
            st.error(str(exc))
            return
        except Exception:
            log.exception("Unable to revise scheduled job %s", job["job_id"])
            st.error(
                "Unable to update the scheduled job. Check the database connection."
            )
            return
        st.session_state["scheduled_jobs_flash"] = (
            f"Job {job['job_id']} updated to Revision {revised['revision']}"
        )
        st.rerun()


def _render_clone_panel(
    service: ScheduledJobService,
    source: dict[str, Any],
    actor: str,
) -> None:
    if st.session_state.get("scheduled_clone_source") != source["job_id"]:
        return
    st.divider()
    st.subheader("Clone configuration")
    st.caption("Review and edit this copy before saving it as a new Revision 1 job.")
    defaults = clone_job_document(source, created_by=actor)
    editor_base = f"scheduled_clone_{source['job_id']}"
    prefix = _editor_prefix(editor_base)
    values = _render_editor(prefix, defaults)
    intent_key = f"{prefix}_intent"
    requested_status = st.session_state.get(intent_key)

    draft = None
    try:
        draft = _candidate_from_editor(
            values,
            status=str(requested_status or "draft"),
            actor=actor,
            identity_prefix=prefix,
        )
    except (ValueError, ValidationError):
        pass
    _render_generated_json(draft, f"{prefix}_download")

    save_column, activate_column, cancel_column, _ = st.columns([1, 1, 1, 2])
    save_column.button(
        "Save Cloned Draft",
        key=f"{prefix}_save_draft",
        on_click=_set_creation_intent,
        args=(intent_key, "draft"),
    )
    activate_column.button(
        "Clone & Activate",
        type="primary",
        key=f"{prefix}_clone_activate",
        on_click=_set_creation_intent,
        args=(intent_key, "active"),
    )
    if cancel_column.button("Cancel", key=f"{prefix}_cancel"):
        st.session_state.pop("scheduled_clone_source", None)
        _rotate_editor(editor_base)
        st.rerun()
    if requested_status not in {"draft", "active"}:
        return
    st.session_state.pop(intent_key, None)
    try:
        cloned = draft or _candidate_from_editor(
            values,
            status=str(requested_status),
            actor=actor,
            identity_prefix=prefix,
        )
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
        return
    _save_new_job(service, cloned, editor_base=editor_base)


def _render_selected_job(
    service: ScheduledJobService,
    job: dict[str, Any],
    actor: str,
) -> None:
    st.divider()
    st.subheader(f"{job['job_name']} · {job['job_id']}")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Status", str(job["status"]).title())
    metric_columns[1].metric("Revision", job["revision"])
    metric_columns[2].metric("Timezone", job["timezone"])

    current_tab, history_tab, control_tab = st.tabs(
        ["Current Configuration", "Histories", "Edit & Actions"]
    )
    with current_tab:
        st.markdown("##### Schedule")
        st.json(job["schedule"])
        _render_generated_json(job, f"download_current_{job['job_id']}")
    with history_tab:
        _show_history("Revision history", job["revision_history"])
        _show_history("Run history", job["run_history"])
        _show_history("Email delivery history", job["email_history"])
        _show_history("Generated report/graph metadata", job["report_history"])
        _show_history("Errors", job["error_history"])
        _show_history("Execution requests", job["execution_requests"])
        _show_history("Action history", job["action_history"])
    with control_tab:
        _render_actions(service, job, actor)
        _render_edit_panel(service, job, actor)

    _render_clone_panel(service, job, actor)


def _render_existing_tab(
    service: ScheduledJobService,
    actor: str,
) -> None:
    st.subheader("Existing scheduled jobs")
    try:
        jobs = service.list_jobs()
    except Exception:
        log.exception("Unable to list scheduled jobs")
        st.error("Unable to load scheduled jobs. Check the database connection.")
        return
    if not jobs:
        st.info("No scheduled jobs found.")
        return

    st.dataframe(_table_rows(jobs), hide_index=True, width="stretch")
    by_id = {job["job_id"]: job for job in jobs}
    selected_id = st.selectbox(
        "Select a job",
        list(by_id),
        format_func=lambda job_id: f"{by_id[job_id]['job_name']} | {job_id}",
        key=_existing_selector_key(),
    )
    _render_selected_job(service, by_id[selected_id], actor)


def main() -> None:
    """Render the authenticated Scheduled Jobs page."""
    if "auth_user" not in st.session_state:
        st.warning("Please login to access this page.")
        st.stop()

    _apply_queued_editor_reset()
    st.title("⏱️ Scheduled Jobs")
    st.caption(
        "Configure reports, schedules, and worker requests. "
        "Execution happens outside Streamlit."
    )
    flash = st.session_state.pop("scheduled_jobs_flash", None)
    if flash:
        st.success(flash)

    try:
        service = get_scheduled_job_service()
    except Exception:
        log.exception("Unable to initialize scheduled jobs")
        st.error("Unable to initialize Scheduled Jobs. Check the database connection.")
        return

    actor = str(st.session_state.get("auth_user") or "unknown")
    create_tab, existing_tab = st.tabs(["Create Job", "Existing Jobs"])
    with create_tab:
        _render_create_tab(service, actor)
    with existing_tab:
        _render_existing_tab(service, actor)


main()
