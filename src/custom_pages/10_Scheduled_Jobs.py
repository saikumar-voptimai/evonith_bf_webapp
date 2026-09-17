"""Streamlit editor for scheduled-job configuration and control requests.

The page persists validated configuration and worker requests; it never runs a
report or contacts a delivery provider directly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, time
from typing import Any

import streamlit as st
from jsonschema import ValidationError

from data.db import ScheduledJobService
from utils.scheduled_job_access import (
    ScheduledJobAuthorizationError,
    ScheduledJobPrincipal,
    can_create_job,
    can_delete_job,
    can_update_job,
    can_view_job,
)
from utils.scheduled_jobs import (
    DEFAULT_FREQUENCY,
    DEFAULT_MODEL_LEVEL,
    DELIVERY_CHANNEL_LABELS,
    DELIVERY_CHANNEL_SETTINGS,
    DELIVERY_CHANNELS,
    FREQUENCIES,
    FREQUENCY_SETTINGS,
    LOCAL_TIMEZONE,
    LOCAL_TIMEZONE_NAME,
    MODEL_LEVEL_HINTS,
    MODEL_LEVELS,
    RETRY_FIELDS,
    WEEKDAYS,
    archive_job,
    build_delivery_configuration,
    build_job_document,
    build_retry_configuration,
    build_schedule,
    clone_job_document,
    pause_job,
    request_run_now,
    resume_job,
    revise_job,
    validate_job_document,
    validate_required_fields,
)
from utils.session import current_scheduled_job_principal
from utils.shift_windows import SHIFT_WINDOWS

log = logging.getLogger(__name__)
ENFORCE_UNIQUE_JOB_NAME = True
EDITOR_RESET_KEY = "scheduled_jobs_editor_reset_prefix"
SELECTOR_GENERATION_KEY = "scheduled_existing_selector_generation"
SERVICE_CACHE_VERSION = 7
HISTORY_LABELS = {
    "revision_history": "Revision history",
    "run_history": "Run history",
    "delivery_history": "Delivery history",
    "report_history": "Generated report/graph metadata",
    "error_history": "Errors",
    "execution_requests": "Execution requests",
    "action_history": "Action history",
}


@st.cache_resource(show_spinner=False)
def _get_cached_service(
    cache_version: int, principal: ScheduledJobPrincipal
) -> ScheduledJobService:
    """Return one relational scheduled-job service per Streamlit process."""
    del cache_version
    return ScheduledJobService(principal=principal)


def get_scheduled_job_service() -> ScheduledJobService:
    """Return a current service and discard stale instances after code reloads."""
    authenticated = current_scheduled_job_principal()
    service = _get_cached_service(SERVICE_CACHE_VERSION, authenticated)
    if service.__class__ is not ScheduledJobService:
        _get_cached_service.clear()
        service = _get_cached_service(SERVICE_CACHE_VERSION, authenticated)
    return service


def _index(options: tuple[str, ...], value: str, fallback: int = 0) -> int:
    """Return an option index, falling back when stored data is outdated."""
    try:
        return options.index(value)
    except ValueError:
        return fallback


def _parse_time(value: Any, fallback: str) -> time:
    """Convert stored HH:MM text to a Streamlit-compatible time value."""
    if isinstance(value, time):
        return value
    try:
        return datetime.strptime(str(value or fallback), "%H:%M").time()
    except ValueError:
        return datetime.strptime(fallback, "%H:%M").time()


def _json_text(job: dict[str, Any]) -> str:
    """Validate and serialize a safe downloadable job document."""
    validate_job_document(job)
    return json.dumps(job, indent=2, ensure_ascii=False)


def _render_schedule_inputs(prefix: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Render schedule controls using the frequency metadata from YAML."""
    frequency_column, detail_column = st.columns(2)
    frequency = frequency_column.selectbox(
        "Frequency",
        FREQUENCIES,
        index=_index(FREQUENCIES, str(defaults.get("frequency", DEFAULT_FREQUENCY))),
        key=f"{prefix}_frequency",
    )
    settings, values = FREQUENCY_SETTINGS[frequency], {"frequency": frequency}
    kind = settings["kind"]
    if kind == "hourly":
        values["execution_minute"] = int(
            detail_column.number_input(
                "Execution minute",
                min_value=0,
                max_value=59,
                value=int(
                    defaults.get("execution_minute", settings["execution_minute"])
                ),
                key=f"{prefix}_execution_minute",
            )
        )
    elif kind == "daily":
        values["execution_time"] = detail_column.time_input(
            "Execution time",
            value=_parse_time(
                defaults.get("execution_time"), settings["execution_time"]
            ),
            key=f"{prefix}_daily_time",
        )
    elif kind == "shift":
        labels = [
            f"{item['label']} ({int(item['start_hour']):02d}:00-"
            f"{int(item['end_hour']):02d}:00)"
            for item in SHIFT_WINDOWS
        ]
        st.info(
            f"Uses configured production shifts: {', '.join(labels) or 'none configured'}"
        )
    elif kind == "weekly":
        values["weekday"] = detail_column.selectbox(
            "Weekday",
            WEEKDAYS,
            index=_index(WEEKDAYS, str(defaults.get("weekday", settings["weekday"]))),
            key=f"{prefix}_weekday",
        )
        values["execution_time"] = detail_column.time_input(
            "Execution time",
            value=_parse_time(
                defaults.get("execution_time"), settings["execution_time"]
            ),
            key=f"{prefix}_weekly_time",
        )
    else:
        values["cron_expression"] = detail_column.text_input(
            "Cron expression",
            value=str(defaults.get("cron", settings["cron"])),
            placeholder="minute hour day-of-month month day-of-week",
            help=settings.get("help"),
            key=f"{prefix}_cron",
        )
    return values


def _field_default(
    defaults: dict[str, Any], name: str, settings: dict[str, Any]
) -> Any:
    """Resolve a current field value, including configured legacy aliases."""
    for candidate in (name, *settings.get("aliases", [])):
        if defaults.get(candidate) not in (None, ""):
            return defaults[candidate]
    return settings.get("default", [] if settings["input"] == "multiselect" else "")


def _render_delivery_inputs(prefix: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Render all delivery channels and fields from their YAML definitions."""
    channels = st.multiselect(
        "Delivery channels",
        DELIVERY_CHANNELS,
        default=[name for name in DELIVERY_CHANNELS if name in defaults],
        format_func=lambda name: DELIVERY_CHANNEL_LABELS[str(name)],
        key=f"{prefix}_delivery_channels",
    )
    values: dict[str, Any] = {}
    for channel in channels:
        st.markdown(f"##### {DELIVERY_CHANNEL_LABELS[channel]}")
        channel_defaults = dict(defaults.get(channel, {}))
        fields = DELIVERY_CHANNEL_SETTINGS[channel]["fields"]
        columns = st.columns(min(len(fields), 3))
        values[channel] = {}
        for index, (name, settings) in enumerate(fields.items()):
            column, default = columns[index % len(columns)], _field_default(
                channel_defaults, name, settings
            )
            common = {
                "label": settings["label"],
                "key": f"{prefix}_{channel}_{name}",
            }
            if settings["input"] == "multiselect":
                value = column.multiselect(
                    options=settings["options"],
                    default=list(default),
                    format_func=lambda option: str(option).upper(),
                    **common,
                )
            else:
                if settings["input"] == "recipients" and not isinstance(default, str):
                    default = ", ".join(default)
                value = column.text_input(
                    value=str(default),
                    placeholder=settings.get("placeholder"),
                    help=settings.get("help"),
                    **common,
                )
            values[channel][name] = value
    if not channels:
        st.caption("No delivery channel selected.")
    return values


def _render_retry_inputs(prefix: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Render retry fields from YAML and return values accepted by the builder."""
    values: dict[str, Any] = {}
    number_fields = [
        item for item in RETRY_FIELDS.items() if item[1]["input"] == "number"
    ]
    columns = st.columns(len(number_fields))
    for column, (name, settings) in zip(columns, number_fields, strict=True):
        values[name] = int(
            column.number_input(
                settings["label"],
                min_value=int(settings["minimum"]),
                value=int(defaults.get(name, settings["default"])),
                key=f"{prefix}_{name}",
            )
        )
    for name, settings in RETRY_FIELDS.items():
        if settings["input"] == "checkbox":
            values[name] = st.checkbox(
                settings["label"],
                value=bool(defaults.get(name, settings["default"])),
                key=f"{prefix}_{name}",
            )
    return values


def _render_editor(
    prefix: str, defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Render the complete reusable create, edit, or clone form."""
    defaults = defaults or {}
    name_column, level_column = st.columns([2, 1])
    job_name = name_column.text_input(
        "Job Name", value=str(defaults.get("job_name", "")), key=f"{prefix}_job_name"
    )
    default_level = str(defaults.get("model_level", DEFAULT_MODEL_LEVEL)).lower()
    model_level = level_column.selectbox(
        "Model Level",
        MODEL_LEVELS,
        index=_index(MODEL_LEVELS, default_level),
        format_func=lambda level: str(level).title(),
        help="Controls the reasoning depth used by the scheduled-job worker.",
        key=f"{prefix}_model_level",
    )
    level_column.caption(MODEL_LEVEL_HINTS[model_level])
    instructions = st.text_area(
        "Job Instructions",
        value=str(defaults.get("instructions", "")),
        key=f"{prefix}_instructions",
    )
    st.markdown("#### Schedule")
    schedule = _render_schedule_inputs(prefix, dict(defaults.get("schedule", {})))
    st.markdown("#### Delivery")
    delivery = _render_delivery_inputs(prefix, dict(defaults.get("delivery", {})))
    st.markdown("#### Retry and timeout")
    retry = _render_retry_inputs(prefix, dict(defaults.get("retry", {})))
    return {
        "job_name": job_name,
        "instructions": instructions,
        "model_level": model_level,
        "schedule_values": schedule,
        "delivery_values": delivery,
        "retry_values": retry,
    }


def _configuration_from_editor(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate values returned by the reusable editor."""
    errors = validate_required_fields(
        job_name=values["job_name"],
        instructions=values["instructions"],
        frequency=values["schedule_values"]["frequency"],
    )
    if errors:
        raise ValueError(errors[0])
    return {
        "job_name": str(values["job_name"]).strip(),
        "instructions": str(values["instructions"]).strip(),
        "model_level": values["model_level"],
        "timezone": LOCAL_TIMEZONE_NAME,
        "schedule": build_schedule(**values["schedule_values"]),
        "delivery": build_delivery_configuration(values["delivery_values"]),
        "retry": build_retry_configuration(**values["retry_values"]),
    }


def _editor_prefix(base: str) -> str:
    """Return a generation-specific widget prefix for an editor."""
    return f"{base}_{int(st.session_state.get(f'{base}_generation', 0))}"


def _rotate_editor(base: str) -> None:
    """Retire an editor's widget keys so its next render starts cleanly."""
    generation_key = f"{base}_generation"
    generation = int(st.session_state.get(generation_key, 0))
    st.session_state[EDITOR_RESET_KEY] = f"{base}_{generation}"
    st.session_state[generation_key] = generation + 1


def _apply_editor_reset() -> None:
    """Remove retired widget values before Streamlit creates replacement widgets."""
    if prefix := st.session_state.pop(EDITOR_RESET_KEY, None):
        for key in [
            key for key in st.session_state if str(key).startswith(f"{prefix}_")
        ]:
            del st.session_state[key]


def _selector_key() -> str:
    """Return a generation-specific existing-job selector key."""
    generation = int(st.session_state.get(SELECTOR_GENERATION_KEY, 0))
    return f"scheduled_existing_selector_{generation}"


def _new_identity(prefix: str, service: ScheduledJobService) -> tuple[str, datetime]:
    """Reserve and retain one job identity for the current editor generation."""
    id_key, time_key = f"{prefix}_document_id", f"{prefix}_created_at"
    if id_key not in st.session_state:
        st.session_state[id_key] = service.reserve_job_id()
        st.session_state[time_key] = datetime.now(LOCAL_TIMEZONE).isoformat()
    return str(st.session_state[id_key]), datetime.fromisoformat(
        st.session_state[time_key]
    )


def _candidate_from_editor(
    values: dict[str, Any],
    *,
    service: ScheduledJobService,
    status: str,
    actor: str,
    prefix: str,
) -> dict[str, Any]:
    """Build a complete job candidate from editor values and a reserved identity."""
    configuration = _configuration_from_editor(values)
    configuration.pop("timezone")
    job_id, created_at = _new_identity(prefix, service)
    return build_job_document(
        **configuration,
        status=status,
        created_by=actor,
        job_id=job_id,
        now=created_at,
    )


def _render_json(job: dict[str, Any] | None, key: str) -> None:
    """Show a validated JSON preview and download button when data is complete."""
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
        document,
        file_name=f"{job['job_id']}_rev_{job['revision']}.json",
        mime="application/json",
        key=key,
    )


def _duplicate_name_message(name: str) -> str:
    """Return the consistent uniqueness validation message."""
    return f"A scheduled job named '{name}' already exists. Job Name must be unique."


def _save_new_job(
    service: ScheduledJobService,
    job: dict[str, Any],
    *,
    editor_base: str,
    clone_source_job_id: str | None = None,
) -> None:
    """Persist a new job, refresh editor state, and show a success flash."""
    try:
        if ENFORCE_UNIQUE_JOB_NAME and service.job_name_exists(job["job_name"]):
            st.error(_duplicate_name_message(job["job_name"]))
            return
        if clone_source_job_id:
            service.create_cloned_job(clone_source_job_id, job)
        else:
            service.create_job(job)
    except (ValueError, ValidationError, ScheduledJobAuthorizationError) as exc:
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


def _render_create_tab(service: ScheduledJobService, actor: str) -> None:
    """Render and process the new-job editor."""
    st.subheader("Create a scheduled job")
    st.caption("Create a schedule configuration. No report is executed here.")
    editor_base, prefix = "scheduled_create", _editor_prefix("scheduled_create")
    values = _render_editor(prefix)
    draft_column, active_column, _ = st.columns([1, 1, 3])
    status = (
        "draft"
        if draft_column.button("Save Draft", key=f"{prefix}_save_draft")
        else (
            "active"
            if active_column.button(
                "Create & Activate", type="primary", key=f"{prefix}_create_activate"
            )
            else None
        )
    )
    if status is None:
        return
    try:
        job = _candidate_from_editor(
            values, service=service, status=status, actor=actor, prefix=prefix
        )
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
        return
    _save_new_job(service, job, editor_base=editor_base)


def _table_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert jobs to concise rows for the existing-jobs table."""
    return [
        {
            "Job Name": job.get("job_name", ""),
            "Model Level": str(job.get("model_level", DEFAULT_MODEL_LEVEL)).title(),
            "Frequency": job.get("schedule", {}).get("frequency", ""),
            "Created By": job.get("created_by", ""),
            "Delivery": ", ".join(
                DELIVERY_CHANNEL_LABELS.get(name, name.title())
                for name in job.get("delivery", {})
            )
            or "None",
            "Job Status": str(job.get("status", "")).title(),
            "Last Run": job.get("last_run") or "—",
            "Next Run": job.get("next_run") or "—",
            "Revision": job.get("revision", ""),
        }
        for job in jobs
    ]


def _update_job(
    service: ScheduledJobService, job: dict[str, Any], message: str
) -> None:
    """Persist an updated job and rerun with a success flash."""
    try:
        service.update_job(job["job_id"], job)
    except (ValueError, ValidationError, ScheduledJobAuthorizationError) as exc:
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
    """Confirm and perform a non-destructive archive transition."""
    st.warning(
        f"Archive '{job_name}'? Its configuration and histories will be preserved, "
        "but future scheduled execution will be disabled."
    )
    confirm, cancel = st.columns(2)
    if confirm.button("Confirm Archive", type="primary", key=f"confirm_{job_id}"):
        try:
            service, current = get_scheduled_job_service(), None
            current = service.get_job(job_id)
            if current is None:
                st.error("Scheduled job was not found.")
                return
            archived = archive_job(current, actor=actor)
        except (ValueError, ValidationError, ScheduledJobAuthorizationError) as exc:
            st.error(str(exc))
            return
        except Exception:
            log.exception("Unable to archive scheduled job %s", job_id)
            st.error(
                "Unable to archive the scheduled job. Check the database connection."
            )
            return
        _update_job(service, archived, f"Job {job_id} archived.")
    if cancel.button("Cancel", key=f"cancel_{job_id}"):
        st.rerun()


@st.dialog("Delete scheduled job")
def _confirm_delete(job_id: str, job_name: str) -> None:
    """Confirm and permanently delete one job and all of its histories."""
    st.error(
        f"Permanently delete '{job_name}'? Its configuration and all histories "
        "will be removed. This action cannot be undone."
    )
    confirm, cancel = st.columns(2)
    if confirm.button("Confirm Delete", type="primary", key=f"confirm_delete_{job_id}"):
        try:
            get_scheduled_job_service().delete_job(job_id)
        except ScheduledJobAuthorizationError as exc:
            st.error(str(exc))
            return
        except KeyError:
            st.error("Scheduled job was not found. It may already be deleted.")
            return
        except Exception:
            log.exception("Unable to delete scheduled job %s", job_id)
            st.error(
                "Unable to delete the scheduled job. Check the database connection."
            )
            return
        st.session_state.pop("scheduled_clone_source", None)
        st.session_state[SELECTOR_GENERATION_KEY] = (
            int(st.session_state.get(SELECTOR_GENERATION_KEY, 0)) + 1
        )
        st.session_state["scheduled_jobs_flash"] = f"Job {job_id} deleted."
        st.rerun()
    if cancel.button("Cancel", key=f"cancel_delete_{job_id}"):
        st.rerun()


def _apply_action(
    service: ScheduledJobService,
    job: dict[str, Any],
    actor: str,
    operation: Callable[..., dict[str, Any]],
    message: str,
) -> None:
    """Apply a pure job operation and persist its validated result."""
    try:
        updated = operation(job, actor=actor)
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
        return
    _update_job(service, updated, message)


def _render_actions(
    service: ScheduledJobService,
    job: dict[str, Any],
    principal: ScheduledJobPrincipal,
) -> None:
    """Render status, run, clone, archive, and delete controls for one job."""
    st.markdown("#### Job actions")
    may_update = can_update_job(principal, job)
    may_delete = can_delete_job(principal, job)
    if not (may_update or may_delete):
        st.info("Read-only: this job is owned by another account.")
        return

    job_id, status = job["job_id"], job["status"]
    if status == "archived":
        st.info("This job is archived. Configuration and history remain read-only.")
        columns = st.columns(2)
        if may_update and columns[0].button("Clone", key=f"clone_{job_id}"):
            st.session_state["scheduled_clone_source"] = job_id
            _rotate_editor(f"scheduled_clone_{job_id}")
            st.rerun()
        if may_delete and columns[1].button("Delete", key=f"delete_{job_id}"):
            _confirm_delete(job_id, job["job_name"])
        return

    columns = st.columns(5 if may_delete else 4)
    status_column, run_column, clone_column, archive_column = columns[:4]
    label, operation = (
        ("Pause", pause_job)
        if status == "active"
        else (
            "Activate" if status == "draft" else "Resume",
            resume_job,
        )
    )
    if status_column.button(label, key=f"status_{job_id}"):
        message = (
            f"Job {job_id} paused."
            if status == "active"
            else f"Job {job_id} is active."
        )
        _apply_action(service, job, principal.username, operation, message)
    if run_column.button("Run Now", key=f"run_{job_id}"):
        _apply_action(
            service,
            job,
            principal.username,
            request_run_now,
            f"Manual execution request queued for job {job_id}.",
        )
    if clone_column.button("Clone", key=f"clone_{job_id}"):
        st.session_state["scheduled_clone_source"] = job_id
        _rotate_editor(f"scheduled_clone_{job_id}")
        st.rerun()
    if archive_column.button("Archive", key=f"archive_{job_id}"):
        _confirm_archive(job_id, job["job_name"], principal.username)
    if may_delete and columns[4].button("Delete", key=f"delete_{job_id}"):
        _confirm_delete(job_id, job["job_name"])


def _render_edit_panel(
    service: ScheduledJobService,
    job: dict[str, Any],
    principal: ScheduledJobPrincipal,
) -> None:
    """Render and save revisioned configuration changes for a non-archived job."""
    if job["status"] == "archived" or not can_update_job(principal, job):
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
            revised = revise_job(job, updates, saved_by=principal.username)
        except (ValueError, ValidationError, ScheduledJobAuthorizationError) as exc:
            st.error(str(exc))
            return
        except Exception:
            log.exception("Unable to revise scheduled job %s", job["job_id"])
            st.error(
                "Unable to update the scheduled job. Check the database connection."
            )
            return
        _update_job(
            service,
            revised,
            f"Job {job['job_id']} updated to Revision {revised['revision']}",
        )


def _render_clone_panel(
    service: ScheduledJobService,
    source: dict[str, Any],
    principal: ScheduledJobPrincipal,
) -> None:
    """Render a prefilled editor that creates an independent copy of a job."""
    if st.session_state.get("scheduled_clone_source") != source[
        "job_id"
    ] or not can_update_job(principal, source):
        return
    st.divider()
    st.subheader("Clone configuration")
    st.caption("Review and edit this copy before saving it as a new Revision 1 job.")
    editor_base = f"scheduled_clone_{source['job_id']}"
    prefix = _editor_prefix(editor_base)
    job_id, created_at = _new_identity(prefix, service)
    defaults = clone_job_document(
        source, created_by=principal.username, job_id=job_id, now=created_at
    )
    values = _render_editor(prefix, defaults)
    try:
        preview = _candidate_from_editor(
            values,
            service=service,
            status="draft",
            actor=principal.username,
            prefix=prefix,
        )
    except (ValueError, ValidationError):
        preview = None
    _render_json(preview, f"{prefix}_download")

    draft_column, active_column, cancel_column, _ = st.columns([1, 1, 1, 2])
    save_draft = draft_column.button("Save Cloned Draft", key=f"{prefix}_save_draft")
    create_active = active_column.button(
        "Clone & Activate", type="primary", key=f"{prefix}_clone_activate"
    )
    if cancel_column.button("Cancel", key=f"{prefix}_cancel"):
        st.session_state.pop("scheduled_clone_source", None)
        _rotate_editor(editor_base)
        st.rerun()
    status = "draft" if save_draft else "active" if create_active else None
    if status is None:
        return
    try:
        cloned = _candidate_from_editor(
            values,
            service=service,
            status=status,
            actor=principal.username,
            prefix=prefix,
        )
    except (ValueError, ValidationError) as exc:
        st.error(str(exc))
        return
    _save_new_job(
        service,
        cloned,
        editor_base=editor_base,
        clone_source_job_id=source["job_id"],
    )


def _render_selected_job(
    service: ScheduledJobService,
    job: dict[str, Any],
    principal: ScheduledJobPrincipal,
) -> None:
    """Render details, histories, editing, and actions for the selected job."""
    st.divider()
    st.subheader(f"{job['job_name']} · {job['job_id']}")
    metrics = st.columns(4)
    for column, (label, value) in zip(
        metrics,
        (
            ("Status", str(job["status"]).title()),
            ("Revision", job["revision"]),
            ("Model Level", str(job["model_level"]).title()),
            ("Timezone", job["timezone"]),
        ),
        strict=True,
    ):
        column.metric(label, value)

    can_manage = can_update_job(principal, job)
    show_controls = can_manage or can_delete_job(principal, job)
    tabs = st.tabs(
        ["Current Configuration", "Histories"]
        + (["Edit & Actions"] if show_controls else [])
    )
    current_tab, history_tab = tabs[:2]
    with current_tab:
        for title, field in (("Schedule", "schedule"), ("Delivery", "delivery")):
            st.markdown(f"##### {title}")
            st.json(job[field])
        _render_json(job, f"download_current_{job['job_id']}")
    with history_tab:
        for field, title in HISTORY_LABELS.items():
            st.markdown(f"##### {title}")
            st.json(job[field]) if job[field] else st.caption("No history yet")
    if show_controls:
        with tabs[2]:
            _render_actions(service, job, principal)
            _render_edit_panel(service, job, principal)
    else:
        st.caption("Read-only access")
    if can_manage:
        _render_clone_panel(service, job, principal)


def _render_existing_tab(
    service: ScheduledJobService, principal: ScheduledJobPrincipal
) -> None:
    """List persisted jobs and render the selected job's management panels."""
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
        tuple(by_id),
        format_func=lambda job_id: f"{by_id[job_id]['job_name']} | {job_id}",
        key=_selector_key(),
    )
    _render_selected_job(service, by_id[selected_id], principal)


def main() -> None:
    """Render the authenticated Scheduled Jobs page."""
    if "auth_user" not in st.session_state:
        st.warning("Please login to access this page.")
        st.stop()
    _apply_editor_reset()
    st.title("⏱️ Scheduled Jobs")
    st.caption(
        "Configure reports, schedules, and worker requests. "
        "Execution happens outside Streamlit."
    )
    if flash := st.session_state.pop("scheduled_jobs_flash", None):
        st.success(flash)
    principal = current_scheduled_job_principal()
    if not can_view_job(principal):
        st.error("You are not authorized to view scheduled jobs.")
        return
    try:
        service = get_scheduled_job_service()
    except Exception:
        log.exception("Unable to initialize scheduled jobs")
        st.error("Unable to initialize Scheduled Jobs. Check the database connection.")
        return
    if can_create_job(principal):
        create_tab, existing_tab = st.tabs(["Create Job", "Existing Jobs"])
        with create_tab:
            _render_create_tab(service, principal.username)
        with existing_tab:
            _render_existing_tab(service, principal)
    else:
        st.caption("Read-only access")
        _render_existing_tab(service, principal)


main()
