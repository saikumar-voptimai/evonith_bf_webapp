"""Create and manage operator-owned scheduled-task JSON definitions.

The Streamlit page validates, stores, edits, clones, archives, deletes, and
exports definitions. Future execution controls remain visible but disabled;
this milestone never manipulates systemd, a Jetson, or FurnaceMind.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st

from data.scheduled_tasks import (
    ScheduledJobConflictError,
    ScheduledJobCreateRequest,
    ScheduledJobListItem,
    ScheduledJobPersistenceError,
    ScheduledJobService,
    ScheduledJobValidationError,
    ScheduledJobView,
)
from ui.styles import apply_styles
from utils.scheduled_tasks.scheduled_job_catalog import (
    ScheduledJobCatalog,
    get_scheduled_job_catalog,
)
from utils.scheduled_tasks.scheduled_task_definition import (
    ANALYSIS_LEVEL_LABELS,
    DATA_PERIOD_LABELS,
    DELIVERY_CHANNEL_LABELS,
    EMAIL_ATTACHMENT_LABELS,
    OUTPUT_FORMAT_LABELS,
    SCHEDULE_KIND_LABELS,
    WEEKDAY_NAMES,
    ScheduledTaskInput,
    ScheduledTaskValidationError,
    build_task_definition,
    describe_schedule,
    fixed_data_period_for_job_type,
    parse_email_recipients,
    task_definition_filename,
    task_definition_json,
    task_input_fingerprint,
    upcoming_run_times,
    validate_task_input,
)
from utils.session import current_user_id
from utils.shift_windows import SHIFT_WINDOWS, shift_windows_description

_GENERATED_KEY = "scheduled_task_generated_definition"
_ATTEMPTED_KEY = "scheduled_task_create_attempted"
_BUILD_ERROR_KEY = "scheduled_task_build_error"
_CLEAR_PENDING_KEY = "scheduled_task_clear_pending"
_TEMPLATE_KEY = "scheduled_task_template"
_STATE_OWNER_KEY = "scheduled_task_state_owner"
_VIEW_KEY = "scheduled_task_page_view"
_SELECTED_JOB_KEY = "scheduled_task_selected_job"
_ARCHIVE_CONFIRM_KEY = "scheduled_task_archive_confirmation"
_DELETE_CONFIRM_KEY = "scheduled_task_delete_confirmation"
_ACTION_MESSAGE_KEY = "scheduled_task_action_message"
_EDIT_JOB_KEY = "scheduled_task_edit_job_id"
_EDIT_UPDATED_AT_KEY = "scheduled_task_edit_updated_at"
_EMPTY_TEMPLATE = "Start from scratch"
_DATA_PERIOD_BY_LABEL = {label: key for key, label in DATA_PERIOD_LABELS.items()}
_OUTPUT_FORMAT_BY_LABEL = {label: key for key, label in OUTPUT_FORMAT_LABELS.items()}
_ANALYSIS_LEVEL_BY_LABEL = {label: key for key, label in ANALYSIS_LEVEL_LABELS.items()}
_SCHEDULE_KIND_BY_LABEL = {label: key for key, label in SCHEDULE_KIND_LABELS.items()}
_DELIVERY_CHANNEL_BY_LABEL = {
    label: key for key, label in DELIVERY_CHANNEL_LABELS.items()
}
_EMAIL_ATTACHMENT_BY_LABEL = {
    label: key for key, label in EMAIL_ATTACHMENT_LABELS.items()
}

_SHIFT_OPTION_TO_LABEL = {
    f"Shift {window['label']} - ends {int(window['end_hour']):02d}:00": str(
        window["label"]
    )
    for window in SHIFT_WINDOWS
}
_CRON_TO_WEEKDAY = {
    "0": "Sunday",
    "1": "Monday",
    "2": "Tuesday",
    "3": "Wednesday",
    "4": "Thursday",
    "5": "Friday",
    "6": "Saturday",
}
_STATUS_LABELS = {
    "pending_provisioning": "Waiting for activation",
    "provisioning_failed": "Setup needs attention",
    "active": "Active",
    "paused": "Paused",
    "completed": "Completed",
    "execution_failed": "Execution failed",
    "deleted": "Archived",
}
_STATUS_FILTERS = {
    "All tasks": None,
    "Active": {"active"},
    "Paused": {"paused"},
    "Waiting": {"pending_provisioning", "provisioning_failed"},
    "Finished": {"completed", "execution_failed"},
    "Archived": {"deleted"},
}


@st.cache_resource(show_spinner=False)
def _get_scheduled_job_service() -> ScheduledJobService:
    """Return the shared PostgreSQL-backed scheduled-job service."""

    return ScheduledJobService()


def _save_task_definition(
    definition: dict[str, object],
    *,
    operator: str,
) -> ScheduledJobView:
    """Persist a validated definition for the authenticated operator."""

    user_id = current_user_id()
    if not user_id:
        raise ScheduledJobPersistenceError(
            "The authenticated operator could not be resolved in the database."
        )
    return _get_scheduled_job_service().create_job(
        ScheduledJobCreateRequest(
            definition=definition,
            created_by_user_id=user_id,
            created_by_username=operator,
        )
    )


def _update_task_definition(
    definition: dict[str, object],
    *,
    operator: str,
) -> ScheduledJobView:
    """Persist an owner-scoped JSON edit using optimistic locking."""

    user_id = current_user_id()
    job_id = st.session_state.get(_EDIT_JOB_KEY)
    expected_updated_at = st.session_state.get(_EDIT_UPDATED_AT_KEY)
    if (
        not user_id
        or not isinstance(job_id, str)
        or not isinstance(expected_updated_at, datetime)
    ):
        raise ScheduledJobPersistenceError(
            "The edit context is no longer available; reload the task."
        )
    return _get_scheduled_job_service().update_job(
        job_id=job_id,
        owner_user_id=user_id,
        changed_by_username=operator,
        expected_updated_at=expected_updated_at,
        definition=definition,
    )


_QUICK_STARTS: dict[str, dict[str, object]] = {
    "Hourly ETA CO report": {
        "scheduled_task_job_type": "ETA CO Report",
        "scheduled_task_name": "Hourly BF2 ETA CO report",
        "scheduled_task_instructions": (
            "Review BF2 body ETA CO over the configured duration, evaluate the "
            "warning and critical limits, and prepare a concise operator report."
        ),
        "scheduled_task_output_format": OUTPUT_FORMAT_LABELS["operator_summary"],
        "scheduled_task_repeat": SCHEDULE_KIND_LABELS["hourly"],
        "scheduled_task_hourly_minute": 5,
    },
    "Daily furnace health report": {
        "scheduled_task_job_type": "Daily Furnace Summary",
        "scheduled_task_name": "Daily BF2 furnace health report",
        "scheduled_task_instructions": (
            "Review the previous day's BF2 process data, flag abnormal trends, "
            "and prepare a concise furnace health report for the morning team."
        ),
        "scheduled_task_data_period": DATA_PERIOD_LABELS["previous_day"],
        "scheduled_task_output_format": OUTPUT_FORMAT_LABELS["operator_summary"],
        "scheduled_task_repeat": SCHEDULE_KIND_LABELS["daily"],
        "scheduled_task_run_time": time(7, 0),
    },
    "Shift handover summary": {
        "scheduled_task_job_type": "Shift Handover Summary",
        "scheduled_task_name": "BF2 shift handover summary",
        "scheduled_task_instructions": (
            "Summarize the completed BF2 shift, highlight important deviations "
            "and operator notes, and list items that need follow-up."
        ),
        "scheduled_task_data_period": DATA_PERIOD_LABELS["previous_shift"],
        "scheduled_task_output_format": OUTPUT_FORMAT_LABELS["operator_summary"],
        "scheduled_task_repeat": SCHEDULE_KIND_LABELS["shift_end"],
        "scheduled_task_shifts": list(_SHIFT_OPTION_TO_LABEL),
        "scheduled_task_shift_delay": 10,
    },
    "Weekly operating review": {
        "scheduled_task_job_type": "Furnace Performance Summary",
        "scheduled_task_name": "Weekly BF2 operating review",
        "scheduled_task_instructions": (
            "Review the last seven days of BF2 operation, explain the main "
            "performance changes, and prepare a detailed weekly report."
        ),
        "scheduled_task_data_period": DATA_PERIOD_LABELS["custom_lookback"],
        "scheduled_task_lookback_value": 7,
        "scheduled_task_lookback_unit": "Days",
        "scheduled_task_output_format": OUTPUT_FORMAT_LABELS["detailed_report"],
        "scheduled_task_repeat": SCHEDULE_KIND_LABELS["weekly"],
        "scheduled_task_weekly_day": "Monday",
        "scheduled_task_run_time": time(8, 0),
    },
}


def _load_css() -> None:
    """Load the stylesheet scoped to the Scheduled Tasks page."""

    css_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "css"
        / "scheduled_tasks_style.css"
    )
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def _initialize_widget_state(operator: str, catalog: ScheduledJobCatalog) -> None:
    """Initialize form defaults and isolate state between signed-in operators."""

    if st.session_state.get(_STATE_OWNER_KEY) != operator:
        for key in tuple(st.session_state):
            if str(key).startswith("scheduled_task_"):
                st.session_state.pop(key, None)
        st.session_state[_STATE_OWNER_KEY] = operator

    default_timezone = catalog.default_timezone
    local_today = datetime.now(ZoneInfo(default_timezone)).date()
    eta_defaults = catalog.defaults["eta_co"]
    reliability = catalog.defaults["reliability"]
    default_signal = catalog.eta_co_signal(str(eta_defaults["signal"]))
    default_aggregation = catalog.aggregation_interval(
        str(eta_defaults["aggregation_interval"])
    )
    defaults: dict[str, object] = {
        _TEMPLATE_KEY: _EMPTY_TEMPLATE,
        "scheduled_task_job_type": catalog.job_types[0].label,
        "scheduled_task_name": "",
        "scheduled_task_instructions": "",
        "scheduled_task_analysis_level": ANALYSIS_LEVEL_LABELS[
            catalog.job_types[0].default_analysis_level
        ],
        "scheduled_task_data_source": catalog.data_sources[0].label,
        "scheduled_task_data_period": DATA_PERIOD_LABELS["previous_shift"],
        "scheduled_task_lookback_value": 8,
        "scheduled_task_lookback_unit": "Hours",
        "scheduled_task_output_format": OUTPUT_FORMAT_LABELS["operator_summary"],
        "scheduled_task_eta_signal": default_signal.label if default_signal else "",
        "scheduled_task_eta_duration": int(eta_defaults["report_duration_minutes"]),
        "scheduled_task_eta_aggregation": (
            default_aggregation.label if default_aggregation else ""
        ),
        "scheduled_task_eta_warning": float(eta_defaults["warning_threshold"]),
        "scheduled_task_eta_critical": float(eta_defaults["critical_threshold"]),
        "scheduled_task_include_graph": bool(eta_defaults["include_graph"]),
        "scheduled_task_include_ai_summary": bool(eta_defaults["include_ai_summary"]),
        "scheduled_task_repeat": SCHEDULE_KIND_LABELS["daily"],
        "scheduled_task_timezone": default_timezone,
        "scheduled_task_run_date": local_today + timedelta(days=1),
        "scheduled_task_interval_start_date": local_today,
        "scheduled_task_run_time": time(7, 0),
        "scheduled_task_hourly_minute": 5,
        "scheduled_task_selected_days": ["Monday", "Wednesday", "Friday"],
        "scheduled_task_weekly_day": "Monday",
        "scheduled_task_month_day": 1,
        "scheduled_task_interval_hours": 8,
        "scheduled_task_shifts": list(_SHIFT_OPTION_TO_LABEL),
        "scheduled_task_shift_delay": 10,
        "scheduled_task_delivery_channel": DELIVERY_CHANNEL_LABELS["in_app"],
        "scheduled_task_delivery_destination": "",
        "scheduled_task_email_recipients": "",
        "scheduled_task_email_subject": "Scheduled BF2 report",
        "scheduled_task_email_attachments": [EMAIL_ATTACHMENT_LABELS["json"]],
        "scheduled_task_notify_failure": bool(reliability["notify_on_failure"]),
        "scheduled_task_maximum_attempts": int(reliability["maximum_attempts"]),
        "scheduled_task_retry_interval": int(reliability["retry_interval_seconds"]),
        "scheduled_task_timeout": int(reliability["timeout_seconds"]),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if st.session_state["scheduled_task_repeat"] not in SCHEDULE_KIND_LABELS.values():
        st.session_state["scheduled_task_repeat"] = SCHEDULE_KIND_LABELS["daily"]


def _load_selected_template() -> None:
    """Copy the selected example values into the current form state."""

    selected = str(st.session_state.get(_TEMPLATE_KEY, _EMPTY_TEMPLATE))
    for key, value in _QUICK_STARTS.get(selected, {}).items():
        st.session_state[key] = value
    st.session_state.pop(_GENERATED_KEY, None)
    st.session_state.pop(_BUILD_ERROR_KEY, None)
    st.session_state[_ATTEMPTED_KEY] = False


def _clear_form() -> None:
    """Remove all Scheduled Tasks values from the current session."""

    for key in tuple(st.session_state):
        if str(key).startswith("scheduled_task_"):
            st.session_state.pop(key, None)


def _request_form_clear() -> None:
    """Show the confirmation prompt before clearing the form."""

    st.session_state[_CLEAR_PENDING_KEY] = True


def _cancel_form_clear() -> None:
    """Dismiss the pending form-clear confirmation."""

    st.session_state.pop(_CLEAR_PENDING_KEY, None)


def _section_header(title: str, description: str) -> None:
    """Render a consistent heading for a form section."""

    st.markdown(
        f"""
        <div class="scheduled-task-section-heading">
            <div>
                <h2 class="scheduled-task-section-title">{html.escape(title)}</h2>
                <p class="scheduled-task-section-copy">{html.escape(description)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _delivery_label(task: ScheduledTaskInput) -> str:
    """Describe the selected delivery destination for the footer summary."""

    channel = DELIVERY_CHANNEL_LABELS.get(task.delivery_channel, "Not selected")
    if task.delivery_channel == "in_app":
        return f"{channel} - signed-in operator"
    if task.delivery_channel == "email":
        count = len(parse_email_recipients(",".join(task.email_recipients)))
        suffix = "recipient" if count == 1 else "recipients"
        return f"{channel} - {count} {suffix}"
    status = (
        "destination entered" if task.delivery_destination.strip() else "not entered"
    )
    return f"{channel} - {status}"


def _render_schedule_summary(task: ScheduledTaskInput, timezone_label: str) -> None:
    """Render the human-readable schedule and next execution preview."""

    runs = upcoming_run_times(task, count=1)
    next_run = (
        f"{runs[0].strftime('%a, %d %b %Y at %H:%M')}"
        if runs
        else "Complete the schedule to preview its next run"
    )
    st.markdown(
        f"""
        <div class="scheduled-task-schedule-summary">
            <div>
                <span>Schedule</span>
                <strong>{html.escape(describe_schedule(task))}</strong>
            </div>
            <div>
                <span>Next run</span>
                <strong>{html.escape(next_run)}</strong>
                <em>{html.escape(timezone_label)}</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_generated_file(task: ScheduledTaskInput) -> None:
    """Render the persisted job receipt and JSON export controls."""

    generated = st.session_state.get(_GENERATED_KEY)
    if not isinstance(generated, dict):
        return

    if generated.get("fingerprint") != task_input_fingerprint(task):
        st.info(
            "The task details have changed. "
            "Select **Schedule** again to apply the changes."
        )
        return

    is_update = generated.get("kind") == "update"
    ready_title = "Changes saved" if is_update else "Task saved"
    ready_copy = "The validated JSON definition is stored and ready to download."
    st.markdown(
        f"""
        <div class="scheduled-task-file-ready">
            <div class="scheduled-task-ready-mark" aria-hidden="true">&#10003;</div>
            <div>
                <strong>{html.escape(ready_title)}</strong>
                <span>{html.escape(ready_copy)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download task JSON",
        data=str(generated["json"]).encode("utf-8"),
        file_name=str(generated["filename"]),
        mime="application/json",
        type="primary",
        icon=":material/download:",
        width="stretch",
        on_click="ignore",
    )
    st.caption(
        f"Task ID: {generated['job_id']} · "
        f"Status: {generated['status']} · File: {generated['filename']}"
    )

    with st.expander("Preview task JSON", expanded=False):
        st.code(str(generated["json"]), language="json", line_numbers=True)


def _format_job_timestamp(value: datetime | None, timezone_name: str) -> str:
    """Return a concise operator-facing timestamp in the job timezone."""

    if value is None:
        return "Not available"
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        zone = timezone.utc
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(zone).strftime("%d %b %Y, %H:%M")


def _definition_schedule_label(definition: dict[str, object]) -> str:
    """Describe a stored schedule without exposing cron syntax to operators."""

    schedule = definition.get("schedule")
    if not isinstance(schedule, dict):
        return "Schedule unavailable"
    frequency = str(schedule.get("frequency", ""))
    base = SCHEDULE_KIND_LABELS.get(frequency, "Schedule unavailable")
    timezone_name = str(schedule.get("timezone", "UTC"))
    trigger = schedule.get("trigger")
    if not isinstance(trigger, dict):
        return f"{base} · {timezone_name}"

    trigger_type = trigger.get("type")
    if trigger_type == "once":
        try:
            run_at = datetime.fromisoformat(str(trigger.get("run_at")))
        except ValueError:
            return f"{base} · {timezone_name}"
        return f"{base} · {run_at.strftime('%d %b %Y, %H:%M')} · {timezone_name}"
    if trigger_type == "interval":
        seconds = trigger.get("interval_seconds")
        if isinstance(seconds, int) and seconds > 0:
            hours = seconds // 3600
            unit = "hour" if hours == 1 else "hours"
            return f"Every {hours} {unit} · {timezone_name}"

    derived = trigger.get("derived_from")
    if frequency == "shift_end" and isinstance(derived, dict):
        shifts = derived.get("shifts")
        labels = (
            [
                str(item.get("label"))
                for item in shifts
                if isinstance(item, dict) and item.get("label")
            ]
            if isinstance(shifts, list)
            else []
        )
        delay = derived.get("delay_minutes")
        timing = "at shift end" if delay == 0 else f"{delay} minutes after shift end"
        selected = ", ".join(labels) if labels else "configured shifts"
        return f"After shifts {selected} · {timing} · {timezone_name}"

    expression = trigger.get("expression")
    fields = str(expression).split() if isinstance(expression, str) else []
    if len(fields) == 5:
        minute, hour, day_of_month, _, days = fields
        if frequency == "hourly" and minute.isdigit():
            return f"{base} at {int(minute):02d} minutes past · {timezone_name}"
        if minute.isdigit() and hour.isdigit():
            detail = base
            if frequency in {"weekly", "selected_days"}:
                labels = [
                    _CRON_TO_WEEKDAY[item]
                    for item in days.split(",")
                    if item in _CRON_TO_WEEKDAY
                ]
                if labels:
                    detail = f"{base} on {', '.join(labels)}"
            elif frequency == "monthly" and day_of_month.isdigit():
                detail = f"{base} on day {int(day_of_month)}"
            return f"{detail} at {int(hour):02d}:{int(minute):02d} · {timezone_name}"
    return f"{base} · {timezone_name}"


def _load_trigger_into_form(
    frequency: str,
    trigger: dict[str, object],
    timezone_name: str,
) -> None:
    """Restore the operator controls represented by a stored trigger object."""

    timestamp_key = "run_at" if frequency == "once" else "anchor_at"
    raw_timestamp = trigger.get(timestamp_key)
    if isinstance(raw_timestamp, str):
        try:
            local_value = datetime.fromisoformat(raw_timestamp)
            if local_value.tzinfo is not None and timezone_name:
                local_value = local_value.astimezone(ZoneInfo(timezone_name))
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            local_value = None
        if local_value is not None:
            date_key = (
                "scheduled_task_run_date"
                if frequency == "once"
                else "scheduled_task_interval_start_date"
            )
            st.session_state[date_key] = local_value.date()
            st.session_state["scheduled_task_run_time"] = local_value.time().replace(
                tzinfo=None
            )

    seconds = trigger.get("interval_seconds")
    if frequency == "interval_hours" and isinstance(seconds, int):
        st.session_state["scheduled_task_interval_hours"] = seconds // 3600

    expression = trigger.get("expression")
    fields = str(expression).split() if isinstance(expression, str) else []
    if len(fields) == 5:
        minute, hour, day_of_month, _, days = fields
        if minute.isdigit():
            st.session_state["scheduled_task_hourly_minute"] = int(minute)
        if minute.isdigit() and hour.isdigit():
            st.session_state["scheduled_task_run_time"] = time(int(hour), int(minute))
        if day_of_month.isdigit():
            st.session_state["scheduled_task_month_day"] = int(day_of_month)
        selected_days = [
            _CRON_TO_WEEKDAY[item]
            for item in days.split(",")
            if item in _CRON_TO_WEEKDAY
        ]
        if frequency == "weekly" and selected_days:
            st.session_state["scheduled_task_weekly_day"] = selected_days[0]
        elif frequency == "selected_days" and selected_days:
            st.session_state["scheduled_task_selected_days"] = selected_days

    derived = trigger.get("derived_from")
    if frequency == "shift_end" and isinstance(derived, dict):
        shifts = derived.get("shifts")
        if isinstance(shifts, list):
            labels = [
                str(item.get("label"))
                for item in shifts
                if isinstance(item, dict)
                and str(item.get("label")) in _SHIFT_OPTION_TO_LABEL.values()
            ]
            st.session_state["scheduled_task_shifts"] = [
                option
                for option, label in _SHIFT_OPTION_TO_LABEL.items()
                if label in labels
            ]
        delay = derived.get("delay_minutes")
        if isinstance(delay, int):
            st.session_state["scheduled_task_shift_delay"] = delay


def _load_definition_into_form(
    definition: dict[str, object],
    catalog: ScheduledJobCatalog,
) -> None:
    """Copy one stored definition into the creation form as an independent clone."""

    job_type = catalog.job_type(str(definition.get("job_type", "")))
    analysis_level = str(definition.get("analysis_level", ""))
    inputs = definition.get("inputs")
    schedule = definition.get("schedule")
    delivery = definition.get("delivery")
    retry = definition.get("retry")
    inputs = inputs if isinstance(inputs, dict) else {}
    schedule = schedule if isinstance(schedule, dict) else {}
    delivery = delivery if isinstance(delivery, dict) else {}
    retry = retry if isinstance(retry, dict) else {}

    original_name = str(definition.get("job_name", "Scheduled task")).strip()
    st.session_state["scheduled_task_name"] = f"Copy of {original_name}"[:80]
    st.session_state["scheduled_task_instructions"] = str(
        definition.get("instructions", "")
    )
    if job_type is not None:
        st.session_state["scheduled_task_job_type"] = job_type.label
    if analysis_level in ANALYSIS_LEVEL_LABELS:
        st.session_state["scheduled_task_analysis_level"] = ANALYSIS_LEVEL_LABELS[
            analysis_level
        ]

    data_source = catalog.data_source(str(inputs.get("data_source", "")))
    if data_source is not None:
        st.session_state["scheduled_task_data_source"] = data_source.label
    output_format = str(inputs.get("output_format", ""))
    if output_format in OUTPUT_FORMAT_LABELS:
        st.session_state["scheduled_task_output_format"] = OUTPUT_FORMAT_LABELS[
            output_format
        ]
    data_period = str(inputs.get("data_period", ""))
    if data_period in DATA_PERIOD_LABELS:
        st.session_state["scheduled_task_data_period"] = DATA_PERIOD_LABELS[data_period]
    lookback = inputs.get("lookback")
    if isinstance(lookback, dict):
        st.session_state["scheduled_task_lookback_value"] = int(
            lookback.get("value", 1)
        )
        unit = str(lookback.get("unit", "hours")).strip().lower()
        st.session_state["scheduled_task_lookback_unit"] = (
            "Days" if unit == "days" else "Hours"
        )

    signal = catalog.eta_co_signal(str(inputs.get("signal", "")))
    if signal is not None:
        st.session_state["scheduled_task_eta_signal"] = signal.label
    aggregation = catalog.aggregation_interval(
        str(inputs.get("aggregation_interval", ""))
    )
    if aggregation is not None:
        st.session_state["scheduled_task_eta_aggregation"] = aggregation.label
    eta_values = {
        "report_duration_minutes": "scheduled_task_eta_duration",
        "warning_threshold": "scheduled_task_eta_warning",
        "critical_threshold": "scheduled_task_eta_critical",
        "include_graph": "scheduled_task_include_graph",
        "include_ai_summary": "scheduled_task_include_ai_summary",
    }
    for source_key, state_key in eta_values.items():
        if source_key in inputs:
            st.session_state[state_key] = inputs[source_key]

    frequency = str(schedule.get("frequency", ""))
    if frequency in SCHEDULE_KIND_LABELS:
        st.session_state["scheduled_task_repeat"] = SCHEDULE_KIND_LABELS[frequency]
    timezone_name = str(schedule.get("timezone", ""))
    if catalog.timezone(timezone_name) is not None:
        st.session_state["scheduled_task_timezone"] = timezone_name
    trigger = schedule.get("trigger")
    _load_trigger_into_form(
        frequency,
        trigger if isinstance(trigger, dict) else {},
        timezone_name,
    )

    channel = str(delivery.get("channel", ""))
    if channel in DELIVERY_CHANNEL_LABELS:
        st.session_state["scheduled_task_delivery_channel"] = DELIVERY_CHANNEL_LABELS[
            channel
        ]
    st.session_state["scheduled_task_delivery_destination"] = str(
        delivery.get("destination", "")
    )
    recipients = delivery.get("recipients")
    if isinstance(recipients, list):
        st.session_state["scheduled_task_email_recipients"] = ", ".join(
            str(item) for item in recipients
        )
    st.session_state["scheduled_task_email_subject"] = str(delivery.get("subject", ""))
    attachments = delivery.get("attachments")
    if isinstance(attachments, list):
        st.session_state["scheduled_task_email_attachments"] = [
            EMAIL_ATTACHMENT_LABELS[item]
            for item in attachments
            if item in EMAIL_ATTACHMENT_LABELS
        ]
    if "notify_on_failure" in delivery:
        st.session_state["scheduled_task_notify_failure"] = bool(
            delivery["notify_on_failure"]
        )

    retry_fields = {
        "maximum_attempts": "scheduled_task_maximum_attempts",
        "retry_interval_seconds": "scheduled_task_retry_interval",
        "timeout_seconds": "scheduled_task_timeout",
    }
    for source_key, state_key in retry_fields.items():
        if source_key in retry:
            st.session_state[state_key] = int(retry[source_key])

    st.session_state[_TEMPLATE_KEY] = _EMPTY_TEMPLATE
    st.session_state[_VIEW_KEY] = "Create task"
    st.session_state[_ATTEMPTED_KEY] = False
    st.session_state.pop(_GENERATED_KEY, None)
    st.session_state.pop(_BUILD_ERROR_KEY, None)
    st.session_state.pop(_CLEAR_PENDING_KEY, None)
    st.session_state.pop(_EDIT_JOB_KEY, None)
    st.session_state.pop(_EDIT_UPDATED_AT_KEY, None)


def _load_definition_for_edit(
    definition: dict[str, object],
    catalog: ScheduledJobCatalog,
    job_id: str,
    updated_at: datetime,
) -> None:
    """Load a stored definition and retain its optimistic edit version."""

    _load_definition_into_form(definition, catalog)
    st.session_state["scheduled_task_name"] = str(
        definition.get("job_name", "Scheduled task")
    )[:80]
    st.session_state[_EDIT_JOB_KEY] = job_id
    st.session_state[_EDIT_UPDATED_AT_KEY] = updated_at


def _switch_to_creator() -> None:
    """Switch the Scheduled Tasks workspace to the creation view."""

    st.session_state.pop(_EDIT_JOB_KEY, None)
    st.session_state.pop(_EDIT_UPDATED_AT_KEY, None)
    st.session_state[_VIEW_KEY] = "Create task"


def _cancel_edit() -> None:
    """Discard the edit context and return to the task-management view."""

    st.session_state.pop(_EDIT_JOB_KEY, None)
    st.session_state.pop(_EDIT_UPDATED_AT_KEY, None)
    st.session_state.pop(_GENERATED_KEY, None)
    st.session_state.pop(_BUILD_ERROR_KEY, None)
    st.session_state[_ATTEMPTED_KEY] = False
    st.session_state[_VIEW_KEY] = "Your tasks"


def _cancel_archive() -> None:
    """Dismiss the pending archive confirmation."""

    st.session_state.pop(_ARCHIVE_CONFIRM_KEY, None)


def _cancel_delete() -> None:
    """Dismiss the pending permanent-delete confirmation."""

    st.session_state.pop(_DELETE_CONFIRM_KEY, None)


def _management_action(status: str) -> tuple[str, str] | None:
    """Return the future lifecycle action displayed for a saved status."""

    if status in {"pending_provisioning", "provisioning_failed"}:
        return "provision", "Activate"
    if status == "active":
        return "pause", "Pause"
    if status == "paused":
        return "resume", "Resume"
    return None


def _archive_task_definition(
    *,
    job_id: str,
    owner_user_id: str,
    expected_updated_at: datetime,
) -> None:
    """Archive a saved definition without changing an external timer."""

    try:
        _get_scheduled_job_service().archive_job(
            job_id=job_id,
            owner_user_id=owner_user_id,
            expected_updated_at=expected_updated_at,
        )
    except (ScheduledJobPersistenceError, ScheduledJobValidationError):
        st.session_state[_ACTION_MESSAGE_KEY] = (
            "error",
            "The task could not be archived. Please refresh and try again.",
        )
    except ScheduledJobConflictError:
        st.session_state[_ACTION_MESSAGE_KEY] = (
            "error",
            "The task changed after the page loaded. Refresh and try again.",
        )
    else:
        st.session_state[_ACTION_MESSAGE_KEY] = (
            "success",
            "The saved task definition was archived.",
        )


def _delete_task_definition(*, job_id: str, owner_user_id: str) -> None:
    """Permanently delete an archived owner definition."""

    try:
        _get_scheduled_job_service().delete_job(
            job_id=job_id,
            owner_user_id=owner_user_id,
        )
    except (ScheduledJobPersistenceError, ScheduledJobValidationError):
        st.session_state[_ACTION_MESSAGE_KEY] = (
            "error",
            "The archived task could not be deleted. Please try again.",
        )
    except ScheduledJobConflictError:
        st.session_state[_ACTION_MESSAGE_KEY] = (
            "error",
            "Only an archived task owned by this operator can be deleted.",
        )
    else:
        st.session_state.pop(_SELECTED_JOB_KEY, None)
        st.session_state[_ACTION_MESSAGE_KEY] = (
            "success",
            "The archived task definition was permanently deleted.",
        )


def _render_action_message() -> None:
    """Render and consume the latest definition-management feedback."""

    message = st.session_state.pop(_ACTION_MESSAGE_KEY, None)
    if not isinstance(message, tuple) or len(message) != 2:
        return
    level, text = message
    if level == "success":
        st.success(str(text))
    else:
        st.error(str(text))


def _render_run_history() -> None:
    """Render the execution-history placeholder retained for the future phase."""

    st.info("This task has not run yet. Execution is not enabled in this milestone.")


def _render_task_management(
    operator: str,
    catalog: ScheduledJobCatalog,
) -> None:
    """Render saved definitions and future-facing lifecycle placeholders."""

    owner_user_id = current_user_id()
    if not owner_user_id:
        st.error("The signed-in operator could not be resolved in the database.")
        return
    service = _get_scheduled_job_service()
    try:
        items = service.list_jobs_for_owner(owner_user_id, limit=100)
    except (ScheduledJobPersistenceError, ScheduledJobValidationError):
        st.error("Scheduled tasks could not be loaded. Check the database connection.")
        return

    with st.container(border=True, key="scheduled_task_manager"):
        st.markdown(
            """
            <div class="scheduled-task-creator-heading">
                <span>TASK LIBRARY</span>
                <h2>Your scheduled tasks</h2>
                <p>Inspect, export, and manage your saved task definitions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_action_message()
        if not items:
            st.info("No scheduled tasks have been saved for this operator yet.")
            st.button(
                "Create your first task",
                key="scheduled_task_empty_create",
                type="primary",
                on_click=_switch_to_creator,
            )
            return

        filter_col, search_col = st.columns([0.36, 0.64])
        with filter_col:
            status_filter = st.selectbox(
                "Status",
                options=list(_STATUS_FILTERS),
                key="scheduled_task_status_filter",
            )
        with search_col:
            search = (
                st.text_input(
                    "Find a task",
                    key="scheduled_task_search",
                    placeholder="Search by task name",
                )
                .strip()
                .casefold()
            )
        allowed_statuses = _STATUS_FILTERS[status_filter]
        filtered = tuple(
            item
            for item in items
            if (allowed_statuses is None or item.job.status in allowed_statuses)
            and (not search or search in item.job.job_name.casefold())
        )
        if not filtered:
            st.info("No tasks match the current filters.")
            return

        label_to_item = {
            f"{item.job.job_name} · {item.job.job_id[:8]}": item for item in filtered
        }
        selected_label = st.selectbox(
            "Task",
            options=list(label_to_item),
            key=_SELECTED_JOB_KEY,
        )
        selected_item: ScheduledJobListItem = label_to_item[selected_label]
        try:
            selected = service.get_job_for_owner(
                selected_item.job.job_id,
                owner_user_id,
            )
        except (ScheduledJobPersistenceError, ScheduledJobValidationError):
            st.error("The selected task could not be loaded.")
            return
        if selected is None:
            st.warning("The selected task is no longer available.")
            return

        schedule = selected.definition.get("schedule")
        schedule = schedule if isinstance(schedule, dict) else {}
        timezone_name = str(schedule.get("timezone", catalog.default_timezone))
        status_label = _STATUS_LABELS.get(
            selected.status, selected.status.replace("_", " ").title()
        )
        last_run = _format_job_timestamp(selected_item.last_run_at, timezone_name)
        last_status = (
            selected_item.last_run_status.replace("_", " ").title()
            if selected_item.last_run_status
            else "Not run yet"
        )
        st.markdown(
            f"""
            <div class="scheduled-task-detail-card">
                <div>
                    <span class="scheduled-task-status scheduled-task-status-{html.escape(selected.status)}">{html.escape(status_label)}</span>
                    <h3>{html.escape(selected.job_name)}</h3>
                    <p>{html.escape(_definition_schedule_label(selected.definition))}</p>
                </div>
                <dl>
                    <div><dt>Last run</dt><dd>{html.escape(last_run)}</dd></div>
                    <div><dt>Outcome</dt><dd>{html.escape(last_status)}</dd></div>
                </dl>
            </div>
            """,
            unsafe_allow_html=True,
        )

        primary = _management_action(selected.status)
        action_col, edit_col, clone_col, archive_col, delete_col = st.columns(5)
        with action_col:
            if primary is not None:
                _, label = primary
                st.button(
                    label,
                    key=f"scheduled_task_action_{selected.job_id}",
                    type="primary",
                    width="stretch",
                    disabled=True,
                    help="Task execution is not enabled in this milestone.",
                )
            else:
                st.button(
                    "No lifecycle action",
                    key=f"scheduled_task_no_action_{selected.job_id}",
                    width="stretch",
                    disabled=True,
                )
        with edit_col:
            st.button(
                "Edit",
                key=f"scheduled_task_edit_{selected.job_id}",
                width="stretch",
                disabled=selected.status
                not in {
                    "pending_provisioning",
                    "provisioning_failed",
                    "paused",
                    "active",
                },
                on_click=_load_definition_for_edit,
                args=(
                    selected.definition,
                    catalog,
                    selected.job_id,
                    selected.updated_at,
                ),
            )
        with clone_col:
            st.button(
                "Clone",
                key=f"scheduled_task_clone_{selected.job_id}",
                width="stretch",
                on_click=_load_definition_into_form,
                args=(selected.definition, catalog),
            )
        with archive_col:
            if st.button(
                "Archive",
                key=f"scheduled_task_archive_{selected.job_id}",
                width="stretch",
                disabled=selected.status == "deleted",
            ):
                st.session_state[_ARCHIVE_CONFIRM_KEY] = selected.job_id
        with delete_col:
            if st.button(
                "Delete",
                key=f"scheduled_task_delete_{selected.job_id}",
                width="stretch",
                disabled=selected.status != "deleted",
                help="Archive the task before permanently deleting it.",
            ):
                st.session_state[_DELETE_CONFIRM_KEY] = selected.job_id

        st.caption(
            "Timer controls are read-only in this milestone. "
            "Activate, Pause, and Resume will be enabled when task execution is deployed."
        )
        if st.session_state.get(_ARCHIVE_CONFIRM_KEY) == selected.job_id:
            st.warning(
                "Archive this saved definition? You can still inspect, clone, and download it."
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "Confirm archive",
                    key=f"scheduled_task_archive_confirm_{selected.job_id}",
                    type="primary",
                    width="stretch",
                ):
                    st.session_state.pop(_ARCHIVE_CONFIRM_KEY, None)
                    _archive_task_definition(
                        job_id=selected.job_id,
                        owner_user_id=owner_user_id,
                        expected_updated_at=selected.updated_at,
                    )
                    st.rerun()
            with cancel_col:
                st.button(
                    "Keep task",
                    key=f"scheduled_task_archive_cancel_{selected.job_id}",
                    width="stretch",
                    on_click=_cancel_archive,
                )

        if st.session_state.get(_DELETE_CONFIRM_KEY) == selected.job_id:
            st.error(
                "Permanently delete this archived task? This removes its stored JSON "
                "and cannot be undone."
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "Confirm permanent delete",
                    key=f"scheduled_task_delete_confirm_{selected.job_id}",
                    type="primary",
                    width="stretch",
                ):
                    st.session_state.pop(_DELETE_CONFIRM_KEY, None)
                    _delete_task_definition(
                        job_id=selected.job_id,
                        owner_user_id=owner_user_id,
                    )
                    st.rerun()
            with cancel_col:
                st.button(
                    "Keep archived task",
                    key=f"scheduled_task_delete_cancel_{selected.job_id}",
                    width="stretch",
                    on_click=_cancel_delete,
                )

        overview_tab, runs_tab, json_tab = st.tabs(("Overview", "Run history", "JSON"))
        with overview_tab:
            job_type = catalog.job_type(str(selected.definition.get("job_type", "")))
            instructions = str(selected.definition.get("instructions", ""))
            st.markdown("**Task type**")
            st.write(job_type.label if job_type is not None else "Unknown task type")
            st.markdown("**Instructions**")
            st.write(instructions)
            created = _format_job_timestamp(selected.created_at, timezone_name)
            creator = selected.created_by_username or operator
            updated = _format_job_timestamp(selected.updated_at, timezone_name)
            st.caption(
                f"Created {created} by {creator} · Updated {updated} · "
                f"Task ID {selected.job_id}"
            )
        with runs_tab:
            _render_run_history()
        with json_tab:
            definition_json = json.dumps(
                selected.definition,
                indent=2,
                ensure_ascii=False,
            )
            st.download_button(
                "Download task JSON",
                data=definition_json.encode("utf-8"),
                file_name=task_definition_filename(selected.job_name),
                mime="application/json",
                key=f"scheduled_task_download_{selected.job_id}",
                on_click="ignore",
            )
            st.code(definition_json, language="json", line_numbers=True)


def render_scheduled_tasks_page() -> None:
    """Render scheduled-task creation and owner management experiences."""

    apply_styles()
    _load_css()
    catalog = get_scheduled_job_catalog()
    operator = str(st.session_state.get("auth_user", "")).strip()
    _initialize_widget_state(operator, catalog)

    job_type_by_label = {item.label: item for item in catalog.job_types}
    data_source_by_label = {item.label: item.id for item in catalog.data_sources}
    signal_by_label = {item.label: item.id for item in catalog.eta_co_signals}
    aggregation_by_label = {
        item.label: item.id for item in catalog.aggregation_intervals
    }
    timezone_by_name = {item.name: item.label for item in catalog.timezones}
    target = catalog.default_target_device

    with st.container(key="scheduled_task_shell"):
        st.markdown(
            """
            <div class="scheduled-task-hero">
                <h1>Scheduled Tasks</h1>
                <p>Create FurnaceMind tasks for BF2 operations and choose when they run.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        view = st.segmented_control(
            "Scheduled task view",
            options=("Create task", "Your tasks"),
            default="Create task",
            key=_VIEW_KEY,
            label_visibility="collapsed",
        )
        if view == "Your tasks":
            _render_task_management(operator, catalog)
            return

        editing = isinstance(st.session_state.get(_EDIT_JOB_KEY), str)
        heading_eyebrow = "EDIT TASK" if editing else "NEW TASK"
        heading_title = "Edit scheduled task" if editing else "Create a scheduled task"
        heading_copy = (
            "Review the full definition and save the updated JSON."
            if editing
            else "Define the task, its schedule, and the JSON details to store."
        )
        with st.container(border=True, key="scheduled_task_creator"):
            st.markdown(
                f"""
                <div class="scheduled-task-creator-heading">
                    <span>{html.escape(heading_eyebrow)}</span>
                    <h2>{html.escape(heading_title)}</h2>
                    <p>{html.escape(heading_copy)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.popover(
                "Use an example",
                icon=":material/auto_awesome:",
            ):
                st.selectbox(
                    "Example",
                    options=[_EMPTY_TEMPLATE, *_QUICK_STARTS],
                    key=_TEMPLATE_KEY,
                    label_visibility="collapsed",
                )
                st.button(
                    "Apply example",
                    key="scheduled_task_use_template",
                    width="stretch",
                    disabled=st.session_state[_TEMPLATE_KEY] == _EMPTY_TEMPLATE,
                    on_click=_load_selected_template,
                )

            with st.container(border=False, key="scheduled_task_brief"):
                _section_header(
                    "Task",
                    "Name the task and describe the outcome in operator language.",
                )
                task_name = st.text_input(
                    "Task name",
                    key="scheduled_task_name",
                    max_chars=80,
                    placeholder="Hourly BF2 ETA CO report",
                )
                instructions = st.text_area(
                    "What should FurnaceMind do?",
                    key="scheduled_task_instructions",
                    max_chars=4000,
                    height=122,
                    placeholder=(
                        "Review BF2 ETA CO, evaluate the limits, and prepare a "
                        "short report for the control-room operator."
                    ),
                    help="Do not enter passwords, API keys, tokens, or connection strings.",
                )
                job_type_label = st.selectbox(
                    "Task type",
                    options=list(job_type_by_label),
                    key="scheduled_task_job_type",
                )
                job_type_option = job_type_by_label[job_type_label]
                analysis_label = str(st.session_state["scheduled_task_analysis_level"])
                analysis_level = _ANALYSIS_LEVEL_BY_LABEL[analysis_label]
                st.caption(job_type_option.description)

            with st.container(border=False, key="scheduled_task_data"):
                _section_header(
                    "Report settings",
                    "Choose the BF2 data and how the result should be prepared.",
                )
                st.markdown(
                    """
                    <div class="scheduled-task-scope-note">
                        <span>Furnace</span><strong>BF2</strong>
                        <em>Assigned to this workspace</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                source_col, result_col = st.columns(2)
                with source_col:
                    data_source_label = st.selectbox(
                        "Data source",
                        options=list(data_source_by_label),
                        key="scheduled_task_data_source",
                    )
                with result_col:
                    st.selectbox(
                        "Result style",
                        options=list(OUTPUT_FORMAT_LABELS.values()),
                        key="scheduled_task_output_format",
                    )
                data_source = data_source_by_label[data_source_label]

                data_period = ""
                custom_lookback_value: int | None = None
                custom_lookback_unit: str | None = None
                eta_co_signal = str(catalog.defaults["eta_co"]["signal"])
                report_duration_minutes = int(
                    catalog.defaults["eta_co"]["report_duration_minutes"]
                )
                aggregation_interval = str(
                    catalog.defaults["eta_co"]["aggregation_interval"]
                )
                warning_threshold = float(
                    catalog.defaults["eta_co"]["warning_threshold"]
                )
                critical_threshold = float(
                    catalog.defaults["eta_co"]["critical_threshold"]
                )
                include_graph = bool(catalog.defaults["eta_co"]["include_graph"])
                include_ai_summary = False

                if job_type_option.input_profile == "eta_co":
                    signal_col, duration_col = st.columns([0.56, 0.44])
                    with signal_col:
                        signal_label = st.selectbox(
                            "Signal",
                            options=list(signal_by_label),
                            key="scheduled_task_eta_signal",
                        )
                        eta_co_signal = signal_by_label[signal_label]
                    with duration_col:
                        report_duration_minutes = int(
                            st.number_input(
                                "Data window (minutes)",
                                min_value=5,
                                max_value=1440,
                                step=5,
                                key="scheduled_task_eta_duration",
                            )
                        )

                    aggregation_label = st.selectbox(
                        "Aggregation interval",
                        options=list(aggregation_by_label),
                        key="scheduled_task_eta_aggregation",
                    )
                    aggregation_interval = aggregation_by_label[aggregation_label]

                    warning_col, critical_col = st.columns(2)
                    with warning_col:
                        warning_threshold = float(
                            st.number_input(
                                "Warning below (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=0.5,
                                key="scheduled_task_eta_warning",
                            )
                        )
                    with critical_col:
                        critical_threshold = float(
                            st.number_input(
                                "Critical below (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=0.5,
                                key="scheduled_task_eta_critical",
                            )
                        )
                    st.caption(
                        "Lower ETA CO is worse. The critical value must be below the warning value."
                    )

                    option_col, ai_col = st.columns(2)
                    with option_col:
                        include_graph = st.toggle(
                            "Include trend graph",
                            key="scheduled_task_include_graph",
                        )
                    if analysis_level == "none":
                        st.session_state["scheduled_task_include_ai_summary"] = False
                    with ai_col:
                        include_ai_summary = st.toggle(
                            "Include AI summary",
                            key="scheduled_task_include_ai_summary",
                            disabled=analysis_level == "none",
                        )
                    if analysis_level == "none":
                        st.caption(
                            "AI summary is off because Analysis level is set to No AI analysis in Advanced."
                        )
                else:
                    data_period = (
                        fixed_data_period_for_job_type(job_type_option.id) or ""
                    )
                    if data_period:
                        st.caption(
                            f"Data period: **{DATA_PERIOD_LABELS[data_period]}** "
                            "(set by task type)"
                        )
                    else:
                        data_period_label = st.selectbox(
                            "Use data from",
                            options=list(DATA_PERIOD_LABELS.values()),
                            key="scheduled_task_data_period",
                        )
                        data_period = _DATA_PERIOD_BY_LABEL[data_period_label]
                    if data_period == "custom_lookback":
                        lookback_col, unit_col = st.columns([0.55, 0.45])
                        with lookback_col:
                            custom_lookback_value = int(
                                st.number_input(
                                    "Look back",
                                    min_value=1,
                                    max_value=720,
                                    step=1,
                                    key="scheduled_task_lookback_value",
                                )
                            )
                        with unit_col:
                            custom_unit_label = st.selectbox(
                                "Unit",
                                options=["Hours", "Days"],
                                key="scheduled_task_lookback_unit",
                            )
                            custom_lookback_unit = custom_unit_label.lower()
                        if (
                            custom_lookback_unit == "days"
                            and custom_lookback_value > 90
                        ):
                            st.warning("For days, choose 90 or fewer.")

            with st.container(border=False, key="scheduled_task_timing"):
                _section_header(
                    "Schedule",
                    "Choose when and how often the task should run.",
                )
                repeat_col, timezone_col = st.columns([0.58, 0.42])
                with repeat_col:
                    repeat_label = st.selectbox(
                        "Repeat",
                        options=list(SCHEDULE_KIND_LABELS.values()),
                        key="scheduled_task_repeat",
                    )
                with timezone_col:
                    timezone_name = st.selectbox(
                        "Timezone",
                        options=list(timezone_by_name),
                        format_func=timezone_by_name.__getitem__,
                        key="scheduled_task_timezone",
                    )
                timezone_label = timezone_by_name[timezone_name]
                schedule_kind = _SCHEDULE_KIND_BY_LABEL[repeat_label]

                run_date = None
                run_time = None
                hourly_minute = 0
                days_of_week: tuple[str, ...] = ()
                day_of_month = None
                interval_hours = None
                shift_labels: tuple[str, ...] = ()
                shift_delay_minutes = 0

                if schedule_kind == "once":
                    date_col, time_col = st.columns(2)
                    with date_col:
                        run_date = st.date_input(
                            "Run date",
                            key="scheduled_task_run_date",
                            min_value=datetime.now(ZoneInfo(timezone_name)).date(),
                        )
                    with time_col:
                        run_time = st.time_input(
                            "Run time",
                            key="scheduled_task_run_time",
                            step=timedelta(minutes=5),
                        )
                elif schedule_kind == "hourly":
                    hourly_minute = int(
                        st.number_input(
                            "Minute past each hour",
                            min_value=0,
                            max_value=59,
                            step=1,
                            key="scheduled_task_hourly_minute",
                            help="Example: 05 runs at 08:05, 09:05, 10:05, and so on.",
                        )
                    )
                elif schedule_kind in {
                    "daily",
                    "weekdays",
                    "selected_days",
                    "weekly",
                    "monthly",
                }:
                    run_time = st.time_input(
                        "Run time",
                        key="scheduled_task_run_time",
                        step=timedelta(minutes=5),
                    )
                    if schedule_kind == "selected_days":
                        selected_days = st.multiselect(
                            "Run on",
                            options=list(WEEKDAY_NAMES),
                            key="scheduled_task_selected_days",
                        )
                        days_of_week = tuple(selected_days)
                    elif schedule_kind == "weekly":
                        weekly_day = st.selectbox(
                            "Run every",
                            options=list(WEEKDAY_NAMES),
                            key="scheduled_task_weekly_day",
                        )
                        days_of_week = (weekly_day,)
                    elif schedule_kind == "monthly":
                        day_of_month = int(
                            st.number_input(
                                "Day of month",
                                min_value=1,
                                max_value=28,
                                step=1,
                                key="scheduled_task_month_day",
                                help="Days 1-28 work reliably in every month.",
                            )
                        )
                elif schedule_kind == "interval_hours":
                    interval_col, date_col, time_col = st.columns([0.32, 0.34, 0.34])
                    with interval_col:
                        interval_hours = int(
                            st.number_input(
                                "Every (hours)",
                                min_value=1,
                                max_value=168,
                                step=1,
                                key="scheduled_task_interval_hours",
                            )
                        )
                    with date_col:
                        run_date = st.date_input(
                            "Start date",
                            key="scheduled_task_interval_start_date",
                        )
                    with time_col:
                        run_time = st.time_input(
                            "Start time",
                            key="scheduled_task_run_time",
                            step=timedelta(minutes=5),
                        )
                elif schedule_kind == "shift_end":
                    selected_shifts = st.multiselect(
                        "Run after",
                        options=list(_SHIFT_OPTION_TO_LABEL),
                        key="scheduled_task_shifts",
                    )
                    shift_labels = tuple(
                        _SHIFT_OPTION_TO_LABEL[option] for option in selected_shifts
                    )
                    shift_delay_minutes = int(
                        st.number_input(
                            "Wait after shift end (minutes)",
                            min_value=0,
                            max_value=180,
                            step=5,
                            key="scheduled_task_shift_delay",
                        )
                    )
                    st.caption(shift_windows_description())
            delivery_destination = ""
            email_recipients: tuple[str, ...] = ()
            email_subject = ""
            email_attachments: tuple[str, ...] = ()
            with st.container(border=False, key="scheduled_task_delivery"):
                with st.expander("Delivery", expanded=False):
                    st.caption("Choose where the completed result should be delivered.")
                    delivery_channel_label = st.selectbox(
                        "Delivery channel",
                        options=list(DELIVERY_CHANNEL_LABELS.values()),
                        key="scheduled_task_delivery_channel",
                    )
                    delivery_channel = _DELIVERY_CHANNEL_BY_LABEL[
                        delivery_channel_label
                    ]

                    if delivery_channel == "email":
                        recipient_text = st.text_area(
                            "Recipients",
                            key="scheduled_task_email_recipients",
                            height=90,
                            placeholder=(
                                "shift.lead@example.com, process.team@example.com"
                            ),
                            help=(
                                "Separate addresses with commas, semicolons, or new lines."
                            ),
                        )
                        email_recipients = parse_email_recipients(recipient_text)
                        email_subject = st.text_input(
                            "Email subject",
                            key="scheduled_task_email_subject",
                            max_chars=160,
                        )
                        selected_attachments = st.multiselect(
                            "Attachments",
                            options=list(EMAIL_ATTACHMENT_LABELS.values()),
                            key="scheduled_task_email_attachments",
                        )
                        email_attachments = tuple(
                            _EMAIL_ATTACHMENT_BY_LABEL[label]
                            for label in selected_attachments
                        )
                    elif delivery_channel in {"whatsapp", "telegram"}:
                        labels = {
                            "whatsapp": (
                                "WhatsApp number",
                                "+91 98765 43210",
                                "Include the country code.",
                            ),
                            "telegram": (
                                "Telegram username or chat ID",
                                "@bf2_operator",
                                "Enter a username or numeric chat ID.",
                            ),
                        }
                        destination_label, placeholder, help_text = labels[
                            delivery_channel
                        ]
                        delivery_destination = st.text_input(
                            destination_label,
                            key="scheduled_task_delivery_destination",
                            placeholder=placeholder,
                            help=help_text,
                        )
                    else:
                        st.caption("Results will appear for the signed-in operator.")

            with st.container(border=False, key="scheduled_task_advanced"):
                with st.expander("Advanced", expanded=False):
                    analysis_label = st.selectbox(
                        "Analysis level",
                        options=list(ANALYSIS_LEVEL_LABELS.values()),
                        key="scheduled_task_analysis_level",
                        help=(
                            "Controls how much AI analysis FurnaceMind performs "
                            "for this task."
                        ),
                    )
                    analysis_level = _ANALYSIS_LEVEL_BY_LABEL[analysis_label]
                    attempts_col, retry_col, timeout_col = st.columns(3)
                    with attempts_col:
                        maximum_attempts = int(
                            st.number_input(
                                "Maximum attempts",
                                min_value=1,
                                max_value=10,
                                step=1,
                                key="scheduled_task_maximum_attempts",
                            )
                        )
                    with retry_col:
                        retry_interval_seconds = int(
                            st.number_input(
                                "Retry wait (seconds)",
                                min_value=1,
                                max_value=3600,
                                step=15,
                                key="scheduled_task_retry_interval",
                            )
                        )
                    with timeout_col:
                        timeout_seconds = int(
                            st.number_input(
                                "Timeout (seconds)",
                                min_value=60,
                                max_value=86400,
                                step=60,
                                key="scheduled_task_timeout",
                            )
                        )
                    notify_on_failure = st.toggle(
                        "Notify me if the task fails",
                        key="scheduled_task_notify_failure",
                        help="Adds failure notifications to this task.",
                    )
                    st.caption(
                        "These reliability preferences are stored in the JSON for the future execution phase."
                    )

            footer_slot = st.empty()

        output_format_label = str(st.session_state["scheduled_task_output_format"])
        if job_type_option.id == "eta_co_report" and "png" in email_attachments:
            # Selecting the explicit graph attachment is itself an operator
            # request for a graph, even when the ETA toggle began disabled.
            include_graph = True
        task = ScheduledTaskInput(
            name=task_name,
            instructions=instructions,
            furnace="BF2",
            data_period=data_period,
            output_format=_OUTPUT_FORMAT_BY_LABEL[output_format_label],
            schedule_kind=schedule_kind,
            delivery_channel=delivery_channel,
            job_type=job_type_option.id,
            analysis_level=analysis_level,
            data_source=data_source,
            target_device_id=target.device_id,
            target_device_type=target.device_type,
            timezone_name=timezone_name,
            run_date=run_date,
            run_time=run_time,
            hourly_minute=hourly_minute,
            days_of_week=days_of_week,
            day_of_month=day_of_month,
            interval_hours=interval_hours,
            shift_labels=shift_labels,
            shift_delay_minutes=shift_delay_minutes,
            custom_lookback_value=custom_lookback_value,
            custom_lookback_unit=custom_lookback_unit,
            eta_co_signal=eta_co_signal,
            report_duration_minutes=report_duration_minutes,
            aggregation_interval=aggregation_interval,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            include_graph=include_graph,
            include_ai_summary=include_ai_summary,
            delivery_destination=delivery_destination,
            email_recipients=email_recipients,
            email_subject=email_subject,
            email_attachments=email_attachments,
            notify_on_failure=notify_on_failure,
            maximum_attempts=maximum_attempts,
            retry_interval_seconds=retry_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
        validation_errors = validate_task_input(task)

        with footer_slot.container():
            _render_schedule_summary(task, timezone_label)
            st.markdown(
                f"""
                <div class="scheduled-task-action-context">
                    <span>{html.escape(job_type_option.label)}</span>
                    <span>{html.escape(_delivery_label(task))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            clear_col, create_col = st.columns([0.34, 0.66])
            with clear_col:
                if editing:
                    st.button(
                        "Cancel edit",
                        key="scheduled_task_cancel_edit",
                        width="stretch",
                        on_click=_cancel_edit,
                    )
                else:
                    st.button(
                        "Clear",
                        key="scheduled_task_clear",
                        width="stretch",
                        on_click=_request_form_clear,
                    )
            with create_col:
                create_clicked = st.button(
                    "Save changes" if editing else "Schedule",
                    key="scheduled_task_create",
                    type="primary",
                    icon=":material/save:" if editing else ":material/schedule:",
                    width="stretch",
                )

            if not editing and st.session_state.get(_CLEAR_PENDING_KEY):
                st.warning("Discard all entered task details?")
                confirm_col, keep_col = st.columns(2)
                with confirm_col:
                    st.button(
                        "Yes, clear everything",
                        key="scheduled_task_clear_confirm",
                        type="primary",
                        width="stretch",
                        on_click=_clear_form,
                    )
                with keep_col:
                    st.button(
                        "Keep my details",
                        key="scheduled_task_clear_cancel",
                        width="stretch",
                        on_click=_cancel_form_clear,
                    )

            if create_clicked:
                st.session_state.pop(_CLEAR_PENDING_KEY, None)
                st.session_state[_ATTEMPTED_KEY] = True
                st.session_state.pop(_BUILD_ERROR_KEY, None)
                if validation_errors:
                    st.session_state.pop(_GENERATED_KEY, None)
                else:
                    try:
                        definition = build_task_definition(
                            task,
                            generated_at=datetime.now(timezone.utc),
                        )
                        stored_job = (
                            _update_task_definition(
                                definition,
                                operator=operator,
                            )
                            if editing
                            else _save_task_definition(
                                definition,
                                operator=operator,
                            )
                        )
                    except ScheduledTaskValidationError as exc:
                        validation_errors = exc.errors
                        st.session_state.pop(_GENERATED_KEY, None)
                    except ScheduledJobValidationError as exc:
                        validation_errors = exc.errors
                        st.session_state.pop(_GENERATED_KEY, None)
                    except ScheduledJobConflictError:
                        st.session_state[_BUILD_ERROR_KEY] = (
                            "The task changed after this edit was opened. "
                            "Return to Your tasks and reload it before trying again."
                        )
                        st.session_state.pop(_GENERATED_KEY, None)
                    except (ScheduledJobPersistenceError, ValueError):
                        st.session_state[_BUILD_ERROR_KEY] = (
                            "The task definition is valid, but it could not be saved. "
                            "Please check the database connection and try again."
                        )
                        st.session_state.pop(_GENERATED_KEY, None)
                    else:
                        st.session_state[_GENERATED_KEY] = {
                            "fingerprint": task_input_fingerprint(task),
                            "filename": task_definition_filename(task.name),
                            "json": task_definition_json(stored_job.definition),
                            "job_id": stored_job.job_id,
                            "status": stored_job.status,
                            "kind": "update" if editing else "create",
                        }

            if st.session_state.get(_ATTEMPTED_KEY) and validation_errors:
                st.error("Review these details before scheduling the task:")
                st.markdown("\n".join(f"- {error}" for error in validation_errors))
            if st.session_state.get(_BUILD_ERROR_KEY):
                st.error(str(st.session_state[_BUILD_ERROR_KEY]))
            _render_generated_file(task)


__all__ = ["render_scheduled_tasks_page"]
