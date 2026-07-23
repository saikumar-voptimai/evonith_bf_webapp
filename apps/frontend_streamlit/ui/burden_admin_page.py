"""Burden distribution admin page for ring-charge pattern management."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

import streamlit as st

from apps.frontend_streamlit.services.api_errors import FrontendApiError
from apps.frontend_streamlit.services.welcome_gateway import WelcomeGateway

LOCAL_TZ = ZoneInfo("Asia/Kolkata")


def _api_error_text(exc: FrontendApiError) -> str:
    parts = [exc.message]
    if exc.error_code:
        parts.append(f"code={exc.error_code}")
    if exc.request_id:
        parts.append(f"request_id={exc.request_id}")
    return " ".join(parts)


def _local_effective_datetime(prefix: str) -> datetime | None:
    st.markdown("#### Effective From")
    col_date, col_time = st.columns(2)
    with col_date:
        from_date = st.date_input("Date", key=f"{prefix}_date")
    with col_time:
        time_str = st.text_input("Time (HH:MM)", value="00:00", key=f"{prefix}_time")
    try:
        from_time = datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        st.error("Invalid time format. Use HH:MM.")
        return None
    return datetime.combine(from_date, from_time, tzinfo=LOCAL_TZ)


class BurdenAdminPage:
    """Streamlit admin interface for editing burden distribution fields."""

    def __init__(self, gateway: WelcomeGateway) -> None:
        self.gateway = gateway

    def _load_context(self) -> dict[str, Any] | None:
        try:
            return self.gateway.get_burden_context(at=datetime.now(LOCAL_TZ))
        except FrontendApiError as exc:
            st.error(f"Could not load burden context: {_api_error_text(exc)}")
            return None
        except Exception as exc:
            st.error(f"Could not load burden context: {exc}")
            return None

    def _handle_submission(
        self,
        *,
        values: dict[str, str | float | None],
        effective_at: datetime,
        expected_snapshot_id: int | None,
    ) -> None:
        payload = {
            "effective_at": effective_at.isoformat(),
            "expected_snapshot_id": expected_snapshot_id,
            "values": values,
        }
        signature = repr(payload)
        if st.session_state.get("burden_last_submit_signature") == signature:
            st.info("This burden submission is already being processed.")
            return
        st.session_state["burden_last_submit_signature"] = signature
        try:
            self.gateway.update_burden_distribution(payload)
        except FrontendApiError as exc:
            st.session_state.pop("burden_last_submit_signature", None)
            if exc.status_code == 409 or exc.error_code == "CONFIG_VERSION_CONFLICT":
                st.warning("The burden configuration changed before your save. Reloading the latest context.")
                st.session_state["burden_conflict_message"] = True
                st.rerun()
            st.error(f"Error updating burden distribution: {_api_error_text(exc)}")
            return
        except Exception as exc:
            st.session_state.pop("burden_last_submit_signature", None)
            st.error(f"Error updating burden distribution: {exc}")
            return

        st.session_state.pop("burden_last_submit_signature", None)
        st.session_state["burden_success_msg"] = "Burden distribution updated successfully."
        st.rerun()

    def render_form(self, context: dict[str, Any]) -> None:
        effective_at = _local_effective_datetime("burden_effective")
        if effective_at is None:
            return

        st.divider()
        st.markdown("#### Set Values for Burden Distribution Fields")
        fields = context.get("fields", [])
        current_values = context.get("values", {}) or {}
        updated_values: dict[str, str | float | None] = {}

        for i in range(0, len(fields), 3):
            cols = st.columns(3)
            for j, field in enumerate(fields[i : i + 3]):
                key = field["key"]
                label = field.get("label") or key.replace("_", " ").title()
                initial = current_values.get(key)
                with cols[j]:
                    if field.get("value_type") == "text":
                        raw_value = st.text_input(
                            label,
                            value=str(initial) if initial is not None else "",
                            key=f"burden_{key}",
                        )
                        updated_values[key] = raw_value if raw_value != "" else None
                    else:
                        null_key = f"burden_{key}_null"
                        is_null = st.checkbox(
                            "No value",
                            value=initial is None,
                            key=null_key,
                        )
                        number_value = st.number_input(
                            label,
                            value=float(initial) if isinstance(initial, (int, float)) else 0.0,
                            step=float(field.get("step") or 0.1),
                            disabled=is_null,
                            key=f"burden_{key}",
                        )
                        updated_values[key] = None if is_null else float(number_value)

        submitted = st.form_submit_button("Save Burden Distribution")
        if submitted:
            self._handle_submission(
                values=updated_values,
                effective_at=effective_at,
                expected_snapshot_id=context.get("snapshot_id"),
            )

    def render_history(self, fields: list[dict[str, Any]]) -> None:
        st.markdown("### Burden Distribution History")
        limit = st.selectbox("Rows per page", [25, 50, 100], index=1, key="burden_history_limit")
        offset = int(st.session_state.get("burden_history_offset", 0))

        try:
            history = self.gateway.list_burden_history(limit=int(limit), offset=offset)
        except FrontendApiError as exc:
            st.warning(f"Could not load burden history: {_api_error_text(exc)}")
            return
        except Exception as exc:
            st.warning(f"Could not load burden history: {exc}")
            return

        total = int(history.get("total", 0))
        items = history.get("items", [])
        if not items:
            st.info("No history available yet.")
            return

        field_keys = [field["key"] for field in fields]
        rows = []
        for item in items:
            row = {
                "snapshot_id": item.get("snapshot_id"),
                "effective_at": item.get("effective_at"),
                "source_type": item.get("source_type"),
                "actor": (item.get("actor") or {}).get("username") or (item.get("actor") or {}).get("user_id"),
                "created_at": item.get("created_at"),
                "delete": False,
            }
            row.update({field: (item.get("values") or {}).get(field) for field in field_keys})
            rows.append(row)

        edited = st.data_editor(
            rows,
            hide_index=True,
            width="stretch",
            column_config={"delete": st.column_config.CheckboxColumn("Delete")},
        )
        delete_ids = [int(row["snapshot_id"]) for row in edited if row.get("delete")]
        confirm = st.checkbox("Confirm history deletion", disabled=not delete_ids, key="burden_delete_confirm")
        if st.button("Delete Selected", disabled=not delete_ids or not confirm):
            try:
                result = self.gateway.delete_burden_history(delete_ids)
                st.success(f"Deleted {result.get('deleted_count', 0)} record(s).")
                st.rerun()
            except FrontendApiError as exc:
                st.error(f"Error deleting records: {_api_error_text(exc)}")
            except Exception as exc:
                st.error(f"Error deleting records: {exc}")

        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("Previous", disabled=offset <= 0, key="burden_prev"):
                st.session_state["burden_history_offset"] = max(0, offset - int(limit))
                st.rerun()
        with c2:
            if st.button("Next", disabled=offset + int(limit) >= total, key="burden_next"):
                st.session_state["burden_history_offset"] = offset + int(limit)
                st.rerun()
        with c3:
            st.caption(f"Showing {offset + 1}-{min(offset + len(items), total)} of {total}")

    def render(self, username: str | None = None) -> None:
        st.subheader("Burden Distribution Admin Panel")
        if st.session_state.get("burden_success_msg"):
            st.success(st.session_state.pop("burden_success_msg"))
        if st.session_state.pop("burden_conflict_message", None):
            st.info("Loaded the latest burden distribution after a version conflict.")

        context = self._load_context()
        if context is None:
            return

        with st.form("burden_update_form", clear_on_submit=False):
            self.render_form(context)
        self.render_history(context.get("fields", []))


def burden_admin_page(username: str | None = None, gateway: WelcomeGateway | None = None) -> None:
    if gateway is None:
        from apps.frontend_streamlit.services.welcome_gateway import get_welcome_gateway

        gateway = get_welcome_gateway()
    BurdenAdminPage(gateway).render(username)
