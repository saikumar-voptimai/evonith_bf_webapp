"""Hopper-to-material mapping admin interface."""

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
    st.markdown("### Effective From Time")
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


class HopperAdminPage:
    """Streamlit admin interface for hopper-to-material mapping management."""

    def __init__(self, gateway: WelcomeGateway) -> None:
        self.gateway = gateway

    def _load_context(self) -> dict[str, Any] | None:
        try:
            return self.gateway.get_hopper_context(at=datetime.now(LOCAL_TZ))
        except FrontendApiError as exc:
            st.error(f"Could not load hopper mapping context: {_api_error_text(exc)}")
            return None
        except Exception as exc:
            st.error(f"Could not load hopper mapping context: {exc}")
            return None

    def _handle_submission(
        self,
        *,
        assignments: dict[str, str | None],
        effective_at: datetime,
        expected_snapshot_id: int | None,
    ) -> None:
        payload = {
            "effective_at": effective_at.isoformat(),
            "expected_snapshot_id": expected_snapshot_id,
            "assignments": assignments,
        }
        signature = repr(payload)
        if st.session_state.get("hopper_last_submit_signature") == signature:
            st.info("This hopper mapping submission is already being processed.")
            return
        st.session_state["hopper_last_submit_signature"] = signature
        try:
            self.gateway.update_hopper_mapping(payload)
        except FrontendApiError as exc:
            st.session_state.pop("hopper_last_submit_signature", None)
            if exc.status_code == 409 or exc.error_code == "CONFIG_VERSION_CONFLICT":
                st.warning("The hopper configuration changed before your save. Reloading the latest context.")
                st.session_state["hopper_conflict_message"] = True
                st.rerun()
            st.error(f"Error updating hopper mapping: {_api_error_text(exc)}")
            return
        except Exception as exc:
            st.session_state.pop("hopper_last_submit_signature", None)
            st.error(f"Error updating hopper mapping: {exc}")
            return

        st.session_state.pop("hopper_last_submit_signature", None)
        st.session_state["hopper_success_message"] = "Hopper mapping updated successfully."
        st.rerun()

    def render_form(self, context: dict[str, Any]) -> None:
        st.markdown("## Assign Materials to Hoppers")
        effective_at = _local_effective_datetime("hopper_effective")
        if effective_at is None:
            return

        st.divider()
        st.markdown("### Hopper Material Selection")

        materials = context.get("materials", [])
        hoppers = context.get("hoppers", [])
        assignments = context.get("assignments", {}) or {}
        material_options: list[str | None] = [None] + [item["code"] for item in materials]
        material_labels = {None: "Unassigned"}
        material_labels.update({item["code"]: item["display_name"] for item in materials})

        updated_values: dict[str, str | None] = {}
        for i in range(0, len(hoppers), 3):
            cols = st.columns(3)
            for j, hopper in enumerate(hoppers[i : i + 3]):
                hopper_code = hopper["code"]
                with cols[j]:
                    current = assignments.get(hopper_code)
                    index = material_options.index(current) if current in material_options else 0
                    updated_values[hopper_code] = st.selectbox(
                        hopper.get("display_name") or hopper_code,
                        material_options,
                        index=index,
                        format_func=lambda code: material_labels.get(code, str(code)),
                        key=f"{hopper_code}_dropdown",
                    )

        submitted = st.form_submit_button("Save Changes")
        if submitted:
            self._handle_submission(
                assignments=updated_values,
                effective_at=effective_at,
                expected_snapshot_id=context.get("snapshot_id"),
            )

    def render_history(self, hoppers: list[dict[str, Any]]) -> None:
        st.markdown("### Hopper -> Material History")
        limit = st.selectbox("Rows per page", [25, 50, 100], index=1, key="hopper_history_limit")
        offset = int(st.session_state.get("hopper_history_offset", 0))

        try:
            history = self.gateway.list_hopper_history(limit=int(limit), offset=offset)
        except FrontendApiError as exc:
            st.warning(f"Could not load hopper history: {_api_error_text(exc)}")
            return
        except Exception as exc:
            st.warning(f"Could not load hopper history: {exc}")
            return

        total = int(history.get("total", 0))
        items = history.get("items", [])
        if not items:
            st.info("No hopper-material history found.")
            return

        hopper_codes = [item["code"] for item in hoppers]
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
            row.update({hopper: (item.get("assignments") or {}).get(hopper) for hopper in hopper_codes})
            rows.append(row)

        edited = st.data_editor(
            rows,
            hide_index=True,
            width="stretch",
            column_config={"delete": st.column_config.CheckboxColumn("Delete")},
        )
        delete_ids = [int(row["snapshot_id"]) for row in edited if row.get("delete")]
        confirm = st.checkbox("Confirm history deletion", disabled=not delete_ids, key="hopper_delete_confirm")
        if st.button("Delete Selected", disabled=not delete_ids or not confirm):
            try:
                result = self.gateway.delete_hopper_history(delete_ids)
                st.success(f"Deleted {result.get('deleted_count', 0)} record(s).")
                st.rerun()
            except FrontendApiError as exc:
                st.error(f"Error deleting records: {_api_error_text(exc)}")
            except Exception as exc:
                st.error(f"Error deleting records: {exc}")

        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("Previous", disabled=offset <= 0):
                st.session_state["hopper_history_offset"] = max(0, offset - int(limit))
                st.rerun()
        with c2:
            if st.button("Next", disabled=offset + int(limit) >= total):
                st.session_state["hopper_history_offset"] = offset + int(limit)
                st.rerun()
        with c3:
            st.caption(f"Showing {offset + 1}-{min(offset + len(items), total)} of {total}")

    def render(self, username: str | None = None) -> None:
        st.subheader("Hopper Material Mapping")
        if st.session_state.get("hopper_success_message"):
            st.success(st.session_state.pop("hopper_success_message"))
        if st.session_state.pop("hopper_conflict_message", None):
            st.info("Loaded the latest hopper mapping after a version conflict.")

        context = self._load_context()
        if context is None:
            return

        with st.form("hopper_map_form"):
            self.render_form(context)
        self.render_history(context.get("hoppers", []))


def hopper_admin_page(username: str | None = None, gateway: WelcomeGateway | None = None) -> None:
    if gateway is None:
        from apps.frontend_streamlit.services.welcome_gateway import get_welcome_gateway

        gateway = get_welcome_gateway()
    HopperAdminPage(gateway).render(username)
