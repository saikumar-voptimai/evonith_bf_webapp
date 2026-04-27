"""Burden distribution admin page for ring-charge pattern management.

Allows supervisors and admins to update burden distribution parameters
(charge patterns, coke/non-coke angles, portions) stored via
:class:`~data.db.Database` SCD Type-2 logic.
"""

# --------- burden_admin_page.py ----------
from datetime import datetime

import streamlit as st

from data.db import Database


class BurdenAdminPage:
    """Streamlit admin interface for editing burden distribution fields.

    Attributes:
        db:            :class:`~data.db.Database` instance.
        burden_fields: Dict of burden field metadata loaded from ``materials.yml``.
        history:       In-session list of recently applied changes.
    """

    TEXT_FIELDS = [
        "COKE_CHARGE_PATTERN",
        "NON_COKE_CHARGE_PATTERN",
        "BURDEN_CHANGING_PURPOSE",
    ]

    def __init__(self) -> None:
        """Initialise with a fresh database connection and current burden state."""
        self.db = Database()
        self.burden_fields = self.db.burden_fields  # loaded from materials.yml
        self.history = []

    # -----------------------------------------------
    # IP extraction
    # -----------------------------------------------
    def get_client_ip(self):
        try:
            forwarded_for = st.context.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()

            return st.context.headers.get("REMOTE_ADDR", "unknown")
        except:
            return "unknown"

    # -----------------------------------------------
    # Get latest burden values
    # -----------------------------------------------
    def get_current_burden_values(self):
        now = datetime.now()
        return self.db.get_all_current_burden_values(now)

    # -----------------------------------------------
    # RENDER FORM
    # -----------------------------------------------
    def render_form(self, username):

        current_values = self.get_current_burden_values()
        updated_values = {}

        # ----------------------- Time selection -----------------------
        st.markdown("#### ⏱️ Effective From")

        col_date, col_time = st.columns(2)

        with col_date:
            from_date = st.date_input("Date", key="burden_from_date")

        with col_time:
            time_str = st.text_input(
                "Time (HH:MM)", value="00:00", key="burden_from_time"
            )

        try:
            from_time = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            st.error(" Invalid time format. Use HH:MM")
            return

        from_dt = datetime.combine(from_date, from_time)

        st.divider()

        # ----------------------- Burden Fields -----------------------
        st.markdown("#### Set Values for Burden Distribution Fields")

        for i in range(0, len(self.burden_fields), 3):
            cols = st.columns(3)

            for j, field_name in enumerate(self.burden_fields[i : i + 3]):
                with cols[j]:

                    initial_value = current_values.get(field_name)

                    if field_name in self.TEXT_FIELDS:
                        val = st.text_input(
                            field_name,
                            value=(
                                str(initial_value) if initial_value is not None else ""
                            ),
                            key=f"burden_{field_name}",
                        )
                    else:
                        val = st.number_input(
                            field_name,
                            value=(
                                float(initial_value)
                                if isinstance(initial_value, (int, float))
                                else 0.0
                            ),
                            step=0.1,
                            key=f"burden_{field_name}",
                        )

                    updated_values[field_name] = val

        submitted = st.form_submit_button("💾 Save Burden Distribution")

        if submitted:
            ip = self.get_client_ip()
            self.handle_submission(
                updated_values, current_values, from_dt, username, ip
            )

    # -----------------------------------------------
    # HANDLE SUBMISSION
    # -----------------------------------------------
    def handle_submission(self, updated_values, current_values, from_dt, username, ip):
        changes = 0
        errors = False

        for field_name, new_value in updated_values.items():
            old_value = current_values.get(field_name)

            if isinstance(old_value, (int, float)) and isinstance(new_value, str):
                try:
                    new_value = float(new_value)
                except:
                    pass

            if str(old_value) == str(new_value):
                continue

            try:
                self.db.update_burden_field(
                    field_name=field_name,
                    value=new_value,
                    valid_from=from_dt,
                    modifier=username,
                    ip=ip,
                )
                changes += 1
            except Exception as e:
                st.error(f"❌ Error updating {field_name}: {e}")
                errors = True

        if not errors:
            if changes == 0:
                st.info("ℹ️ No fields were changed.")
            else:
                st.session_state["burden_success_msg"] = (
                    f"✅ Updated {changes} field(s)."
                )
                st.rerun()

    # -----------------------------------------------
    # RENDER HISTORY
    # -----------------------------------------------
    def render_history(self):
        st.markdown("### 📋 Burden Distribution History")

        history = self.db.get_burden_history()

        if not history:
            st.info("No history available yet.")
            return

        for row in history:
            row["delete"] = False

        edited = st.data_editor(
            history,
            hide_index=True,
            width="stretch",
            column_config={
                "delete": st.column_config.CheckboxColumn("Delete"),
                "id": None,
            },
            column_order=[
                "id",
                "field_name",
                "value",
                "valid_from",
                "valid_upto",
                "modifier",
                "ip_address",
                "delete",
            ],
        )

        delete_ids = [row["id"] for row in edited if row["delete"]]

        if st.button("🗑️ Delete Selected", disabled=len(delete_ids) == 0):
            try:
                self.db.delete_burden_history(delete_ids)
                st.success(f"🗑️ Deleted {len(delete_ids)} record(s).")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error deleting: {e}")

    # -----------------------------------------------
    # MAIN RENDER
    # -----------------------------------------------
    def render(self, username):
        st.subheader("🔧 Burden Distribution Admin Panel")

        if st.session_state.get("burden_success_msg"):
            st.success(st.session_state["burden_success_msg"])
            st.session_state.pop("burden_success_msg", None)

        with st.form("burden_update_form", clear_on_submit=False):
            self.render_form(username)

        self.render_history()


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
def burden_admin_page(username):
    BurdenAdminPage().render(username)
