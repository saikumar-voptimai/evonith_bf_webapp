import streamlit as st
from datetime import datetime
from data.db import Database
from sqlalchemy import text


class ChargeAdminPage:

    TEXT_FIELDS = [
        "COKE_CHARGE_PATTERN",
        "NON_COKE_CHARGE_PATTERN"
    ]

    def __init__(self):
        self.db = Database()
        self.charge_fields = self.db.charge_fields   # loaded from materials.yml
        self.history = []

    # -----------------------------------------------
    # IP extraction (same as hopper logic)
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
    # Get latest charge values for all fields
    # -----------------------------------------------
    def get_current_charge_values(self):
        """Fetch current SCD2 values for all charge fields in ONE query."""
        now = datetime.now()
        return self.db.get_all_current_charge_values(now)

    # -----------------------------------------------
    # RENDER FORM
    # -----------------------------------------------
    def render_form(self, username):

        # Load current values once
        current_values = self.get_current_charge_values()
        updated_values = {}

        # ----------------------- Time selection -----------------------
        st.markdown("#### ⏱️ Effective From")

        col_date, col_time = st.columns(2)

        with col_date:
            from_date = st.date_input("Date", key="charge_from_date")

        with col_time:
            time_str = st.text_input(
                "Time (HH:MM)",
                value="00:00",
                key="charge_from_time"
            )

        try:
            from_time = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            st.error("❌ Invalid time format. Use HH:MM")
            return

        from_dt = datetime.combine(from_date, from_time)

        st.divider()

        # ----------------------- Charge Fields -----------------------
        st.markdown("#### ➕ Set Values for Charge Distribution Fields")

        for i in range(0, len(self.charge_fields), 3):
            cols = st.columns(3)

            for j, field_name in enumerate(self.charge_fields[i:i+3]):
                with cols[j]:

                    initial_value = current_values.get(field_name)

                    # Text fields
                    if field_name in self.TEXT_FIELDS:
                        val = st.text_input(
                            field_name,
                            value=str(initial_value) if initial_value is not None else "",
                            key=f"charge_{field_name}"
                        )

                    # Numeric fields
                    else:
                        val = st.number_input(
                            field_name,
                            value=float(initial_value) if isinstance(initial_value, (int, float)) else 0.0,
                            step=0.1,
                            key=f"charge_{field_name}"
                        )

                    updated_values[field_name] = val

        # ----------------------- SUBMIT -----------------------
        submitted = st.form_submit_button("💾 Save Charge Distribution")

        if submitted:
            ip = self.get_client_ip()
            self.handle_submission(updated_values, current_values, from_dt, username, ip)


    # -----------------------------------------------
    # HANDLE SUBMISSION
    # -----------------------------------------------
    def handle_submission(self, updated_values, current_values, from_dt, username, ip):
        changes = 0
        errors = False

        for field_name, new_value in updated_values.items():
            old_value = current_values.get(field_name)

            # Normalize types (number_input returns float, DB may return int/float)
            if isinstance(old_value, (int, float)) and isinstance(new_value, str):
                try:
                    new_value = float(new_value)
                except:
                    pass

            # Skip if no change
            if str(old_value) == str(new_value):
                continue

            # Update only changed fields
            try:
                self.db.update_charge_field(
                    field_name=field_name,
                    value=new_value,
                    valid_from=from_dt,
                    modifier=username,
                    ip=ip
                )
                changes += 1
            except Exception as e:
                st.error(f"❌ Error updating {field_name}: {e}")
                errors = True

        if not errors:
            if changes == 0:
                st.info("ℹ️ No fields were changed.")
            else:
                st.session_state["charge_success_msg"] = f"✅ Updated {changes} field(s)."
                st.rerun()

    # -----------------------------------------------
    # RENDER HISTORY TABLE
    # -----------------------------------------------
    def render_history(self):
        st.markdown("### 📋 Charge Distribution History")

        history = self.db.get_charge_history()

        if not history:
            st.info("No history available yet.")
            return

        # Add delete checkbox
        for row in history:
            row["delete"] = False

        edited = st.data_editor(
            history,
            hide_index=True,
            width="stretch",
            column_config={
                "delete": st.column_config.CheckboxColumn("Delete"),
                "id": None
            },
            column_order=[
                "id",
                "field_name",
                "value",
                "valid_from",
                "valid_upto",
                "modifier",
                "ip_address",
                "delete"
            ]
        )

        delete_ids = [row["id"] for row in edited if row["delete"]]

        if st.button("🗑️ Delete Selected", disabled=len(delete_ids) == 0):
            try:
                with self.db.engine.begin() as conn:
                    conn.execute(
                        text("DELETE FROM charge_distribution_history WHERE id = ANY(:ids)"),
                        {"ids": delete_ids}
                    )
                st.success(f"🗑️ Deleted {len(delete_ids)} record(s).")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error deleting: {e}")

    # -----------------------------------------------
    # MAIN RENDER
    # -----------------------------------------------
    def render(self, username):
        st.subheader("🔧 Charge Distribution Admin Panel")

        if st.session_state.get("charge_success_msg"):
            st.success(st.session_state["charge_success_msg"])
            st.session_state.pop("charge_success_msg", None)

        with st.form("charge_update_form", clear_on_submit=False):
            self.render_form(username)

        self.render_history()


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
def charge_admin_page(username):
    ChargeAdminPage().render(username)
