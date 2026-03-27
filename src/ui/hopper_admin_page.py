import streamlit as st
from datetime import datetime
from data.db import Database


class HopperAdminPage:
    """
    Admin interface for managing hopper-to-material mappings with timestamp tracking.
    Uses ONLY hopper_material_history (no snapshot table).
    """

    def __init__(self) -> None:
        self.db = Database()
        self.materials = self.db.materials
        self.hoppers = self.db.hoppers

        # ✅ UPDATED: derive current mapping from history table
        self.hopper_materials = self.db.get_current_hopper_materials()

    def get_client_ip(self):
        try:
            forwarded_for = st.context.headers.get("X-Forwarded-For")
            if forwarded_for:
                return forwarded_for.split(",")[0].strip()
            return st.context.headers.get("REMOTE_ADDR", "unknown")
        except Exception:
            return "unknown"

    # ---------------------------------------------------
    # Form Rendering
    # ---------------------------------------------------
    def render_form(self, username: str) -> None:
        st.markdown("## Assign Materials to Hoppers")

        updated_values = {}

        # ---- Effective Time ----
        st.markdown("### ⏱️ Effective From Time")
        col_date, col_time = st.columns(2)

        with col_date:
            from_date = st.date_input("Date")

        with col_time:
            time_str = st.text_input("Time (HH:MM)", value="00:00")

        try:
            from_time = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            st.error("❌ Invalid time format. Use HH:MM.")
            return

        from_dt = datetime.combine(from_date, from_time)
        st.divider()

        # ---- Hopper Material Selection ----
        st.markdown("### ⚙️ Hopper Material Selection")

        for i in range(0, len(self.hoppers), 3):
            cols = st.columns(3)
            for j, hopper in enumerate(self.hoppers[i:i + 3]):
                with cols[j]:
                    current_material = self.hopper_materials.get(hopper, "UNASSIGNED")
                    options = ["UNASSIGNED"] + self.materials

                    selected_material = st.selectbox(
                        hopper,
                        options,
                        index=options.index(current_material)
                        if current_material in options else 0,
                        key=f"{hopper}_dropdown"
                    )
                    updated_values[hopper] = selected_material

        submitted = st.form_submit_button("💾 Save Changes")

        if submitted:
            ip = self.get_client_ip()
            self.handle_submission(updated_values, from_dt, username, ip)

    # ----------------------------------------------------
    # Submission Handling
    # ----------------------------------------------------
    def handle_submission(
        self,
        updated_values: dict[str, str],
        from_time: datetime,
        username: str,
        ip_address: str,
    ) -> None:

        changes = 0
        errors = False

        for hopper, new_material in updated_values.items():
            current_material = self.hopper_materials.get(hopper, "UNASSIGNED")

            if new_material != current_material:
                try:
                    self.db.update_hopper_material_with_time(
                        hopper=hopper,
                        material=new_material,
                        from_time=from_time,
                        modifier=username,
                        ip_address=ip_address,
                    )
                    changes += 1
                except Exception as e:
                    st.error(f"❌ Error updating {hopper}: {e}")
                    errors = True

        if not errors:
            if changes == 0:
                st.info("ℹ️ No changes detected.")
            else:
                st.session_state["hopper_success_message"] = (
                    f"✅ {changes} hopper(s) updated successfully."
                )
                st.rerun()

    # ----------------------------------------------------
    # Render Page
    # ----------------------------------------------------
    def render(self, username: str) -> None:
        st.subheader("Hopper Material Mapping")

        if st.session_state.get("hopper_success_message"):
            st.success(st.session_state.pop("hopper_success_message"))

        with st.form("hopper_map_form"):
            self.render_form(username)

        # ---- History Table ----
        st.markdown("### 📋 Hopper → Material History")

        history = self.db.get_hopper_material_history()

        if history:
            for row in history:
                row["delete"] = False

            edited = st.data_editor(
                history,
                hide_index=True,
                column_config={
                    "delete": st.column_config.CheckboxColumn("Delete"),
                    "id": None,
                },
                column_order=[
                    "id",
                    "hopper",
                    "material",
                    "valid_from",
                    "valid_upto",
                    "modifier",
                    "ip_address",
                ],
            )

            delete_ids = [r["id"] for r in edited if r["delete"]]

            if st.button("🗑️ Delete", disabled=not delete_ids):
                try:
                    self.db.delete_hopper_material_history(delete_ids)
                    st.success(f"🗑️ Deleted {len(delete_ids)} record(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting records: {e}")
        else:
            st.info("No hopper-material history found.")


# -------------------------------
# Entry Point
# -------------------------------
def hopper_admin_page(username: str) -> None:
    HopperAdminPage().render(username)
