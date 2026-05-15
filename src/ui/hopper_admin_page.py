"""Hopper-to-material mapping admin interface.

Allows supervisors and admins to assign raw materials to hoppers with
a user-specified effective timestamp.  All changes are written via
:meth:`~data.db.HopperConfigService.update_hopper_material_with_time` which appends
a new SCD Type-2 history row.
"""

from datetime import datetime

import streamlit as st

from data.db import HopperConfigService


class HopperAdminPage:
    """Streamlit admin interface for hopper-to-material mapping management.

    Derives the current mapping entirely from the ``hopper_material_history``
    table (no separate snapshot table required).

    Attributes:
        db:              :class:`~data.db.HopperConfigService` instance.
        materials:       List of material names available for assignment.
        hoppers:         List of hopper identifiers.
        hopper_materials: Dict of ``{hopper_id: material_name}`` (current mapping).
    """

    def __init__(self) -> None:
        """Initialise with a fresh database connection and current hopper mapping."""
        self.db = HopperConfigService()
        self.materials = self.db.materials
        self.hoppers = self.db.hoppers

        # \u2705 UPDATED: derive current mapping from history table
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
            for j, hopper in enumerate(self.hoppers[i : i + 3]):
                with cols[j]:
                    current_material = self.hopper_materials.get(hopper, "UNASSIGNED")
                    options = ["UNASSIGNED"] + self.materials

                    selected_material = st.selectbox(
                        self.db.hopper_display_names.get(hopper, hopper),
                        options,
                        index=(
                            options.index(current_material)
                            if current_material in options
                            else 0
                        ),
                        key=f"{hopper}_dropdown",
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

        changed_values = {}
        for hopper, new_material in updated_values.items():
            current_material = self.hopper_materials.get(hopper, "UNASSIGNED")

            if new_material != current_material:
                changed_values[hopper] = new_material
                changes += 1

        if changed_values:
            try:
                self.db.update_hopper_materials_snapshot(
                    hopper_materials=changed_values,
                    from_time=from_time,
                    modifier=username,
                    ip_address=ip_address,
                )
            except Exception as e:
                st.error(f"Error updating hopper mapping: {e}")
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
                    "date_time",
                    *self.hoppers,
                    "source_type",
                    "user_modified",
                    "ip_address",
                    "delete",
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
