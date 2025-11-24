import streamlit as st
from datetime import datetime, time
from data.db import Database


class HopperAdminPage:
    """
    Admin interface for managing hopper-to-material mappings with timestamp tracking.
    Allows admins to specify the 'effective from' time for material changes.
    """

    def __init__(self) -> None:
        """Initialize database connection and load materials and hopper mappings."""
        self.db = Database()
        self.materials = self.db.materials
        self.hoppers = self.db.hoppers
        self.hopper_materials = self.db.get_hopper_materials()

    # ---------------------------------------------------
    #  Form Rendering
    # ---------------------------------------------------
    def render_form(self, username: str) -> None:
        """
        Renders the editable hopper-material mapping form with one effective datetime input.
        """
        st.markdown("Assign Materials to Hoppers")

        updated_values = {}

        # ---- Time Selection ----
        st.markdown("###  Effective From Time")
        col_from_date, col_from_time = st.columns(2)

        with col_from_date:
            from_date = st.date_input(
                "Date",
                key="valid_from_date",
                help="Select the date when this change becomes effective."
            )
        with col_from_time:
            time_str = st.text_input(
                "Time (HH:MM)",
                value="00:00",
                key="valid_from_time",
                help="Please enter 24 hours format."
            )

        # Parse text → time object
        try:
            from_time = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            st.error("❌ Invalid time format. Please use HH:MM (e.g., 14:30).")
            return


        # Combine into datetime
        from_dt = datetime.combine(from_date, from_time)

        st.divider()

        # ---- Hopper Material Dropdowns ----
        st.markdown("### ⚙️ Hopper Material Selection")

        for i in range(0, len(self.hoppers), 3):
            cols = st.columns(3)
            for j, hopper in enumerate(self.hoppers[i:i + 3]):
                with cols[j]:
                    current_material = self.hopper_materials.get(hopper, "UNASSIGNED")
                    options = ["UNASSIGNED"] + self.materials

                    selected_material = st.selectbox(
                        label=f"{hopper}",
                        options=options,
                        index=options.index(current_material) if current_material in options else 0,
                        key=f"{hopper}_dropdown"
                    )
                    updated_values[hopper] = selected_material

        # ---- Submit ----
        submitted = st.form_submit_button("💾 Save Changes")

        if submitted:
            if not from_dt:
                st.error("❌ Please specify an effective date and time before saving.")
            else:
                self.handle_submission(updated_values, from_dt, username)

    # ----------------------------------------------------
    #  Submission Handling
    # ----------------------------------------------------
    def handle_submission(self, updated_values: dict[str, str], from_time: datetime, username: str) -> None:
        """
        Updates only changed hopper-material assignments.

        Args:
            updated_values (dict): {hopper: selected_material}
            from_time (datetime): Start timestamp (effective from)
            username (str): Logged-in user's name (modifier)
        """
        errors = False
        changes = 0

        for hopper, new_material in updated_values.items():
            current_material = self.hopper_materials.get(hopper, "UNASSIGNED")

            # Only process if material actually changed
            if new_material != current_material:
                try:
                    self.db.update_hopper_material_with_time(
                        hopper=hopper,
                        material=new_material,
                        from_time=from_time,
                        modifier=username
                    )
                    changes += 1
                except Exception as e:
                    st.error(f"❌ Error updating {hopper}: {e}")
                    errors = True

        if not errors:
            if changes == 0:
                st.info("ℹ️ No changes detected — all materials remain the same.")
            else:
                st.session_state["hopper_success_message"] = f"✅ {changes} hopper(s) updated successfully."
                st.rerun()

    # ----------------------------------------------------
    #  Render Page
    # ----------------------------------------------------
    def render(self, username: str) -> None:
        """
        Renders the full Hopper Admin Page with timestamped updates.
        """
        st.subheader("Hopper Material Mapping")

        # Show success message if available
        if st.session_state.get("hopper_success_message"):
            st.success(st.session_state["hopper_success_message"])
            st.session_state.pop("hopper_success_message", None)

        with st.form(key="hopper_map_form", clear_on_submit=False):
            self.render_form(username)

        # ---- Current Mapping Table ----
        st.markdown("### 📋 Hopper → Material History")

        history = self.db.get_hopper_material_history()

        if history:

            # Add checkbox column
            for row in history:
                row["delete"] = False

            edited = st.data_editor(
                history,
                hide_index=True,
                use_container_width=True,
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
                ]
            )

            # Extract selected IDs
            delete_ids = [row["id"] for row in edited if row["delete"]]

            st.write("---")

            if st.button("🗑️ Delete ", disabled=len(delete_ids) == 0):
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
    """Streamlit entry point for Hopper Admin Page."""
    HopperAdminPage().render(username)
